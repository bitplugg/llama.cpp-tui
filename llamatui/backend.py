"""Слой над движками генерации.

Поддерживаемые режимы (--engine):
  auto         — цепочка ollama → llama-server → llama-cli (первый доступный)
  llama        — локальный llama.cpp: llama-server (OpenAI-совместимый API),
                 при неудаче fallback на llama-cli
  llama-cli    — только llama-cli (разовый процесс на сообщение)
  ollama       — Ollama-демон (http://localhost:11434, /api/chat со стримингом NDJSON);
                 вместо GGUF-файла передаётся имя модели («qwen2.5:7b»)
  openai       — любой OpenAI-совместимый endpoint (LM Studio, vLLM, llama-server
                 отдалённый…) — задай base_url и api_key в конфиге

Генерация идёт стримингом; метрики (токены/с, TTFT, время) считаются здесь —
rich и tqdm рисуют это снаружи.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from .downloader import find_binary, llama_dir


@dataclass
class GenStats:
    """Метрики одного ответа."""
    tokens_out: int = 0
    prompt_tokens: int = 0
    completion_tokens_api: int | None = None
    t_start: float = 0.0
    t_first_token: float | None = None
    t_end: float | None = None
    stopped_by_eos: bool = False

    @property
    def elapsed(self) -> float:
        return (self.t_end or time.monotonic()) - self.t_start

    @property
    def ttft(self) -> float | None:
        if self.t_first_token is None:
            return None
        return self.t_first_token - self.t_start

    @property
    def tok_per_sec(self) -> float:
        gen_time = (self.t_end or time.monotonic()) - (self.t_first_token or self.t_start)
        return self.tokens_out / gen_time if gen_time > 0.01 else 0.0


@dataclass
class ChatMessage:
    role: str
    content: str


# ---------------------------------------------------------------------------
# llama-server (предпочтительный путь)
# ---------------------------------------------------------------------------

class LlamaServerError(RuntimeError):
    pass


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class ServerBackend:
    """Держит llama-server на localhost и ходит в /v1/chat/completions."""

    def __init__(self, model: Path, *, n_ctx: int = 8192, n_gpu_layers: int | None = None,
                 extra_args: list[str] | None = None, temp: float = 0.7):
        exe = find_binary("llama-server") or find_binary("llama-server.exe")
        if exe is None:
            raise LlamaServerError("llama-server не найден")
        self.model = Path(model)
        self.port = _free_port()
        self.temp = temp
        cmd = [
            str(exe),
            "-m", str(self.model),
            "--host", "127.0.0.1",
            "--port", str(self.port),
            "-c", str(n_ctx),
            "--no-webui",
        ]
        # --temp задаём на запросе, но базовый тоже полезно
        if n_gpu_layers is not None:
            cmd += ["-ngl", str(n_gpu_layers)]
        if extra_args:
            cmd += extra_args
        self._log_path = Path(llama_dir() / "server.log")
        env = dict(os.environ)
        libdir = str(exe.parent)
        for var in ("LD_LIBRARY_PATH", "DYLD_LIBRARY_PATH", "PATH"):
            env[var] = libdir + os.pathsep + env.get(var, "")
        try:
            self._log = open(self._log_path, "ab")
            self.proc = subprocess.Popen(
                cmd, stdout=self._log, stderr=subprocess.STDOUT,
                cwd=str(exe.parent), env=env,
            )
        except OSError as exc:
            raise LlamaServerError(f"не удалось запустить llama-server: {exc}") from exc
        self._wait_ready(timeout=float(os.environ.get("LLAMATUI_BOOT_TIMEOUT", "180")))

    # -- запуск/ожидание ------------------------------------------------------

    def _wait_ready(self, timeout: float) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                tail = ""
                try:
                    tail = self._log_path.read_text(errors="replace")[-1500:]
                except OSError:
                    pass
                raise LlamaServerError(
                    f"llama-server завершился с кодом {self.proc.returncode} при старте.\n"
                    f"Лог: {self._log_path}\n{tail}"
                )
            try:
                with self._get("/health", timeout=2) as r:
                    if r.status == 200:
                        return
            except Exception:  # noqa: BLE001
                time.sleep(0.4)
        raise LlamaServerError(f"llama-server не поднялся за {timeout:.0f} c (см. {self._log_path})")

    def _url(self, path: str) -> str:
        return f"http://127.0.0.1:{self.port}{path}"

    def _get(self, path: str, timeout: float = 5):
        req = urllib.request.Request(self._url(path), headers={"Accept": "application/json"})
        return urllib.request.urlopen(req, timeout=timeout)

    # -- генерация ------------------------------------------------------------

    def chat_stream(self, messages: list[ChatMessage], *, temp: float | None = None,
                    max_tokens: int = 1024, top_p: float = 0.95) -> tuple[Iterator[str], GenStats]:
        stats = GenStats(t_start=time.monotonic())
        body = json.dumps({
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": self.temp if temp is None else temp,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "stream": True,
        }).encode()
        req = urllib.request.Request(
            self._url("/v1/chat/completions"), data=body,
            headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
        )
        resp = urllib.request.urlopen(req, timeout=600)

        first = [True]  # мутабельно для _handle

        def gen() -> Iterator[str]:
            buf = ""  # SSE-событие может прийти частями — буферим по \n
            try:
                for raw in resp:
                    buf += raw.decode("utf-8", errors="replace")
                    while "\n" in buf:
                        line, buf = buf.split("\n", 1)
                        yield from _handle(line)
                    if buf.startswith("data: [DONE]"):
                        break
            finally:
                stats.t_end = time.monotonic()
                resp.close()

        def _handle(line: str) -> Iterator[str]:
            line = line.strip()
            if not line.startswith("data:"):
                return
            payload = line[5:].strip()
            if payload == "[DONE]":
                return
            try:
                obj = json.loads(payload)
            except json.JSONDecodeError:
                return
            if usage := obj.get("usage"):
                stats.prompt_tokens = usage.get("prompt_tokens", 0)
                stats.completion_tokens_api = usage.get("completion_tokens")
            for choice in obj.get("choices", []):
                delta = (choice.get("delta") or {}).get("content") or ""
                if choice.get("finish_reason") == "eos_token":
                    stats.stopped_by_eos = True
                if delta:
                    if first[0]:
                        stats.t_first_token = time.monotonic()
                        first[0] = False
                    stats.tokens_out += 1
                    yield delta
        return gen(), stats

    def tokenize_count(self, text: str) -> int | None:
        try:
            body = json.dumps({"content": text}).encode()
            req = urllib.request.Request(self._url("/tokenize"), data=body,
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=5) as r:
                return len(json.load(r).get("tokens", []))
        except Exception:  # noqa: BLE001
            return None

    def stop(self) -> None:
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        try:
            self._log.close()
        except Exception:  # noqa: BLE001
            pass


# ---------------------------------------------------------------------------
# llama-cli (fallback: разовый процесс на сообщение, чистый stdout)
# ---------------------------------------------------------------------------

class CliBackend:
    """Гоняет `llama-cli --no-display-prompt -st ...` и читает stdout по строкам.

    Контекст держим сами: сериализуем историю в промпт через chat template?
    Нет — проще: llama-cli >= b4900 умеет `-cn` (конверсация) и `--reverse-prompt`,
    но это интерактив. Для надёжности передаём историю как единый user-промпт
    с явной разметкой ролей — работает с любой моделью.
    """

    def __init__(self, model: Path, *, n_ctx: int = 8192, n_gpu_layers: int | None = None,
                 extra_args: list[str] | None = None, temp: float = 0.7):
        exe = find_binary("llama-cli") or find_binary("cli")
        if exe is None:
            raise LlamaServerError("ни llama-server, ни llama-cli не найдены")
        self.exe = exe
        self.model = Path(model)
        self.n_ctx = n_ctx
        self.temp = temp
        self.extra = list(extra_args or [])
        self.n_gpu_layers = n_gpu_layers

    @staticmethod
    def render_history(messages: list[ChatMessage]) -> str:
        parts = []
        for m in messages:
            tag = {"system": "System", "user": "User", "assistant": "Assistant"}.get(m.role, m.role.title())
            parts.append(f"{tag}: {m.content}")
        parts.append("Assistant:")
        return "\n\n".join(parts)

    def chat_stream(self, messages: list[ChatMessage], *, temp: float | None = None,
                    max_tokens: int = 1024, top_p: float = 0.95) -> tuple[Iterator[str], GenStats]:
        stats = GenStats(t_start=time.monotonic())
        cmd = [
            str(self.exe), "-m", str(self.model),
            "--temp", str(self.temp if temp is None else temp),
            "--top-p", str(top_p),
            "-n", str(max_tokens),
            "--no-display-prompt", "--no-color", "--quiet", "-st",
            "--ctx-size", str(self.n_ctx),
        ]
        if self.n_gpu_layers is not None:
            cmd += ["-ngl", str(self.n_gpu_layers)]
        cmd += self.extra + ["--prompt", self.render_history(messages)]
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1,
        )

        def gen() -> Iterator[str]:
            first = True
            assert proc.stdout is not None
            try:
                for chunk in iter(lambda: proc.stdout.read(16), ""):
                    if first and chunk:
                        stats.t_first_token = time.monotonic()
                        first = False
                    stats.tokens_out += max(1, len(chunk) // 4)  # эвристика ~4 chars/token
                    yield chunk
            finally:
                stats.t_end = time.monotonic()
                proc.wait()

        return gen(), stats

    def stop(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Ollama (локальный демон http://localhost:11434)
# ---------------------------------------------------------------------------

class OpenAICompatBackend:
    """Клиент к любому OpenAI-совместимому /v1/chat/completions со стримингом SSE.

    Используется и для режима `openai` (LM Studio, vLLM, удалённый llama-server),
    и как общий SSE-парсер.
    """

    kind = "openai"

    def __init__(self, base_url: str, model: str, *, api_key: str | None = None,
                 temp: float = 0.7):
        self.base = base_url.rstrip("/")
        if not self.base.startswith("http"):
            self.base = "http://" + self.base
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.temp = temp

    def _headers(self) -> dict:
        h = {"Content-Type": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    def ping(self, timeout: float = 3.0) -> bool:
        try:
            req = urllib.request.Request(self.base + "/models", headers=self._headers())
            with urllib.request.urlopen(req, timeout=timeout):
                return True
        except Exception:  # noqa: BLE001
            return False

    def chat_stream(self, messages: list[ChatMessage], *, temp: float | None = None,
                    max_tokens: int = 1024, top_p: float = 0.95) -> tuple[Iterator[str], GenStats]:
        stats = GenStats(t_start=time.monotonic())
        body = json.dumps({
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": self.temp if temp is None else temp,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "stream": True,
        }).encode()
        req = urllib.request.Request(self.base + "/v1/chat/completions", data=body,
                                     headers=self._headers())
        resp = urllib.request.urlopen(req, timeout=600)
        first = [True]

        def gen() -> Iterator[str]:
            buf = ""
            try:
                for raw in resp:
                    buf += raw.decode("utf-8", errors="replace")
                    while "\n" in buf:
                        line, buf = buf.split("\n", 1)
                        yield from _handle(line)
            finally:
                stats.t_end = time.monotonic()
                resp.close()

        def _handle(line: str) -> Iterator[str]:
            line = line.strip()
            if not line.startswith("data:"):
                return
            payload = line[5:].strip()
            if payload == "[DONE]":
                return
            try:
                obj = json.loads(payload)
            except json.JSONDecodeError:
                return
            if usage := obj.get("usage"):
                stats.prompt_tokens = usage.get("prompt_tokens", 0) or stats.prompt_tokens
                stats.completion_tokens_api = usage.get("completion_tokens")
            for choice in obj.get("choices", []):
                delta = (choice.get("delta") or {}).get("content") or \
                        (choice.get("message") or {}).get("content") or ""
                if choice.get("finish_reason") in ("eos_token", "stop"):
                    stats.stopped_by_eos = True
                if delta:
                    if first[0]:
                        stats.t_first_token = time.monotonic()
                        first[0] = False
                    stats.tokens_out += 1
                    yield delta
        return gen(), stats

    def stop(self) -> None:
        pass


class OllamaBackend(OpenAICompatBackend):
    """Ollama-демон: нативный /api/chat (NDJSON-стриминг) + список моделей.

    Модель задаётся именем («qwen2.5:7b»), GGUF-файлы не нужны — их тянет ollama.
    """

    kind = "ollama"

    def __init__(self, base_url: str = "", model: str = "llama3.2", *, temp: float = 0.7):
        super().__init__(base_url or os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434"),
                         model, temp=temp)
        self.ensure_up()

    # -- запуск демона по необходимости ----------------------------------------

    @staticmethod
    def _which_ollama() -> str | None:
        import shutil
        return shutil.which("ollama")

    def ensure_up(self) -> None:
        if self.ping(timeout=1.5):
            return
        exe = self._which_ollama()
        if exe is None:
            raise LlamaServerError(
                f"Ollama не отвечает на {self.base} и бинарник `ollama` не найден в PATH.\n"
                "Поставь: https://ollama.com/download или переключись: llamatui --engine llama")
        try:
            subprocess.Popen([exe, "serve"],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except OSError as exc:
            raise LlamaServerError(f"не удалось запустить `ollama serve`: {exc}") from exc
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            if self.ping(timeout=1.5):
                return
            time.sleep(0.4)
        raise LlamaServerError(f"`ollama serve` запустился, но {self.base} не отвечает за 20 c")

    def model_exists(self) -> bool:
        return self.model in self.list_models()

    def list_models(self) -> list[str]:
        try:
            req = urllib.request.Request(self.base + "/api/tags", headers=self._headers())
            with urllib.request.urlopen(req, timeout=5) as r:
                data = json.load(r)
            return [m.get("name", "") for m in data.get("models", [])]
        except Exception:  # noqa: BLE001
            return []

    # -- генерация через нативный /api/chat -------------------------------------

    def chat_stream(self, messages: list[ChatMessage], *, temp: float | None = None,
                    max_tokens: int = 1024, top_p: float = 0.95) -> tuple[Iterator[str], GenStats]:
        stats = GenStats(t_start=time.monotonic())
        body = json.dumps({
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "stream": True,
            "options": {
                "temperature": self.temp if temp is None else temp,
                "top_p": top_p,
                "num_predict": max_tokens,
                "num_ctx": int(os.environ.get("LLAMATUI_OLLAMA_CTX", "8192")),
            },
        }).encode()
        req = urllib.request.Request(self.base + "/api/chat", data=body,
                                     headers={"Content-Type": "application/json"})
        resp = urllib.request.urlopen(req, timeout=600)
        first = [True]

        def gen() -> Iterator[str]:
            try:
                for raw in resp:
                    line = raw.decode("utf-8", errors="replace").strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    piece = (obj.get("message") or {}).get("content") or ""
                    if obj.get("done"):
                        stats.prompt_tokens = obj.get("prompt_eval_count", stats.prompt_tokens)
                        stats.completion_tokens_api = obj.get("eval_count")
                        stats.stopped_by_eos = obj.get("done_reason") == "stop"
                    if piece:
                        if first[0]:
                            stats.t_first_token = time.monotonic()
                            first[0] = False
                        stats.tokens_out += 1
                        yield piece
            finally:
                stats.t_end = time.monotonic()
                resp.close()
        return gen(), stats

    def count_tokens(self, text: str) -> int | None:
        # /api/generate с neg_prompt не генерирует, но возвращает prompt_eval_count
        try:
            body = json.dumps({"model": self.model, "prompt": text, "stream": False,
                               "raw": True, "options": {"seed": -1}}).encode()
            req = urllib.request.Request(self.base + "/api/generate", data=body,
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.load(r).get("prompt_eval_count")
        except Exception:  # noqa: BLE001
            return None

    def stop(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Единая точка входа
# ---------------------------------------------------------------------------

@dataclass
class Backend:
    impl: ServerBackend | CliBackend | OllamaBackend | OpenAICompatBackend
    kind: str  # "server" | "cli" | "ollama" | "openai"
    model_name: str
    started_at: float = field(default_factory=time.time)

    def chat_stream(self, messages, **kw):
        return self.impl.chat_stream(messages, **kw)

    def count_tokens(self, text: str) -> int | None:
        if isinstance(self.impl, (ServerBackend, OllamaBackend)):
            return self.impl.count_tokens(text) if isinstance(self.impl, OllamaBackend) \
                else self.impl.tokenize_count(text)
        return None  # грубая оценка снаружи

    def shutdown(self) -> None:
        self.impl.stop()


ENGINES = ("auto", "llama", "llama-cli", "llama-server", "ollama", "openai")


def create_backend(model: Path | str, *, engine: str = "auto", n_ctx: int = 8192,
                   n_gpu_layers: int | None = None, extra_args: list[str] | None = None,
                   temp: float = 0.7, base_url: str = "", api_key: str | None = None) -> Backend:
    """Фабрика движков.

    engine=auto: ollama (если модель уже подтянута/демон жив) → llama-server → llama-cli.
    Для ollama/openai `model` — это имя модели, а не путь к GGUF.
    """
    kwargs = dict(n_ctx=n_ctx, n_gpu_layers=n_gpu_layers, extra_args=extra_args, temp=temp)
    errors: list[str] = []

    def try_llama_server() -> Backend | None:
        if find_binary("llama-server"):
            try:
                return Backend(ServerBackend(Path(model), **kwargs), "server", Path(model).stem)
            except LlamaServerError as exc:
                errors.append(f"llama-server: {exc}")
        else:
            errors.append("llama-server: бинарник не найден (llamatui setup)")
        return None

    def try_llama_cli() -> Backend | None:
        if find_binary("llama-cli"):
            try:
                return Backend(CliBackend(Path(model), **kwargs), "cli", Path(model).stem)
            except LlamaServerError as exc:
                errors.append(f"llama-cli: {exc}")
        else:
            errors.append("llama-cli: бинарник не найден (llamatui setup)")
        return None

    def try_ollama() -> Backend | None:
        name = str(model)
        try:
            ob = OllamaBackend(base_url or "", name, temp=temp)
        except LlamaServerError as exc:
            errors.append(f"ollama: {exc}")
            return None
        if not ob.model_exists():
            errors.append(f"ollama: модели «{name}» нет в демоне (ollama pull {name})")
            return None
        return Backend(ob, "ollama", name)

    if engine == "ollama":
        b = try_ollama()
        if b is None:
            raise LlamaServerError("\n".join(errors))
        return b
    if engine == "openai":
        if not base_url:
            raise LlamaServerError("engine=openai: укажи base_url "
                                   "(--base-url http://host:port или в конфиге)")
        ob = OpenAICompatBackend(base_url, str(model), api_key=api_key, temp=temp)
        if not ob.ping():
            raise LlamaServerError(f"OpenAI-эндпоинт {ob.base} не отвечает")
        return Backend(ob, "openai", str(model))
    if engine == "llama-cli":
        b = try_llama_cli()
        if b is None:
            raise LlamaServerError("\n".join(errors))
        return b
    if engine in ("llama", "llama-server"):
        b = try_llama_server() or try_llama_cli()
        if b is None:
            raise LlamaServerError("\n".join(errors))
        return b

    # ---- auto ----
    path_like = str(model).endswith(".gguf") or Path(str(model)).is_file()
    if not path_like:
        b = try_ollama()
        if b:
            return b
    b = try_llama_server() or try_llama_cli()
    if b:
        return b
    if path_like:
        b = try_ollama()
        if b:
            return b
    raise LlamaServerError("\n".join(errors) or "подходящий движок не найден")
