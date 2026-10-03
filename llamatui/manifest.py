"""Генерация Modelfile-манифестов для Ollama (и `ollama create`).

Поддерживаются все валидные инструкции Modelfile v0.13+:
  FROM / PARAMETER / TEMPLATE / SYSTEM / LICENSE / ADAPTER / MESSAGE

PARAMETER'ы и системный промпт подтягиваются из GGUF-метаданных модели,
так что полученный манифест ведёт себя так же, как «голый» GGUF в llama.cpp.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from .gguf import GGUFError, read_gguf_metadata

# GGUF-ключ -> имя параметра для ollama (список по docs Ollama)
GGUF_TO_PARAM = {
    "general.name": None,                    # служебный
    "tokenizer.ggml.model": None,            # служебный
    "general.architecture": None,            # служебный
    "llama.context_length": "num_ctx",
    "tokenizer.ggml.pre": None,
    "tokenizer.ggml.bos_token_id": None,
    "tokenizer.ggml.eos_token_id": None,
    "tokenizer.chat_template": "TEMPLATE",   # идёт отдельной инструкцией
    "general.alignment": None,
    "tokenizer.ggml.tokens": None,
    "tokenizer.ggml.add_bos_token": "add_bos_token",
    "tokenizer.ggml.add_eos_token": "add_eos_token",
    "tokenizer.ggml.padding_token_id": None,
    "tokenizer.ggml.token_type": None,
    "tokenizer.ggml.scores": None,
    "tokenizer.ggml.merges": None,
    "sampler.top_k": "top_k",
    "sampler.top_p": "top_p",
    "sampler.min_p": "min_p",
    "sampler.temp": "temperature",
    "sampler.repeat_penalty": "repeat_penalty",
    "sampler.repeat_last_n": "repeat_last_n",
    "sampler.presence_penalty": "presence_penalty",
    "sampler.frequency_penalty": "frequency_penalty",
    "sampler.seed": "seed",
    "vocab.special": None,
    "tokenizer.ggml.utf8_incremental_detokenize": None,
    "tokenizer.ggml.byte_fallback": None,
}


@dataclass
class Manifest:
    """Модель данных Modelfile."""

    from_: str = ""                       # путь к .gguf или имя base-модели
    template: str = ""                    # chat template (Jinja2)
    system: str = ""
    license: str = ""
    adapter: str = ""                     # LoRA-адаптер (.gguf)
    parameters: dict = field(default_factory=dict)
    messages: list[tuple[str, str]] = field(default_factory=list)  # few-shot (role, content)

    def render(self) -> str:
        lines: list[str] = []
        if self.from_:
            lines.append(f"FROM {self.from_}")
        if self.license:
            lines.append("LICENSE \"\"\"")
            lines.append(self.license.strip("\n"))
            lines.append('"""')
        if self.template:
            lines.append("TEMPLATE \"\"\"")
            lines.append(self.template.strip("\n"))
            lines.append('"""')
        if self.system:
            lines.append("SYSTEM \"\"\"")
            lines.append(self.system.strip("\n"))
            lines.append('"""')
        if self.adapter:
            lines.append(f"ADAPTER {self.adapter}")
        for role, content in self.messages:
            lines.append('MESSAGE {role} """'.format(role=role))
            lines.append(content.strip("\n"))
            lines.append('"""')
        for key in sorted(self.parameters):
            val = self.parameters[key]
            if isinstance(val, bool):
                val = "true" if val else "false"
            sval = str(val)
            if "\n" in sval or sval == "" or any(c.isspace() for c in sval):
                lines.append(f'PARAMETER {key} """{sval}"""')
            else:
                lines.append(f"PARAMETER {key} {sval}")
        return "\n".join(lines) + "\n"

    def write(self, path: Path | str) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.render(), encoding="utf-8")
        return p


def manifest_from_gguf(gguf_path: Path | str, *, name_hint: str = "") -> tuple[Manifest, list[str]]:
    """Собирает Manifest из метаданных GGUF. Возвращает (manifest, список подсказок)."""
    gp = Path(gguf_path).expanduser().resolve()
    md = read_gguf_metadata(gp)
    man = Manifest(from_=str(gp))
    hints: list[str] = []

    tmpl = md.chat_template
    if tmpl:
        man.template = tmpl
    else:
        hints.append("⚠ в GGUF нет tokenizer.chat_template — ollama подставит шаблон "
                     "по умолчанию; проверь ответы и при необходимости допиши TEMPLATE вручную.")

    # sampler.* значения лежат в metadata.values через raw kv; берём из полного словаря
    params = _collect_parameters(md)
    if not params:
        hints.append("ℹ в файле нет sampler.-параметров — применятся дефолты Ollama "
                     "(temp 0.8, top_p 0.9, num_ctx 2048). Допиши PARAMETER при желании.")
    man.parameters = params

    arch = md.arch or "модель"
    man.system = ""  # SYSTEM не храним в GGUF — пусть пользователь сам решит через --system
    hints.append(f"✔ архитектура: {arch}, квант: {md.quant or '?'}, параметров: {md.n_params or '?'}")
    if name_hint:
        pass
    return man, hints


# префикс-варианты sampler.-ключей у разных архитекторов
SAMPLER_PREFIXES = ("sampler.", "tokenizer.ggml.")

PARAM_BY_SUFFIX = {
    "top_k": "top_k",
    "top_p": "top_p",
    "min_p": "min_p",
    "temp": "temperature",
    "repeat_penalty": "repeat_penalty",
    "repeat_last_n": "repeat_last_n",
    "presence_penalty": "presence_penalty",
    "frequency_penalty": "frequency_penalty",
    "seed": "seed",
}


def _collect_parameters(md) -> dict:
    """Достаёт PARAMETER'ы из KV-метаданных GGUF (sampler.* / context_length / add_bos)."""
    out: dict = {}
    kv = getattr(md, "kv", {}) or {}
    arch = md.arch
    for key, value in kv.items():
        if not isinstance(value, (int, float, bool)) or isinstance(value, (list, tuple)):
            continue
        base = key.rsplit(".", 1)[-1]
        if key.startswith("sampler.") and base in PARAM_BY_SUFFIX:
            out[PARAM_BY_SUFFIX[base]] = value
        elif key.endswith(".add_bos_token"):
            out["add_bos_token"] = bool(value)
    # num_ctx из контекста обучения (Ollama по умолчанию режет до 2048 — поднимаем)
    if md.ctx_train and "num_ctx" not in out:
        out["num_ctx"] = min(int(md.ctx_train), 32768)
    if arch:
        pass
    return out


def default_manifest_name(gguf_path: Path | str) -> str:
    """qwen2.5-coder-7b-instruct-q4_k_m.gguf -> qwen2.5-coder-7b:q4_k_m"""
    stem = Path(gguf_path).name.lower().removesuffix(".gguf")
    parts = [p for p in stem.replace("_", "-").split("-") if p]
    quant = ""
    for tok in ("q8_0", "q6_k", "q5_k_m", "q5_k_s", "q4_k_m", "q4_k_s", "q4_0", "q3_k_m",
                "iq3_xxs", "iq2_xxs", "f16", "bf16"):
        if tok in parts[-2:] or stem.endswith(tok):
            quant = ":" + tok.replace("_", "_")
            break
    # схлопываем имя: убираем мусор вида 'gguf', 'v2' оставляем
    name = "-".join(p for p in parts if p not in {"gguf"})
    if quant and name.endswith(quant.lstrip(":")):
        name = name[: -len(quant.lstrip(":"))].rstrip("-")
    return (name or "custom-model") + (quant or ":latest")


def ollama_available() -> bool:
    return shutil_which("ollama") is not None


def shutil_which(name: str) -> str | None:
    import shutil
    return shutil.which(name)


def ollama_create(manifest_path: Path | str, name: str, *, timeout: int = 600) -> subprocess.CompletedProcess:
    """Запускает `ollama create <name> -f <Modelfile>`."""
    return subprocess.run(
        ["ollama", "create", name, "-f", str(manifest_path)],
        capture_output=True, text=True, timeout=timeout,
    )
