"""llamatui — красивая консольная оболочка над llama.cpp / Ollama.

Подкоманды:
  chat   — интерактивный чат (по умолчанию)
  pull   — поиск и скачивание GGUF-моделей с Hugging Face
  setup  — скачать/найти бинарники llama.cpp
  engines— какие движки доступны прямо сейчас (llama.cpp, ollama, openai-эндпоинт)
  info   — метаданные GGUF-файла
  run    — разовый запрос к модели
"""

from __future__ import annotations

import argparse
import atexit
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text

from . import __version__
from .backend import (ENGINES, Backend, ChatMessage, LlamaServerError, OllamaBackend,
                      OpenAICompatBackend, create_backend)
from .downloader import DownloadError, auto_install, find_all, llama_dir, resolve_cli_binary
from .gguf import GGUFError, read_gguf_metadata, render_model_card
from .manifest import (Manifest, default_manifest_name, manifest_from_gguf,
                       ollama_available, ollama_create)
from .models import (
    MODELS_DIR, ModelError, ModelFile, download_model, human_size,
    model_info, render_model_files, render_search_results, search_models,
)
from .ui import (
    GenerationProgress, assistant_prefix, model_loaded_badge,
    print_user_bubble, show_banner, wait_with_spinner,
)

console = Console()

CONFIG_PATH = Path(llama_dir()) / "config.json"


# ---------------------------------------------------------------------------
# Конфиг
# ---------------------------------------------------------------------------

DEFAULTS = {
    "model": None,
    "engine": "auto",            # auto | llama | llama-cli | llama-server | ollama | openai
    "base_url": "",              # для engine=openai / ollama (пусто = дефолт)
    "api_key": None,             # для engine=openai (можно через env OPENAI_API_KEY)
    "n_ctx": 8192,
    "n_gpu_layers": -1,          # -1 = сколько влезет (сервер сам решит по default)
    "temp": 0.7,
    "top_p": 0.95,
    "max_tokens": 1024,
    "system": "Ты — полезный ассистент. Отвечай кратко и по делу.",
    "extra_args": [],
    "render_markdown": True,
}


def load_config() -> dict:
    cfg = dict(DEFAULTS)
    if CONFIG_PATH.exists():
        try:
            cfg.update(json.loads(CONFIG_PATH.read_text()))
        except json.JSONDecodeError:
            console.print("[yellow]config.json битый — беру дефолты[/]")
    return cfg


def save_config(cfg: dict) -> None:
    CONFIG_PATH.write_text(json.dumps(cfg, indent=2, ensure_ascii=False))


# ---------------------------------------------------------------------------
# Поиск локальных моделей
# ---------------------------------------------------------------------------

def local_models() -> list[Path]:
    dirs = [MODELS_DIR, Path.cwd()]
    found: list[Path] = []
    seen: set[str] = set()
    for d in dirs:
        if not d.is_dir():
            continue
        for p in sorted(d.glob("*.gguf")):
            if p.name not in seen:
                seen.add(p.name)
                found.append(p)
    return found


def ollama_models() -> list[str]:
    """Список моделей, подтянутых в локальном Ollama-демоне (пусто, если демона нет)."""
    try:
        ob = OpenAICompatBackend("", "")  # без ping
        from .backend import OllamaBackend as _OB  # переиспользуем list_models без ensure_up
        holder = _OB.__new__(_OB)
        OpenAICompatBackend.__init__(holder, "http://127.0.0.1:11434", "", temp=0.7)
        return [m for m in holder.list_models() if m]
    except Exception:  # noqa: BLE001
        return []


def pick_model_target(args_model: str | None, cfg: dict) -> str:
    """Разрешает цель чата в строку.

    --engine llama* → обязателен GGUF (путь / имя из ~/.llamatui/models / единственный локальный);
    --engine ollama/openai → цель это имя модели; если не указана — список из демона/подсказка;
    --engine auto → сначалаtrying ollama-имя, иначе GGUF-цепочка.
    """
    engine = cfg.get("engine", "auto")
    if args_model:
        p = Path(args_model).expanduser()
        if p.is_file():
            return str(p)
        cand = MODELS_DIR / args_model
        if cand.is_file():
            return str(cand)
        matches = list(MODELS_DIR.glob(f"*{args_model}*.gguf"))
        if len(matches) == 1:
            return str(matches[0])
        if matches:
            console.print(Panel(
                "\n".join(f"[cyan]{m.name}[/]  [dim]{human_size(m.stat().st_size)}[/]" for m in matches),
                title=f"Несколько файлов подходят под «{args_model}» — уточни", border_style="yellow"))
            raise SystemExit(1)
        # не файл → возможно имя модели для ollama/openai
        if engine in ("ollama", "openai"):
            return args_model
        if engine == "auto":
            om = ollama_models()
            if any(m == args_model or m.startswith(args_model) for m in om):
                return args_model
            console.print(f"[yellow]нет такого GGUF и такой модели в ollama:[/] {args_model}\n"
                          f"[dim]GGUF: `llamatui pull` · ollama: `ollama pull {args_model}`[/]")
            raise SystemExit(1)
        console.print(f"[red]не нашёл модель:[/] {args_model}\n"
                      f"[dim]проверь путь или скачай через: llamatui pull[/]")
        raise SystemExit(1)

    if cfg.get("model"):
        p = Path(cfg["model"]).expanduser()
        if p.is_file():
            return str(p)
        if engine in ("ollama", "openai") or ":" in str(cfg["model"]):
            return str(cfg["model"])

    models = local_models()
    if engine in ("ollama", "openai"):
        console.print("[red]для --engine ollama/openai укажи имя модели:[/] "
                      "[bold]llamatui --model qwen2.5:7b[/] "
                      "[dim](ollama models покажет доступные)[/]")
        raise SystemExit(1)
    if engine == "auto":
        om = ollama_models()
        if not models and om:
            console.print("[dim]GGUF не найдено, но живёт Ollama — беру первую его модель[/]")
            return om[0]
    if len(models) == 1:
        return str(models[0])
    if not models:
        om = ollama_models()
        hint = ("[dim]Ollama: `llamatui --engine ollama --model " + om[0] + "`[/]\n"
                if om else "")
        console.print("[red]Вроде нет ни одной GGUF-модели.[/] "
                      "[dim]скачай: `llamatui pull` или укажи путь: `--model путь.gguf`[/]\n" + hint)
        raise SystemExit(1)
    body = "\n".join(
        f"[bold cyan]{i}[/] · {m.name}  [dim]{human_size(m.stat().st_size)}[/]"
        for i, m in enumerate(models, 1))
    console.print(Panel(body, title="Локальные модели — выбери через --model <имя|путь>",
                        border_style="cyan"))
    raise SystemExit(1)


# ---------------------------------------------------------------------------
# Команда: engines
# ---------------------------------------------------------------------------

def cmd_engines(args: argparse.Namespace) -> int:
    from rich.table import Table
    show_banner(__version__)
    t = Table(title="Движки, доступные прямо сейчас", header_style="bold white on grey23")
    t.add_column("движок"); t.add_column("статус"); t.add_column("детали", style="dim")

    bins = find_all()
    llama_ok = bins.get("llama-server") or bins.get("llama-cli")
    t.add_row("llama.cpp (server+cli)",
              "[green]✔ готов[/]" if llama_ok else "[red]✖ нет бинарников[/]",
              str(llama_dir()) if llama_ok else "llamatui setup")
    for name, p in bins.items():
        t.add_row(f"  · {name}", "[green]✔[/]" if p else "[dim]—[/]", str(p or ""))

    try:
        ob = OllamaBackend.__new__(OllamaBackend)
        OpenAICompatBackend.__init__(ob, os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434"), "", temp=0.7)
        up = ob.ping(timeout=1.5)
        names = ob.list_models() if up else []
        which = __import__("shutil").which("ollama")
        status = "[green]✔ демон жив[/]" if up else ("[yellow]○ установлен, не запущен[/]"
                 if which else "[red]✖ не найден[/]")
        t.add_row("Ollama", status,
                  (f"{len(names)} моделей: " + ", ".join(names[:6])) if names else (which or "https://ollama.com/download"))
    except Exception as exc:  # noqa: BLE001
        t.add_row("Ollama", "[red]✖ " + str(exc)[:40] + "[/]", "")

    base = args.base_url or load_config().get("base_url") or ""
    if base:
        ok = OpenAICompatBackend(base, "").ping()
        t.add_row("OpenAI-эндпоинт", "[green]✔ отвечает[/]" if ok else "[red]✖ молчит[/]", base)
    else:
        t.add_row("OpenAI-эндпоинт", "[dim]не настроен[/]",
                  "llamatui --engine openai --base-url http://localhost:1234/v1 …")
    console.print(t)
    return 0


# ---------------------------------------------------------------------------
# Команда: setup
# ---------------------------------------------------------------------------

def cmd_setup(args: argparse.Namespace) -> int:
    show_banner(__version__)
    bins = find_all()
    ok = all(bins.values())
    if ok and not args.force:
        t = _bins_table(bins)
        console.print(t)
        console.print("[green]✔ всё на месте[/]  "
                      f"[dim]перекачать: llamatui setup --force[/]")
        return 0
    if args.binaries_only or args.no_download:
        console.print(_bins_table(bins))
        return 0 if any(bins.values()) else 1
    console.print("[cyan]llama.cpp не найден — качаю свежие бинарники…[/]")
    try:
        res = auto_install(force_cpu=args.cpu)
    except (DownloadError, KeyboardInterrupt) as exc:
        console.print(f"[red]✖ {exc}[/]")
        return 1
    console.print(Panel(
        f"[green]✔ установлено:[/] llama.cpp [bold]{res.tag}[/] · {res.target_label}\n"
        + ", ".join(f"[bold]{n}[/]" for n in res.installed)
        + f"\n[dim]{llama_dir()}[/]",
        border_style="green"))
    return 0


def _bins_table(bins: dict) -> "Table":
    from rich.table import Table
    t = Table(title="Бинарники llama.cpp", header_style="bold white on grey23")
    t.add_column("бинарник"); t.add_column("статус"); t.add_column("путь", style="dim")
    for name, p in bins.items():
        t.add_row(name, "[green]✔ найден[/]" if p else "[red]✖ нет[/]", str(p or "—"))
    return t


# ---------------------------------------------------------------------------
# Команда: pull
# ---------------------------------------------------------------------------

def cmd_pull(args: argparse.Namespace) -> int:
    show_banner(__version__)
    query = args.query or console.input("[bold cyan]🔎 ищем модель (название/автор): [/]").strip()
    if not query:
        return 1
    with console.status("[magenta]ищу на Hugging Face…[/]"):
        try:
            results = search_models(query, limit=args.limit)
        except ModelError as exc:
            console.print(f"[red]✖ {exc}[/]")
            return 1
    if not results:
        console.print("[yellow]ничего не нашлось 🤷[/]")
        return 1
    console.print(render_search_results(results))

    choice = console.input("[bold]номер модели[/] [dim](или repo id)[/][cyan] › [/]").strip()
    if choice.isdigit() and 1 <= int(choice) <= len(results):
        repo_id = results[int(choice) - 1][0]
    elif choice:
        repo_id = choice
    else:
        return 1

    with console.status(f"[magenta]читаю список файлов {repo_id}…[/]"):
        try:
            info = model_info(repo_id)
        except ModelError as exc:
            console.print(f"[red]✖ {exc}[/]")
            return 1
    files = info.gguf_files
    if not files:
        console.print("[red]в репозитории нет .gguf файлов[/]")
        return 1
    console.print(render_model_files(info))

    sel = console.input("[bold]номер файла для скачивания[/][cyan] › [/]").strip()
    idx = int(sel) - 1 if sel.isdigit() else -1
    if not (0 <= idx < len(files)):
        console.print("[red]нет такого номера[/]")
        return 1
    f = files[idx]
    dest = Path(args.dest or MODELS_DIR)
    console.print(f"\n[dim]получаем {f.quant} → {dest / f.name.split('/')[-1]}[/]")
    try:
        path = download_model(f, dest)
    except (ModelError, KeyboardInterrupt) as exc:
        console.print(f"[red]✖ {exc}[/]")
        return 1
    console.print(Panel(
        f"[green]✔ готово![/]\n[bold bright_white]{path}[/]\n"
        f"[dim]запуск: llamatui chat --model {path.name}[/]",
        border_style="green"))

    # бонус: не теряем скачанный GGUF даром — предлагаем сразу собрать манифест Ollama
    try:
        ans = console.input("\n[bold]🦙 Собрать Modelfile (манифест) для Ollama из этой модели?[/] "
                            "[dim][y/N][/][cyan] › [/]").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return 0
    if ans in ("y", "yes", "д", "да"):
        cmd_ollama(argparse.Namespace(
            ollama_action="create", model=str(path), name=None, out=None,
            system=None, set=None, run=False, print=True, action="ollama"))
    return 0


# ---------------------------------------------------------------------------
# Команда: info
# ---------------------------------------------------------------------------

def cmd_info(args: argparse.Namespace) -> int:
    path = Path(args.path).expanduser()
    if not path.is_file():
        # попробуем из папки моделей
        alt = MODELS_DIR / args.path
        if alt.is_file():
            path = alt
        else:
            console.print(f"[red]нет файла:[/] {args.path}")
            return 1
    try:
        md = read_gguf_metadata(path)
    except GGUFError as exc:
        console.print(f"[red]✖ {exc}[/]")
        return 1
    console.print(render_model_card(md, path, console))
    return 0


# ---------------------------------------------------------------------------
# Команда: run (разовый запрос)
# ---------------------------------------------------------------------------

def cmd_run(args: argparse.Namespace) -> int:
    cfg = load_config()
    if getattr(args, "engine", None): cfg["engine"] = args.engine
    if getattr(args, "base_url", None): cfg["base_url"] = args.base_url
    model = pick_model_target(args.model, cfg)
    backend = _boot_backend(model, cfg, args)
    prompt = " ".join(args.prompt) if args.prompt else sys.stdin.read().strip()
    if not prompt:
        console.print("[red]пустой промпт[/]")
        return 1
    messages = []
    if cfg.get("system"):
        messages.append(ChatMessage("system", cfg["system"]))
    messages.append(ChatMessage("user", prompt))
    try:
        text = _generate(backend, messages, cfg, stream=not args.quiet)
    finally:
        backend.shutdown()
    if args.quiet:
        print(text)
    return 0


# ---------------------------------------------------------------------------
# Команда: chat
# ---------------------------------------------------------------------------

HELP_TEXT = """[bold]команды чата[/]
/gs [py]текст[/]              задать системный промпт
/temp [py]0.7[/] · /topp [py]0.9[/] · /tokens [py]1024[/]   параметры генерации
/ctx [py]8192[/]              размер контекста (перезапустит сервер)
/md on|off                    рендер markdown в ответах
/save                         сохранить историю в JSON · /load [py]файл[/]
/clear                        очистить историю
/model [py]имя.gguf | qwen2.5:7b[/]   сменить модель на лету
/engine [py]auto|llama|llama-cli|ollama|openai[/]   сменить движок
/ollama                       список моделей в локальном Ollama
/modelfile                    собрать Modelfile (манифест) из текущей GGUF-модели
/setup · /pull                доустановить бинарники / скачать модель
/info                         инфо о текущей GGUF-модели
/help                         эта справка · /exit (Ctrl-D) выход
[dim]Esc/Esc — стереть строку · ↑↓ — история ввода[/]"""


class ChatApp:
    def __init__(self, cfg: dict, target: str):
        self.cfg = cfg
        self.target = target          # путь к .gguf ИЛИ имя модели для ollama/openai
        self.model = Path(target)     # для gguf-специфичных фич (info и т.п.)
        self.backend: Backend | None = None
        self.history: list[ChatMessage] = []
        self._setup_prompt_toolkit()

    @property
    def is_gguf(self) -> bool:
        return self.target.endswith(".gguf") and self.model.is_file()

    # -- ввод -----------------------------------------------------------------

    def _setup_prompt_toolkit(self) -> None:
        try:
            from prompt_toolkit import PromptSession
            from prompt_toolkit.completion import WordCompleter
            from prompt_toolkit.history import FileHistory

            completer = WordCompleter(
                ["/help", "/gs", "/system", "/temp", "/topp", "/tokens", "/ctx",
                 "/md", "/save", "/load", "/clear", "/model", "/engine", "/ollama",
                 "/modelfile", "/setup", "/pull", "/info", "/exit", "/quit"],
                sentence=True,
            )
            histfile = Path(llama_dir()) / "chat_history.txt"
            self.session: object | None = PromptSession(
                history=FileHistory(histfile), completer=completer, multiline=False,
            )
        except ImportError:
            self.session = None

    def input_line(self, prompt_str: str) -> str | None:
        if self.session is not None:
            from prompt_toolkit import PromptSession
            from prompt_toolkit.shortcuts import CompleteStyle
            assert isinstance(self.session, PromptSession)
            # prompt_toolkit не понимает rich-разметку — подаём plain
            return self.session.prompt("ты › ", color=True)
        try:
            return console.input(prompt_str)
        except EOFError:
            return None

    # -- жизненный цикл -------------------------------------------------------

    def boot(self) -> bool:
        cfg = self.cfg
        engine = cfg.get("engine", "auto")
        try:
            self.backend = wait_with_spinner(
                console,
                lambda: create_backend(
                    self.target,
                    engine=engine,
                    n_ctx=int(cfg["n_ctx"]),
                    n_gpu_layers=None if int(cfg.get("n_gpu_layers", -1)) < 0 else int(cfg["n_gpu_layers"]),
                    extra_args=cfg.get("extra_args"),
                    temp=float(cfg["temp"]),
                    base_url=cfg.get("base_url", ""),
                    api_key=cfg.get("api_key"),
                ),
                text=f"Загружаю {Path(self.target).name if self.is_gguf else self.target} (это может занять минуту)…",
            )
        except LlamaServerError as exc:
            console.print(Panel(str(exc), title="✖ запуск не удался", border_style="red"))
            return False
        gpu = "—" if self.backend.kind in ("ollama", "openai") else (
            "авто/все" if int(cfg.get("n_gpu_layers", -1)) < 0 else cfg["n_gpu_layers"])
        console.print(model_loaded_badge(
            Path(self.target).name if self.is_gguf else self.target,
            self.backend.kind, int(cfg["n_ctx"]), str(gpu)))
        atexit.register(self.shutdown)
        return True

    def shutdown(self) -> None:
        if self.backend:
            self.backend.shutdown()

    # -- генерация ------------------------------------------------------------

    def generate(self, user_text: str) -> None:
        assert self.backend is not None
        cfg = self.cfg
        messages = ([ChatMessage("system", cfg["system"])] if cfg.get("system") else []) + self.history
        gen_iter, stats = self.backend.chat_stream(
            messages,
            temp=float(cfg["temp"]),
            top_p=float(cfg["top_p"]),
            max_tokens=int(cfg["max_tokens"]),
        )

        console.print(assistant_prefix(), end="")
        collected: list[str] = []
        md_mode = bool(cfg.get("render_markdown")) and self.backend.kind == "server"
        # markdown рендерим постфактум; во время стрима пишем plain-текстом
        with GenerationProgress(char_hint=int(cfg["max_tokens"]) * 4 if md_mode else None) as prog:
            try:
                for piece in gen_iter:
                    collected.append(piece)
                    console.print(Text(piece, style="bright_white"), end="", highlight=False)
                    prog.feed(len(piece), stats)
            except KeyboardInterrupt:
                console.print("\n[yellow]⚠ остановлено пользователем[/]")
            finally:
                answer = "".join(collected)
                console.print()
                prog.finish(stats, ctx_used=self.backend.count_tokens(user_text))
        if md_mode and answer.strip():
            console.print(Panel(Markdown(answer), border_style="green", padding=(0, 1),
                                title="[bold green]🤖 markdown[/]", title_align="left"))
        self.history.append(ChatMessage("user", user_text))
        self.history.append(ChatMessage("assistant", answer))

    # -- команды ---------------------------------------------------------------

    def handle_command(self, line: str) -> bool:
        """True — команда обработана, False — это обычный текст."""
        if not line.startswith("/"):
            return False
        parts = line.split(maxsplit=1)
        cmd = parts[0].lower()
        arg = parts[1].strip() if len(parts) > 1 else ""
        cfg = self.cfg

        if cmd in ("/exit", "/quit"):
            raise KeyboardInterrupt
        if cmd == "/help":
            console.print(Panel(HELP_TEXT, border_style="cyan", title="справка"))
        elif cmd in ("/gs", "/system"):
            cfg["system"] = arg
            console.print(f"[green]системный промпт обновлён:[/] [dim]{arg[:120]}[/]")
        elif cmd == "/temp":
            cfg["temp"] = float(arg); console.print(f"temp = [cyan]{cfg['temp']}[/]")
        elif cmd == "/topp":
            cfg["top_p"] = float(arg); console.print(f"top_p = [cyan]{cfg['top_p']}[/]")
        elif cmd == "/tokens":
            cfg["max_tokens"] = int(arg); console.print(f"max_tokens = [cyan]{cfg['max_tokens']}[/]")
        elif cmd == "/ctx":
            cfg["n_ctx"] = int(arg)
            console.print(f"ctx = [cyan]{cfg['n_ctx']}[/] — перезапускаю сервер…")
            self.shutdown(); 
            if not self.boot():
                raise SystemExit(1)
        elif cmd == "/md":
            cfg["render_markdown"] = arg.lower() in ("on", "1", "yes", "да")
            console.print(f"markdown = [cyan]{'вкл' if cfg['render_markdown'] else 'выкл'}[/]")
        elif cmd == "/clear":
            self.history.clear(); console.print("[dim]история очищена[/]")
        elif cmd == "/save":
            fname = arg or f"chat_{time.strftime('%Y%m%d_%H%M%S')}.json"
            Path(fname).write_text(json.dumps(
                [asdict(m) for m in self.history], indent=2, ensure_ascii=False))
            console.print(f"[green]сохранено:[/] {fname}")
        elif cmd == "/load":
            data = json.loads(Path(arg).read_text())
            self.history = [ChatMessage(d["role"], d["content"]) for d in data]
            console.print(f"[green]загружено {len(self.history)} сообщений[/]")
        elif cmd == "/model":
            try:
                new = pick_model_target(arg or None, cfg)
            except SystemExit:
                return True
            console.print(f"меняю модель на [bold]{Path(new).name if new.endswith('.gguf') else new}[/]…")
            self.target = new; self.model = Path(new)
            cfg["model"] = new; save_config(cfg)
            self.history.clear()
            self.shutdown()
            if not self.boot():
                raise SystemExit(1)
        elif cmd == "/engine":
            from .backend import ENGINES
            if arg not in ENGINES:
                console.print(f"[yellow]движки:[/] [cyan]{', '.join(ENGINES)}[/]")
                return True
            cfg["engine"] = arg
            console.print(f"engine = [cyan]{arg}[/] — перезапускаю…")
            self.shutdown()
            if not self.boot():
                raise SystemExit(1)
        elif cmd == "/ollama":
            names = ollama_models()
            if names:
                body = "\n".join(f"[bold cyan]{i}[/] · {n}" for i, n in enumerate(names, 1))
                console.print(Panel(body, title="Модели в Ollama", border_style="magenta"))
            else:
                console.print("[dim]Ollama-демон не отвечает (или там пусто)[/]")
        elif cmd == "/modelfile":
            if not self.is_gguf:
                console.print("[dim]/modelfile работает, когда цель — локальный .gguf[/]")
            else:
                cmd_ollama(argparse.Namespace(
                    ollama_action="create", model=str(self.model), name=arg or None,
                    out=None, system=self.cfg.get("system") or None, set=None,
                    run=False, print=True, action="ollama"))
        elif cmd == "/info":
            if not self.is_gguf:
                console.print("[dim]/info показывает метаданные GGUF-файла — "
                              "сейчас цель не локальный .gguf[/]")
            else:
                try:
                    md = read_gguf_metadata(self.model)
                    console.print(render_model_card(md, self.model, console))
                except GGUFError as exc:
                    console.print(f"[red]{exc}[/]")
        elif cmd == "/setup":
            cmd_setup(argparse.Namespace(force=False, cpu=False, no_download=False,
                                         binaries_only=False, action="setup"))
        elif cmd == "/pull":
            cmd_pull(argparse.Namespace(query=arg, limit=15, dest=None, action="pull"))
        else:
            console.print(f"[yellow]неизвестная команда {cmd} — /help покажет список[/]")
        return True

    # -- главный цикл -----------------------------------------------------------

    def run(self) -> int:
        show_banner(__version__)
        if not self.boot():
            return 1
        console.print(f"[dim]режим:[/] [yellow]{self.backend.kind}[/] · "
                      f"[dim]/help — команды, Ctrl-D — выход[/]\n")
        while True:
            try:
                line = self.input_line("[bold blue]ты › [/]")
            except KeyboardInterrupt:
                console.print("\n[dim]пока! 👋[/]")
                break
            except EOFError:
                console.print("\n[dim]пока! 👋[/]")
                break
            if line is None:
                break
            line = line.strip()
            if not line:
                continue
            try:
                if self.handle_command(line):
                    continue
                print_user_bubble(line)
                self.generate(line)
            except KeyboardInterrupt:
                console.print("\n[dim]прервано, продолжаем[/]")
            except ValueError as exc:
                console.print(f"[red]некорректное значение:[/] {exc}")
            except Exception as exc:  # noqa: BLE001 — чат не должен умирать от сбоя генерации
                console.print(Panel(str(exc), title="✖ ошибка", border_style="red"))
        self.shutdown()
        save_config(self.cfg)
        return 0


def cmd_chat(args: argparse.Namespace) -> int:
    cfg = load_config()
    if getattr(args, "engine", None): cfg["engine"] = args.engine
    if getattr(args, "base_url", None): cfg["base_url"] = args.base_url
    if getattr(args, "api_key", None): cfg["api_key"] = args.api_key
    if args.temp is not None: cfg["temp"] = args.temp
    if args.tokens is not None: cfg["max_tokens"] = args.tokens
    if args.ctx is not None: cfg["n_ctx"] = args.ctx
    if args.ngl is not None: cfg["n_gpu_layers"] = args.ngl
    if args.system is not None: cfg["system"] = args.system
    if args.no_md: cfg["render_markdown"] = False
    model = pick_model_target(args.model, cfg)
    app = ChatApp(cfg, model)
    return app.run()


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# llamatui ollama — генерация Modelfile-манифестов для Ollama
# ---------------------------------------------------------------------------

def cmd_ollama(args: argparse.Namespace) -> int:
    from rich.table import Table
    action = args.ollama_action or "create"
    if action == "list":
        models = ollama_models()
        if not models:
            console.print("[yellow]Ollama-демон не отвечает или моделей нет.[/] "
                          "Попробуй: [bold]ollama serve[/]")
            return 1
        t = Table(title="🦙 Модели в Ollama", header_style="bold magenta")
        t.add_column("модель", style="cyan")
        for m in models:
            t.add_row(m)
        console.print(t)
        return 0

    # action == "create": собираем манифест из GGUF
    try:
        target = pick_model_target(args.model, load_config())
    except SystemExit:
        raise
    gguf_path = Path(target)
    if not gguf_path.is_file():
        console.print(Panel(f"[bold]{target}[/] — это не путь к GGUF-файлу.\n"
                            "Для `ollama create` нужен локальный .gguf "
                            "(скачай через [bold]llamatui pull[/]).",
                            title="✖ манифест", border_style="red"))
        return 1

    name = args.name or default_manifest_name(gguf_path)
    try:
        man, hints = manifest_from_gguf(gguf_path)
    except GGUFError as exc:
        console.print(Panel(str(exc), title="✖ GGUF", border_style="red"))
        return 1
    if args.system:
        man.system = args.system
    for kv_pair in (args.set or []):
        if "=" not in kv_pair:
            console.print(f"[red]--set ждёт key=value, получил {kv_pair!r}[/]")
            return 2
        k, v = kv_pair.split("=", 1)
        man.parameters[k.strip()] = v.strip()

    out = Path(args.out).expanduser() if args.out else Path.cwd() / "Modelfile"
    if out.is_dir():
        out = out / "Modelfile"
    man.write(out)

    body = Text()
    body.append(f"FROM {man.from_}\n", style="cyan")
    if man.template:
        first = man.template.strip().splitlines()[0] if man.template.strip() else ""
        body.append(f'TEMPLATE """{first[:60]}…"""\n', style="green")
    if man.system:
        body.append(f'SYSTEM """{man.system[:60]}…"""\n', style="yellow")
    for k, v in sorted(man.parameters.items()):
        body.append(f"PARAMETER {k} {v}\n", style="magenta")
    console.print(Panel(body, title=f"📝 {out}", border_style="bright_blue", expand=False))
    for h in hints:
        console.print(f"  [dim]{h}[/]")

    if args.print:
        console.print(Markdown("```dockerfile\n" + man.render() + "\n```"))

    if args.run:
        if not ollama_available():
            console.print("[red]Бинарник `ollama` не найден в PATH — нечем выполнять create.[/] "
                          "Манифест сохранён, запусти вручную: "
                          f"[bold]ollama create {name} -f {out}[/]")
            return 1
        console.print(f"[cyan]▶ ollama create {name} -f {out} …[/]")
        try:
            res = ollama_create(out, name)
        except Exception as exc:  # noqa: BLE001
            console.print(Panel(str(exc), title="✖ ollama create", border_style="red"))
            return 1
        if res.returncode == 0:
            console.print(f"[bold green]✔ Модель «{name}» создана.[/] "
                          f"Чат: [bold]llamatui chat --engine ollama --model {name}[/]")
        else:
            console.print(Panel((res.stderr or res.stdout or "").strip(),
                                title="✖ ollama create", border_style="red"))
            return res.returncode
    else:
        console.print(f"[dim]Готово. Запуск: [bold]ollama create {name} -f {out}[/] "
                      f"или повтори с [bold]--run[/].[/]")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="llamatui", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--version", action="version", version=f"llamatui {__version__}")
    sub = p.add_subparsers(dest="action")

    c = sub.add_parser("chat", help="интерактивный чат (по умолчанию)")
    c.add_argument("--model", help="GGUF-файл, имя из ~/.llamatui/models или имя модели ollama (qwen2.5:7b)")
    c.add_argument("--engine", choices=list(ENGINES),
                   help="движок: llama.cpp server/cli, ollama, openai-эндпоинт (default: из конфига/auto)")
    c.add_argument("--base-url", dest="base_url", help="для --engine openai/ollama: http://host:port")
    c.add_argument("--api-key", dest="api_key", help="ключ для --engine openai (или env OPENAI_API_KEY)")
    c.add_argument("--temp", type=float)
    c.add_argument("--topp", type=float, dest="top_p")
    c.add_argument("--tokens", type=int, help="max_tokens")
    c.add_argument("--ctx", type=int, help="размер контекста")
    c.add_argument("--ngl", type=int, help="слоёв на GPU (-1 = все)")
    c.add_argument("--system", help="системный промпт")
    c.add_argument("--no-md", action="store_true", help="не рендерить markdown")
    c.set_defaults(func=cmd_chat)

    r = sub.add_parser("run", help="разовый запрос без UI-чата")
    r.add_argument("prompt", nargs="*")
    r.add_argument("--model")
    r.add_argument("--engine", choices=list(ENGINES))
    r.add_argument("--base-url", dest="base_url")
    r.add_argument("--quiet", "-q", action="store_true", help="только текст ответа в stdout")
    r.set_defaults(func=cmd_run)

    pl = sub.add_parser("pull", help="найти и скачать GGUF-модель с HF")
    pl.add_argument("query", nargs="?", help="что искать")
    pl.add_argument("--limit", type=int, default=15)
    pl.add_argument("--dest", help="куда класть (по умолчанию ~/.llamatui/models)")
    pl.set_defaults(func=cmd_pull)

    st = sub.add_parser("setup", help="проверить/скачать бинарники llama.cpp")
    st.add_argument("--force", action="store_true", help="перекачать даже если найдены")
    st.add_argument("--cpu", action="store_true", help="принудительно CPU-сборка")
    st.add_argument("--binaries-only", action="store_true", help="только показать статус")
    st.add_argument("--no-download", action="store_true", help="только показать статус")
    st.set_defaults(func=cmd_setup)

    eng = sub.add_parser("engines", help="какие движки доступны (llama.cpp / ollama / openai)")
    eng.add_argument("--base-url", dest="base_url", default="", help="проверить этот OpenAI-эндпоинт")
    eng.set_defaults(func=cmd_engines)

    inf = sub.add_parser("info", help="метаданные GGUF-файла")
    inf.add_argument("path", help="путь или имя в ~/.llamatui/models")
    inf.set_defaults(func=cmd_info)

    ol = sub.add_parser("ollama", help="манифесты Ollama: создать Modelfile из GGUF / список моделей")
    ol_sub = ol.add_subparsers(dest="ollama_action")
    oc = ol_sub.add_parser("create", help="сгенерировать Modelfile из локального GGUF")
    oc.add_argument("--model", "-m", help="GGUF-файл (путь или имя в ~/.llamatui/models)")
    oc.add_argument("--name", "-n", help="имя модели для ollama create (по умолчанию — из имени файла)")
    oc.add_argument("--out", "-o", help="куда записать Modelfile (по умолчанию ./Modelfile)")
    oc.add_argument("--system", help="SYSTEM-промпт для манифеста")
    oc.add_argument("--set", action="append", metavar="KEY=VAL",
                    help="доп./переопределяемый PARAMETER, можно несколько раз (--set temperature=0.5)")
    oc.add_argument("--run", action="store_true", help="сразу выполнить `ollama create`")
    oc.add_argument("--print", action="store_true", help="распечатать готовый манифест")
    oc.set_defaults(ollama_action="create")
    ol_list = ol_sub.add_parser("list", help="модели, установленные в локальном Ollama")
    ol_list.set_defaults(ollama_action="list")
    ol.set_defaults(func=cmd_ollama, ollama_action="create")
    return p


def _boot_backend(target: str, cfg: dict, args: argparse.Namespace) -> Backend:
    try:
        return wait_with_spinner(
            console,
            lambda: create_backend(
                target,
                engine=cfg.get("engine", "auto"),
                n_ctx=int(cfg["n_ctx"]),
                n_gpu_layers=None if int(cfg.get("n_gpu_layers", -1)) < 0 else int(cfg["n_gpu_layers"]),
                extra_args=cfg.get("extra_args"),
                temp=float(cfg["temp"]),
                base_url=cfg.get("base_url", ""),
                api_key=cfg.get("api_key"),
            ),
            text=f"Загружаю {Path(target).name if target.endswith('.gguf') else target}…",
        )
    except LlamaServerError as exc:
        console.print(Panel(str(exc), title="✖ запуск не удался", border_style="red"))
        raise SystemExit(1)


def _generate(backend: Backend, messages: list[ChatMessage], cfg: dict, *, stream: bool) -> str:
    gen_iter, stats = backend.chat_stream(
        messages, temp=float(cfg["temp"]), top_p=float(cfg["top_p"]),
        max_tokens=int(cfg["max_tokens"]))
    collected = []
    if stream:
        console.print(assistant_prefix(), end="")
        with GenerationProgress() as prog:
            try:
                for piece in gen_iter:
                    collected.append(piece)
                    console.print(Text(piece, style="bright_white"), end="", highlight=False)
                    prog.feed(len(piece), stats)
            except KeyboardInterrupt:
                pass
            finally:
                console.print()
                prog.finish(stats)
    else:
        for piece in gen_iter:
            collected.append(piece)
    return "".join(collected)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    # без подкоманды => сразу chat
    known = {"chat", "run", "pull", "setup", "engines", "info", "ollama", "--version", "--help", "-h"}
    if not argv or argv[0] not in known:
        argv = ["chat", *argv]
    args = parser.parse_args(argv)
    if not getattr(args, "func", None):
        parser.print_help()
        return 0
    try:
        return args.func(args)
    except KeyboardInterrupt:
        console.print("\n[dim]выход.[/]")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
