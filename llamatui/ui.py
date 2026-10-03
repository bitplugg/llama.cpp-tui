"""Красивые rich/tqdm-виджеты: баннер, статус-панели, прогресс генерации."""

from __future__ import annotations

import shutil
import threading
import time
from datetime import datetime

from rich.align import Align
from rich.columns import Columns
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.rule import Rule
from rich.spinner import Spinner
from rich.table import Table
from rich.text import Text
from tqdm import tqdm

from .backend import GenStats

console = Console()

# ---------------------------------------------------------------------------
# Баннер
# ---------------------------------------------------------------------------

_LOGO_SMALL = r"""
  __     __    _                     _  ___   _
  \ \   / /_ _| |___ ___ __ _ _ __(_)/ _ ) /_\
   \ \/ / _` | / -_) -_) _` | '_ \ |  _/  / _ \
    \__/ \__,_|_\___\___|__,_| .__/_| |_| /_/ \_\
                             |_|   v{ver}
"""

_LOGO_BLOCK = """   ██╗     ██╗     █████╗ ███╗   ███╗ █████╗ ████████╗██╗   ██╗
   ██║     ██║    ██╔══██╗████╗ ████║██╔══██╗╚══██╔══╝╚██╗ ██╔╝
   ██║     ██║    ███████║██╔████╔██║███████║   ██║    ╚████╔╝
   ██║     ██║    ██╔══██║██║╚██╔╝██║██╔══██║   ██║     ╚███╔╝
   ███████╗██║    ██║  ██║██║ ╚═╝ ██║██║  ██║   ██║     ██╔╝██╗
   ╚══════╝╚═╝    ╚═╝  ╚═╝╚═╝     ╚═╝╚═╝  ╚═╝   ╚═╝     ╚═╝ ╚═╝"""


def render_banner(version: str) -> Panel:
    cols = console.width
    logo = _LOGO_BLOCK if cols >= 100 else _LOGO_SMALL.format(ver=version)
    art = Text.from_ansi(logo, style="bright_magenta")
    sub = Text.assemble(
        ("красивая оболочка над llama.cpp\n", "italic cyan"),
        (f"rich + tqdm · без textual · {datetime.now():%d.%m.%Y}", "dim"),
    )
    body = Group(Align.center(art), Align.center(sub))
    return Panel(
        body,
        border_style="bright_magenta",
        padding=(1, 2),
        expand=False,
    )


def show_banner(version: str) -> None:
    console.print(Align.center(render_banner(version)))
    console.print(Rule(style="grey35"))


# ---------------------------------------------------------------------------
# Статус-строки
# ---------------------------------------------------------------------------

def status_line(**kv: str) -> str:
    parts = [f"[dim]{k}:[/] [bold cyan]{v}[/]" for k, v in kv.items()]
    return "  ".join(parts)


def model_loaded_badge(model: str, backend_kind: str, n_ctx: int, gpu_layers: str) -> Panel:
    t = Table.grid(padding=(0, 2))
    t.add_row(
        "[green]✔[/]",
        f"[bold bright_white]{model}[/]",
        f"[dim]бэкенд:[/] [yellow]{backend_kind}[/]",
        f"[dim]ctx:[/] [cyan]{n_ctx:,}[/]",
        f"[dim]GPU-слои:[/] [cyan]{gpu_layers}[/]",
    )
    return Panel(t, border_style="green", padding=(0, 2))


# ---------------------------------------------------------------------------
# Прогресс генерации: tqdm + rich Live
# ---------------------------------------------------------------------------

class GenerationProgress:
    """tqdm-бар по символам + живой счётчик токенов/с в postfix.

    После финиша печатает rich-табличку метрик (ток/с, TTFT, время)."""

    def __init__(self, *, char_hint: int | None = None):
        self.bar: tqdm | None = None
        self.char_hint = char_hint
        self._chars = 0
        self._last_render = 0.0

    def __enter__(self) -> "GenerationProgress":
        self.bar = tqdm(
            total=self.char_hint,
            unit="ch",
            unit_scale=True,
            desc="[bold blue]⚡ генерация[/]",
            bar_format="{l_bar}{bar}| {n_fmt}{postfix}",
            dynamic_ncols=True,
            leave=False,
        )
        return self

    def feed(self, piece_len: int, stats: GenStats) -> None:
        """Один кусок стрима длиной piece_len символов."""
        assert self.bar is not None
        self._chars += piece_len
        self.bar.update(piece_len)
        now = time.monotonic()
        if now - self._last_render > 0.2:  # не спамим в терминал
            self.bar.set_postfix_str(f" ~{stats.tokens_out} tok · [cyan]{stats.tok_per_sec:.1f} tok/s[/]",
                                     refresh=False)
            self._last_render = now

    def __exit__(self, *exc) -> bool:
        # выход без метрик — просто закрываем tqdm-бар
        if self.bar is not None:
            self.bar.close()
            self.bar = None
        return False

    def finish(self, stats: GenStats, *, ctx_used: int | None = None) -> None:
        if self.bar is not None:
            self.bar.close()
            self.bar = None
        console.print(render_stats(stats, ctx_used=ctx_used))


def render_stats(stats: GenStats, *, ctx_used: int | None = None) -> Panel:
    ttft = f"{stats.ttft * 1000:.0f} мс" if stats.ttft is not None else "—"
    prompt = stats.prompt_tokens or (ctx_used or 0)
    grid = Table.grid(padding=(0, 3))
    grid.add_row(
        f"[bold green]↯ {stats.tok_per_sec:.1f}[/] [dim]ток/с[/]",
        f"[bold yellow]⏱ {stats.elapsed:.2f}[/] [dim]сек всего[/]",
        f"[bold cyan]🧩 {stats.tokens_out}[/] [dim]токенов[/]",
        f"[dim]первый токен:[/] [magenta]{ttft}[/]",
        f"[dim]контекст:[/] [blue]{prompt + stats.tokens_out:,}[/]",
    )
    stop = "EOS" if stats.stopped_by_eos else "лимит"
    return Panel(
        Group(grid, Text(f"остановка: {stop}", style="dim")),
        border_style="grey35",
        padding=(0, 2),
    )


# ---------------------------------------------------------------------------
# Сплэш «модель грузится» (пока ждём readiness сервера)
# ---------------------------------------------------------------------------

def wait_with_spinner(console_: Console, fn, *, text: str = "Загружаю модель…"):
    """Выполняет блокирующий fn в фоне, показывая rich-спиннер с таймером."""
    start = time.monotonic()
    result: list = []
    error: list = []

    def worker():
        try:
            result.append(fn())
        except Exception as exc:  # noqa: BLE001
            error.append(exc)

    th = threading.Thread(target=worker, daemon=True)
    th.start()
    frame = Text.assemble(("◜ ", "bright_magenta"), (text, "bright_white"))
    with Live(frame, console=console_, transient=True, refresh_per_second=10) as live:
        while th.is_alive():
            frame.plain = ""
            frame.append(f" {text} ", style="bright_white")
            frame.append(f"({time.monotonic() - start:.1f} c)", style="dim")
            time.sleep(0.1)
    th.join()
    if error:
        raise error[0]
    return result[0] if result else None


# ---------------------------------------------------------------------------
# Разговорные рамки
# ---------------------------------------------------------------------------

def print_user_bubble(text: str) -> None:
    console.print(Panel(
        Text(text, style="white"),
        title="[bold blue]🧑 ты[/]", title_align="left",
        border_style="blue", padding=(0, 1),
    ))


def assistant_prefix() -> Text:
    return Text.assemble(("🤖 ", ""), ("ассистент › ", "bold green"))
