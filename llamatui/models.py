"""Скачивание GGUF-моделей с Hugging Face / ModelScope / прямого URL (через tqdm)."""

from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

from rich.console import Console
from rich.table import Table
from tqdm import tqdm

console = Console()

HF_API = "https://huggingface.co/api"
USER_AGENT = "llamatui/0.1 (+https://github.com)"


class ModelError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# Поиск
# ---------------------------------------------------------------------------

@dataclass
class ModelFile:
    name: str
    size: int            # байты
    url: str             # для скачивания
    quant: str = ""      # Q4_K_M и т.п.


@dataclass
class ModelInfo:
    repo_id: str
    display_name: str
    downloads: int = 0
    likes: int = 0
    files: list[ModelFile] = field(default_factory=list)

    @property
    def gguf_files(self) -> list[ModelFile]:
        return [f for f in self.files if f.name.lower().endswith(".gguf")]


QUANT_RE = re.compile(
    r"(IQ\d(?:_[A-Z0-9]+)+|Q\d_\d+(?:_[A-Z0-9]+)*|Q\d_[KMLXS](?:_[KMLXS])?|F16|BF16|F32)",
    re.I,
)


def _parse_quant(name: str) -> str:
    m = QUANT_RE.search(name.replace("__", "_"))
    return m.group(1).upper().replace("_", "") if m else "?"


def search_models(query: str, limit: int = 15) -> list[tuple[str, int, int]]:
    """Возвращает [(repo_id, downloads, likes)]."""
    q = urllib.parse.quote(query)
    url = f"{HF_API}/models?search={q}&filter=gguf&sort=downloads&direction=-1&limit={limit}"
    try:
        with _open(url) as resp:
            data = json.load(resp)
    except Exception as exc:  # noqa: BLE001
        raise ModelError(f"Поиск не удался: {exc}") from exc
    out = []
    for m in data:
        mid = m.get("modelId") or m.get("id")
        s = m.get("stats", {})
        out.append((mid, s.get("downloads", 0), s.get("likes", 0)))
    return out


def _open(url: str):
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    return urllib.request.urlopen(req, timeout=30)


def model_info(repo_id: str) -> ModelInfo:
    """Инфа о модели + список GGUF-файлов с размерами."""
    url = f"{HF_API}/models/{urllib.parse.quote(repo_id)}"
    try:
        with _open(url) as resp:
            data = json.load(resp)
    except Exception as exc:  # noqa: BLE001
        raise ModelError(f"Не нашёл модель «{repo_id}»: {exc}") from exc

    info = ModelInfo(
        repo_id=data.get("modelId", repo_id),
        display_name=data.get("displayName") or repo_id,
        downloads=data.get("stats", {}).get("downloads", 0),
        likes=data.get("stats", {}).get("likes", 0),
    )
    files = data.get("siblings", [])
    download_url = f"https://huggingface.co/{info.repo_id}/resolve/main/"
    for f in files:
        rf = f.get("rfilename", "")
        if not rf.lower().endswith(".gguf"):
            continue
        size = 0
        try:  # HEAD за запросом на размер
            req = urllib.request.Request(download_url + urllib.parse.quote(rf), method="HEAD",
                                         headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=20) as r:
                size = int(r.headers.get("Content-Length", 0))
        except Exception:  # noqa: BLE001
            pass
        info.files.append(
            ModelFile(
                name=rf,
                size=size,
                url=download_url + urllib.parse.quote(rf),
                quant=_parse_quant(rf),
            )
        )
    return info


# ---------------------------------------------------------------------------
# Скачивание с докачкой
# ---------------------------------------------------------------------------

MODELS_DIR = Path.home() / ".llamatui" / "models"


def human_size(n: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if n < 1024 or unit == "TiB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{int(n)} B"
        n /= 1024
    return f"{n:.1f} TiB"


def download_model(file: ModelFile, dest_dir: Path | None = None, *, resume: bool = True) -> Path:
    """Скачивает GGUF в ~/.llamatui/models с докачкой и прогрессом tqdm."""
    dest_dir = dest_dir or MODELS_DIR
    dest_dir.mkdir(parents=True, exist_ok=True)
    fname = file.name.split("/")[-1]
    dest = dest_dir / fname
    part = dest.with_suffix(dest.suffix + ".part")

    existing = part.stat().st_size if (resume and part.exists()) else 0
    headers = {"User-Agent": USER_AGENT}
    if existing:
        headers["Range"] = f"bytes={existing}-"

    req = urllib.request.Request(file.url, headers=headers)
    try:
        resp = urllib.request.urlopen(req, timeout=30)
    except Exception as exc:  # noqa: BLE001
        raise ModelError(f"Не удалось начать скачивание: {exc}") from exc

    total = int(resp.headers.get("Content-Length", 0))
    mode = "ab" if resp.status == 206 else "wb"
    if mode == "wb":
        existing = 0
    overall = total + existing if total else file.size or 0

    bar = tqdm(
        total=overall or None,
        initial=existing,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        desc=f"[bold magenta]{fname}[/]",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{percentage:3.0f}%] {rate_fmt}{postfix} ⏳",
        dynamic_ncols=True,
    )
    try:
        with bar, open(part, mode) as fh:
            while chunk := resp.read(1 << 16):
                fh.write(chunk)
                bar.update(len(chunk))
    except KeyboardInterrupt:
        console.print("[yellow]Прервано — файл сохранён частично, повтор запустит докачку.[/]")
        raise
    finally:
        resp.close()

    part.rename(dest)
    return dest


# ---------------------------------------------------------------------------
# Красивый вывод (rich)
# ---------------------------------------------------------------------------

def render_search_results(results: list[tuple[str, int, int]]) -> Table:
    t = Table(title="🔎  Найденные GGUF-модели", title_style="bold cyan", header_style="bold white on grey23")
    t.add_column("#", style="dim", justify="right")
    t.add_column("Модель", style="bright_white")
    t.add_column("↓ Загрузок", justify="right", style="green")
    t.add_column("♥ Лайки", justify="right", style="red")
    for i, (rid, dl, lk) in enumerate(results, 1):
        t.add_row(str(i), rid, f"{dl:,}", f"{lk:,}")
    return t


def render_model_files(info: ModelInfo) -> Table:
    t = Table(
        title=f"📦  [bold]{info.display_name}[/] · ↓{info.downloads:,} ♥{info.likes:,}",
        title_style="",
        header_style="bold white on grey23",
    )
    t.add_column("#", style="dim", justify="right")
    t.add_column("Квант", style="bold yellow", no_wrap=True)
    t.add_column("Файл", style="bright_white")
    t.add_column("Размер", justify="right", style="cyan")
    for i, f in enumerate(info.gguf_files, 1):
        t.add_row(str(i), f.quant, f.name, human_size(f.size) if f.size else "—")
    return t
