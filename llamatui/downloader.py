"""Поиск исполняемых файлов llama.cpp и скачивание готовых бинарников через tqdm."""

from __future__ import annotations

import os
import platform
import re
import shutil
import stat
import tarfile
import tempfile
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

from rich.console import Console
from tqdm import tqdm

console = Console()

# ---------------------------------------------------------------------------
# Где искать / куда класть бинарники
# ---------------------------------------------------------------------------

BIN_NAMES = ("llama-cli", "llama-server")

_LLAMA_DIR = Path(os.environ.get("LLAMATUI_HOME", Path.home() / ".llamatui"))


def llama_dir() -> Path:
    _LLAMA_DIR.mkdir(parents=True, exist_ok=True)
    return _LLAMA_DIR


def _candidate_paths(name: str) -> list[Path]:
    """Все места, где может лежать `name`, в порядке приоритета."""
    env_var = name.upper().replace("-", "_")
    cands: list[Path] = []
    if v := os.environ.get(env_var):
        cands.append(Path(v))
    cands.append(llama_dir() / name)
    if p := shutil.which(name):
        cands.append(Path(p))
    # частые места ручной сборки
    for root in ("/usr/local/bin", "/opt/llama.cpp/bin", str(Path.home() / ".local" / "bin")):
        cands.append(Path(root) / name)
    return cands


def find_binary(name: str) -> Path | None:
    for c in _candidate_paths(name):
        if c.is_file() and os.access(c, os.X_OK):
            return c
    return None


def find_all() -> dict[str, Path | None]:
    return {n: find_binary(n) for n in BIN_NAMES}


# ---------------------------------------------------------------------------
# Автодополнение имени бинарника (на случай форков/переименований)
# ---------------------------------------------------------------------------

def resolve_cli_binary(preferred: str | None = None) -> Path | None:
    names = [preferred] if preferred else []
    names += ["llama-cli", "cli", "main", "llama-inference"]
    for n in names:
        if p := find_binary(n):
            return p
    return None


# ---------------------------------------------------------------------------
# Определение подходящего релиза под данную систему
# ---------------------------------------------------------------------------

_TAG_RE = re.compile(r"b\d{4,6}")


@dataclass
class SystemTarget:
    asset: str          # имя архива в релизе
    folder: str         # подпапка внутри архива
    label: str          # человекочитаемое описание


def detect_target(assets: list[str], force_cpu: bool = False) -> SystemTarget:
    """Выбирает самый подходящий ассет релиза под текущую ОС/арх/фичи."""
    system = platform.system().lower()
    machine = platform.machine().lower()
    arm = machine in ("arm64", "aarch64", "arm")

    def has(pattern: str) -> bool:
        return any(re.search(pattern, a) for a in assets)

    def pick(pattern: str) -> str | None:
        rx = re.compile(pattern)
        for a in assets:
            if rx.search(a):
                return a
        return None

    if system == "darwin":
        if arm:
            asset = pick(r"macos-arm64-apple-metal\b") or pick(r"macos-arm64")
            return SystemTarget(asset or "", "llama-bxxx-bin-macos-arm64", "macOS · Apple Silicon (Metal)")
        asset = pick(r"macos-x64")
        return SystemTarget(asset or "", "llama-bxxx-bin-macos-x64", "macOS · Intel (CPU)")

    if system == "windows":
        if not force_cpu and has(r"cuda"):
            asset = pick(r"win-x64-cuda")
            return SystemTarget(asset or "", "llama-bxxx-bin-win-cuda-x64", "Windows · CUDA")
        asset = pick(r"win-x64\b")
        return SystemTarget(asset or "", "llama-bxxx-bin-win-cpu-x64", "Windows · CPU")

    # Linux и прочее
    if not force_cpu:
        if has(r"cpu-only"):  # без gpu-ассетов — только CPU-сборка
            asset = pick(r"linux-(x64|arm64)-cpu-only")
            folder = "llama-bxxx-bin-linux-x64-cpu-only"
            return SystemTarget(asset or "", folder, "Linux · CPU")
        vulkan = pick(r"vulkan")
        cuda = pick(r"cuda")
        rocm = pick(r"rocm")
        chosen_label = "Linux · Vulkan"
        chosen = vulkan
        # эвристика: есть ли у пользователя CUDA/ROCm библиотеки
        if _has_lib("libcuda.so") and cuda:
            chosen, chosen_label = cuda, "Linux · CUDA"
        elif _has_lib("libhsa-runtime") and rocm:
            chosen, chosen_label = rocm, "Linux · ROCm"
        arch = "arm64" if arm else "x64"
        if chosen is None:
            chosen = pick(rf"linux-{arch}-cpu-only") or pick(rf"linux-{arch}")
            chosen_label = f"Linux · CPU ({arch})"
        folder = f"llama-bxxx-bin-linux-{'arm64' if arm else 'x64'}"
        return SystemTarget(chosen or "", folder, chosen_label)

    return SystemTarget("", "", "неизвестная платформа")


def _has_lib(substr: str) -> bool:
    for d in ("/usr/lib", "/usr/lib64", "/usr/local/cuda/lib64", "/opt/rocm/lib"):
        p = Path(d)
        if p.is_dir():
            try:
                if any(substr in f for f in os.listdir(p)):
                    return True
            except OSError:
                pass
    return False


# ---------------------------------------------------------------------------
# Скачивание
# ---------------------------------------------------------------------------

_LATEST_RELEASE_URL = "https://api.github.com/repos/ggml-org/llama.cpp/releases/latest"
_RELEASE_ASSETS_PAGE = "https://github.com/ggml-org/llama.cpp/releases/latest"


def fetch_release_info() -> tuple[str, list[dict]]:
    """Возвращает (tag, список ассетов [{'name', 'browser_download_url', 'size'}])."""
    req = urllib.request.Request(
        _LATEST_RELEASE_URL,
        headers={"User-Agent": "llamatui", "Accept": "application/vnd.github+json"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        import json

        data = json.load(resp)
    tag = data.get("tag_name", "?")
    assets = [
        {
            "name": a["name"],
            "browser_download_url": a["browser_download_url"],
            "size": a.get("size", 0),
        }
        for a in data.get("assets", [])
    ]
    return tag, assets


class DownloadError(RuntimeError):
    pass


def download_asset(url: str, dest: Path, *, size_hint: int = 0) -> Path:
    """Скачивает файл с прогресс-баром tqdm."""
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url, timeout=60) as resp, open(tmp, "wb") as fh:
        total = int(resp.headers.get("Content-Length") or size_hint or 0)
        bar = tqdm(
            total=total or None,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc=f"[bold cyan]{dest.name}[/]",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{percentage:3.0f}%] {rate_fmt}{postfix}",
            dynamic_ncols=True,
        )
        with bar:
            while chunk := resp.read(65536):
                fh.write(chunk)
                bar.update(len(chunk))
    tmp.rename(dest)
    return dest


def extract_archive(archive: Path, workdir: Path) -> Path:
    """Распаковывает .tar.gz/.zip, возвращает корневую папку содержимого."""
    if archive.name.endswith((".tar.gz", ".tgz")):
        with tarfile.open(archive, "r:gz") as tf:
            members = tf.getnames()
            tf.extractall(workdir, filter="data")
    elif archive.name.endswith(".zip"):
        with zipfile.ZipFile(archive) as zf:
            members = zf.namelist()
            zf.extractall(workdir)
    else:
        raise DownloadError(f"Неизвестный формат архива: {archive.name}")
    roots = {m.split("/")[0] for m in members if "/" in m}
    if len(roots) == 1:
        return workdir / roots.pop()
    return workdir


def install_binaries(src_dir: Path, names: tuple[str, ...] = BIN_NAMES) -> dict[str, Path]:
    """Копирует нужные бинарники из распакованной папки в ~/.llamatui."""
    target_dir = llama_dir()
    installed: dict[str, Path] = {}
    found = {p.name: p for p in src_dir.rglob("*") if p.is_file()}
    for name in names:
        exe = found.get(name)
        if exe is None:  # windows-экзешники
            exe = found.get(f"{name}.exe")
        if exe is None:
            continue
        dest = target_dir / exe.name
        shutil.copy2(exe, dest)
        dest.chmod(dest.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
        # подтянуть рядом лежащие .dll/.so — кладём в ту же папку
        for sibling in src_dir.rglob("*"):
            if sibling.is_file() and sibling.parent == exe.parent and sibling.suffix in (".dll", ".so", ".dylib"):
                shutil.copy2(sibling, target_dir / sibling.name)
        installed[name] = dest
    return installed


@dataclass
class InstallResult:
    tag: str
    target_label: str
    installed: dict[str, Path] = field(default_factory=dict)


def auto_install(force_cpu: bool = False) -> InstallResult:
    """Скачивает и устанавливает последние бинарники llama.cpp. Блокирующий вызов."""
    console.print("[dim]Ищу последний релиз llama.cpp…[/]")
    try:
        tag, assets = fetch_release_info()
    except Exception as exc:  # noqa: BLE001
        raise DownloadError(
            f"Не удалось получить инфо о релизе ({exc}).\n"
            f"Открой {_RELEASE_ASSETS_PAGE}, скачай архив сам и распакуй в {llama_dir()}"
        ) from exc

    names = [a["name"] for a in assets]
    tgt = detect_target(names, force_cpu=force_cpu)
    asset = next((a for a in assets if a["name"] == tgt.asset), None)
    if asset is None:
        raise DownloadError(
            f"Не нашёл подходящий ассет для: {tgt.label}\n"
            f"Скачай вручную со {_RELEASE_ASSETS_PAGE} и распакуй в {llama_dir()}"
        )

    console.print(f"Релиз: [bold green]{tag}[/] · сборка: [bold]{tgt.label}[/]")
    with tempfile.TemporaryDirectory(prefix="llamatui-") as td:
        tdp = Path(td)
        archive = tdp / asset["name"]
        download_asset(asset["browser_download_url"], archive, size_hint=asset["size"])
        root = extract_archive(archive, tdp)
        # имя папки вида llama-b1234-... — переписываем generic-папку из таргета
        installed = install_binaries(root)
    if not installed:
        raise DownloadError("В архиве не нашлось ни llama-cli, ни llama-server 😢")
    return InstallResult(tag=tag, target_label=tgt.label, installed=installed)
