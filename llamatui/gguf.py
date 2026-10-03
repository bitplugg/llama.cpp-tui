"""Чтение метаданных GGUF-файлов (без внешних зависимостей).

Формат: https://github.com/ggml-org/ggml/blob/master/docs/gguf.md
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from pathlib import Path

MAGIC = b"GGUF"

# GGUFTYPE
T_UINT8, T_INT8, T_UINT16, T_INT16, T_UINT32, T_INT32 = range(6)
T_FLOAT32, T_BOOL, T_STRING, T_ARRAY, T_UINT64, T_INT64, T_FLOAT64 = range(6, 13)

_SCALAR_FMT = {
    T_UINT8: "<B", T_INT8: "<b", T_UINT16: "<H", T_INT16: "<h",
    T_UINT32: "<I", T_INT32: "<i", T_UINT64: "<Q", T_INT64: "<q",
    T_FLOAT32: "<f", T_FLOAT64: "<d", T_BOOL: "<?",
}


class GGUFError(ValueError):
    pass


@dataclass
class GGUFMetadata:
    version: int
    n_tensors: int
    kv: dict[str, object] = field(default_factory=dict)

    # удобные свойства -------------------------------------------------------
    @property
    def arch(self) -> str:
        return str(self.kv.get("general.architecture", "?"))

    @property
    def name(self) -> str:
        return str(self.kv.get("general.name", "безымянная"))

    @property
    def quant(self) -> str:
        return str(self.kv.get("general.file_type", ""))

    @property
    def ctx_train(self) -> int:
        return int(self.kv.get(f"{self.arch}.context_length", 0))

    @property
    def n_layers(self) -> int:
        return int(self.kv.get(f"{self.arch}.block_count", 0))

    @property
    def n_params(self) -> str:
        for k in (f"{self.arch}.embedding_length", ):
            pass
        total = self.kv.get(f"{self.arch}.tensor_count", None)
        return str(total) if total else "?"

    @property
    def eos(self) -> str:
        toks = self.kv.get(f"{self.arch}.tokenizer.ggml.eos_token_id")
        names = self.kv.get(f"{self.arch}.tokenizer.ggml.tokens")
        if isinstance(toks, list) and toks and isinstance(names, list):
            try:
                return str(names[int(toks[0])])
            except (IndexError, ValueError):
                pass
        if isinstance(toks, int) and isinstance(names, list):
            try:
                return str(names[toks])
            except IndexError:
                pass
        return ""

    @property
    def chat_template(self) -> str:
        return str(self.kv.get(f"{self.arch}.tokenizer.chat_template", ""))

    @property
    def has_vision(self) -> bool:
        return any(k.endswith(".mmproj") or k.startswith("mmp.") for k in ()) or \
               self.arch in ("qwen2vl", "llava", "gemma3", "internvl2", "minicpmv")


def read_gguf_metadata(path: Path | str, max_kv_bytes: int = 64 << 20) -> GGUFMetadata:
    """Читает заголовок и KV-пары. Останавливается на списке тензоров (экономно)."""
    path = Path(path)
    with open(path, "rb") as fh:
        head = fh.read(12)
        if len(head) < 12 or head[:4] != MAGIC:
            raise GGUFError(f"{path.name}: это не GGUF-файл")
        (version,) = struct.unpack("<I", head[4:8])
        if version < 2:
            raise GGUFError(f"Слишком старый GGUF v{version} (нужен v2/v3)")
        (n_tensors,) = struct.unpack("<Q", head[8:12])
        (n_kv,) = struct.unpack("<Q", fh.read(8))

        md = GGUFMetadata(version=version, n_tensors=n_tensors)
        pos = 24
        for _ in range(n_kv):
            key = _read_string(fh)
            vtype = struct.unpack("<I", fh.read(4))[0]
            value = _read_value(fh, vtype, limit=pos + max_kv_bytes)
            md.kv[key] = value
        return md


def _read_string(fh) -> str:
    (length,) = struct.unpack("<Q", fh.read(8))
    if length > (1 << 26):  # >64 MiB — явный мусор
        raise GGUFError("битая строка в заголовке")
    return fh.read(length).decode("utf-8", errors="replace")


def _read_value(fh, vtype: int, *, limit: int):
    if vtype in _SCALAR_FMT:
        fmt = _SCALAR_FMT[vtype]
        size = struct.calcsize(fmt)
        if fh.tell() + size > limit:
            raise GGUFError("KV-данные слишком большие")
        return struct.unpack(fmt, fh.read(size))[0]
    if vtype == T_STRING:
        return _read_string(fh)
    if vtype == T_ARRAY:
        (elem_type,) = struct.unpack("<I", fh.read(4))
        (count,) = struct.unpack("<Q", fh.read(8))
        # огромные массивы токенов читаем, но обрезаем вывод наружу не передаём целиком
        cap = 4096
        arr = []
        for i in range(count):
            if i >= cap:
                # дочитывать лень — пропускаем остаток по фикс.размеру если можем
                fmt = _SCALAR_FMT.get(elem_type)
                if fmt:
                    fh.read(struct.calcsize(fmt) * (count - cap))
                    break
                raise GGUFError("слишком большой массив переменных элементов")
            arr.append(_read_value(fh, elem_type, limit=limit))
        if count > cap:
            arr.append(f"…(+{count - cap})")
        return arr
    raise GGUFError(f"неизвестный тип {vtype}")


# ---------------------------------------------------------------------------
# Красивый рендер (rich)
# ---------------------------------------------------------------------------

from rich.text import Text  # noqa: E402


def render_model_card(md: GGUFMetadata, path: Path, console_obj) -> "object":
    from rich.panel import Panel
    from rich.table import Table

    t = Table.grid(padding=(0, 2))
    t.add_column(style="bold cyan", justify="right")
    t.add_column()
    size_gb = path.stat().st_size / 1e9
    t.add_row("архитектура", f"[bold]{md.arch}[/] · GGUF v{md.version}")
    t.add_row("название", md.name)
    t.add_row("размер", f"{size_gb:.2f} GB · тензоров: {md.n_tensors:,}")
    t.add_row("слоёв", str(md.n_layers))
    t.add_row("контекст (обучен)", f"{md.ctx_train:,}" if md.ctx_train else "—")
    if md.eos:
        t.add_row("EOS-токен", f"[yellow]{md.eos}[/]")
    if md.has_vision:
        t.add_row("vision", "[green]да (нужен mmproj)[/]")
    tmpl = md.chat_template
    body = t
    if tmpl:
        preview = tmpl if len(tmpl) <= 600 else tmpl[:600] + " …"
        from rich.syntax import Syntax
        try:
            syntax = Syntax(preview, "jinja2", theme="one-dark", word_wrap=True)
        except Exception:  # noqa: BLE001 — мало ли какой pygments установлен
            syntax = Text(preview, style="dim")
        body = Table.grid()
        body.add_row(t)
        body.add_row(Panel(
            syntax,
            title="[dim]chat template (фрагмент)[/]", border_style="grey35", padding=(0, 1),
        ))
    return Panel(body, title=f"🧠  [bold bright_white]{path.name}[/]",
                 title_align="left", border_style="bright_magenta", padding=(1, 2))
