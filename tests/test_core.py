"""Быстрые тесты без сети и без настоящих бинарников llama.cpp."""
import io, struct, sys, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from llamatui.backend import ChatMessage, GenStats, ServerBackend
from llamatui.gguf import GGUFError, read_gguf_metadata
from llamatui.manifest import (Manifest, default_manifest_name, manifest_from_gguf)
from llamatui.models import _parse_quant, human_size
from llamatui.ui import GenerationProgress, show_banner


# ---------- мини-писатель GGUF ----------

def _s(x: str) -> bytes:
    b = x.encode()
    return struct.pack("<Q", len(b)) + b


def write_gguf(path: Path, kv: dict, n_tensors: int = 3):
    buf = io.BytesIO()
    buf.write(b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", n_tensors)
              + struct.pack("<Q", len(kv)))
    for k, v in kv.items():
        buf.write(_s(k))
        if isinstance(v, bool):
            buf.write(struct.pack("<I", 7) + struct.pack("<?", v))
        elif isinstance(v, int):
            buf.write(struct.pack("<I", 4) + struct.pack("<I", v))
        elif isinstance(v, float):
            buf.write(struct.pack("<I", 6) + struct.pack("<f", v))
        elif isinstance(v, list):
            if v and isinstance(v[0], str):
                buf.write(struct.pack("<I", 8) + struct.pack("<Q", len(v)))
                for s in v:
                    buf.write(_s(s))
            else:
                buf.write(struct.pack("<I", 9) + struct.pack("<I", 4) + struct.pack("<Q", len(v)))
                for i in v:
                    buf.write(struct.pack("<I", i))
        else:
            buf.write(struct.pack("<I", 8) + _s(str(v)))
    path.write_bytes(buf.getvalue() + b"\x00" * 256)


def test_gguf_roundtrip(tmp_path: Path):
    p = tmp_path / "m.gguf"
    write_gguf(p, {
        "general.architecture": "qwen2",
        "general.name": "Test Model",
        "qwen2.context_length": 32768,
        "qwen2.block_count": 28,
        "tokenizer.ggml.tokens": ["a", "b", "c"],
        "qwen2.tokenizer.ggml.eos_token_id": 2,
        "qwen2.tokenizer.chat_template": "{% for m in messages %}{{ m.content }}{% endfor %}",
    })
    md = read_gguf_metadata(p)
    assert md.arch == "qwen2"
    assert md.n_layers == 28
    assert md.ctx_train == 32768
    assert md.eos == "c"
    assert md.chat_template.startswith("{%")
    bad = tmp_path / "bad.gguf"
    bad.write_bytes(b"NOTGGUF" * 10)
    try:
        read_gguf_metadata(bad)
        raise AssertionError("должен был вылететь GGUFError")
    except GGUFError:
        pass
    print("✔ gguf roundtrip")


def test_manifest(tmp_path: Path):
    # 1) из GGUF с шаблоном и sampler-параметрами
    p = tmp_path / "qwen2.5-coder-7b-instruct-q4_k_m.gguf"
    write_gguf(p, {
        "general.architecture": "qwen2",
        "general.name": "Qwen2.5 Coder",
        "qwen2.context_length": 32768,
        "qwen2.block_count": 28,
        "qwen2.tokenizer.chat_template": "{% for m in messages %}<|im_start|>{{m.role}}{% endfor %}",
        "sampler.temp": 0.5,
        "sampler.top_k": 40,
        "sampler.top_p": 0.9,
    })
    man, hints = manifest_from_gguf(p)
    txt = man.render()
    assert txt.startswith(f"FROM {p}")
    assert "TEMPLATE" in txt and "im_start" in txt
    assert "PARAMETER temperature 0.5" in txt
    assert "PARAMETER top_k 40" in txt
    assert "num_ctx 32768" in txt
    assert any("архитектура" in h for h in hints)

    # имя по умолчанию: qwen2.5-coder-7b-instruct-q4_k_m.gguf -> ...:q4_k_m
    name = default_manifest_name(p)
    assert name == "qwen2.5-coder-7b-instruct:q4_k_m", name

    # запись/перезапись файла манифеста
    out = man.write(tmp_path / "sub" / "Modelfile")
    assert out.read_text(encoding="utf-8") == txt

    # 2) GGUF без чат-шаблона -> предупреждение в hints, TEMPLATE отсутствует
    p2 = tmp_path / "bare.gguf"
    write_gguf(p2, {"general.architecture": "llama", "llama.context_length": 2048})
    man2, hints2 = manifest_from_gguf(p2)
    assert "TEMPLATE" not in man2.render()
    assert any("chat_template" in h for h in hints2)
    assert default_manifest_name(p2) == "bare:latest"

    # 3) ручной манифест: SYSTEM/LICENSE/MESSAGE/ADAPTER/кавочки для строк
    m = Manifest(from_="qwen2", system="Ты — кот.", license="MIT",
                 adapter="lora.gguf", parameters={"stop": "### x", "mirostat": 2},
                 messages=[("user", "привет"), ("assistant", "мяу")])
    r = m.render()
    assert 'SYSTEM """' in r and "Ты — кот." in r
    assert "LICENSE" in r and "ADAPTER lora.gguf" in r
    assert 'MESSAGE user """' in r and "мяу" in r
    assert 'PARAMETER stop """### x"""' in r and "PARAMETER mirostat 2" in r
    print("✔ ollama manifest")


def test_helpers():
    assert _parse_quant("model-Q4_K_M.gguf") == "Q4KM"
    assert _parse_quant("something-IQ3_S.gguf") == "IQ3S"
    assert human_size(3_500_000_000).startswith("3.3")
    st = GenStats(t_start=time.monotonic() - 1.0, t_first_token=time.monotonic() - 0.5,
                  t_end=time.monotonic(), tokens_out=10)
    assert 5 < st.tok_per_sec < 30 and st.ttft and st.ttft < 1
    print("✔ helpers")


def test_ui_render():
    show_banner("0.1.0")
    prog = GenerationProgress(char_hint=100)
    st = GenStats(t_start=time.monotonic())
    with prog:
        prog.feed(40, st)
    print("✔ ui render")


# ---------- фейковый llama-server для SSE ----------
class FakeAPI(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path == "/health":
            self.send_response(200); self.end_headers(); self.wfile.write(b'{"status":"ok"}')
        elif self.path == "/tokenize":
            self.send_response(200); self.end_headers(); self.wfile.write(b'{"tokens":[1,2,3]}')
        elif self.path in ("/v1/models", "/models", "/api/tags"):
            payload = b'{"data":[{"id":"fake"}]}' if self.path != "/api/tags" \
                else b'{"models":[{"name":"qwen-test:7b"},{"name":"llama3.2:latest"}]}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json"); self.end_headers()
            self.wfile.write(payload)
        else:
            self.send_response(404); self.end_headers()

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        self.rfile.read(length)
        if self.path == "/v1/chat/completions":
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream"); self.end_headers()
            events = [
                '{"choices":[{"delta":{"content":"При"}}]}',
                '{"choices":[{"delta":{"content":"вет!"}}]}',
                '{"choices":[{"delta":{"content":""},"finish_reason":"eos_token"},'
                '"usage":{"prompt_tokens":7,"completion_tokens":2}}',
                "[DONE]",
            ]
            for e in events:
                self.wfile.write(f"data: {e}\n\n".encode())
                self.wfile.flush()
        elif self.path == "/api/chat":
            # NDJSON-стриминг как у Ollama
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson"); self.end_headers()
            events = [
                '{"message":{"role":"assistant","content":"При"},"done":false}',
                '{"message":{"role":"assistant","content":"вет!"},"done":false}',
                '{"message":{"role":"assistant","content":""},"done":true,'
                '"done_reason":"stop","prompt_eval_count":5,"eval_count":2}',
            ]
            for e in events:
                self.wfile.write((e + "\n").encode())
                self.wfile.flush()
        elif self.path == "/api/generate":
            self.send_response(200)
            self.send_header("Content-Type", "application/json"); self.end_headers()
            self.wfile.write(b'{"response":"","done":true,"prompt_eval_count":9}')
        else:
            self.send_response(404); self.end_headers()


def _fake_server():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), FakeAPI)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}"


def test_server_backend_sse():
    srv, _ = _fake_server()
    be = ServerBackend.__new__(ServerBackend)   # без запуска реального процесса
    be.model = Path("fake.gguf"); be.port = srv.server_address[1]; be.temp = 0.7
    gen, stats = be.chat_stream([ChatMessage("user", "привет")])
    out = "".join(gen)
    assert out == "Привет!", out
    assert stats.prompt_tokens == 7 and stats.completion_tokens_api == 2
    assert stats.stopped_by_eos and stats.tokens_out == 2 and stats.tok_per_sec > 0
    assert be.tokenize_count("x") == 3
    srv.shutdown()
    print("✔ SSE backend (стриминг/usage/eos)")


def test_ollama_backend():
    from llamatui.backend import OllamaBackend, OpenAICompatBackend
    srv, base = _fake_server()
    ob = OllamaBackend.__new__(OllamaBackend)   # минуем ensure_up
    OpenAICompatBackend.__init__(ob, base, "qwen-test:7b", temp=0.7)
    assert ob.ping() and ob.model_exists()
    assert set(ob.list_models()) == {"qwen-test:7b", "llama3.2:latest"}
    gen, stats = ob.chat_stream([ChatMessage("user", "привет")])
    out = "".join(gen)
    assert out == "Привет!", out
    assert stats.prompt_tokens == 5 and stats.completion_tokens_api == 2
    assert stats.stopped_by_eos and stats.tok_per_sec >= 0
    assert ob.count_tokens("hi") == 9
    srv.shutdown()
    print("✔ ollama backend (NDJSON /api/chat, tags, count_tokens)")


def test_openai_compat_backend():
    from llamatui.backend import OpenAICompatBackend
    srv, base = _fake_server()
    ob = OpenAICompatBackend(base + "/v1", "fake", temp=0.7)
    gen, stats = ob.chat_stream([ChatMessage("user", "привет")])
    out = "".join(gen)
    assert out == "Привет!", out
    assert stats.prompt_tokens == 7 and stats.stopped_by_eos
    srv.shutdown()
    print("✔ openai-compat backend (SSE через base_url/v1)")


if __name__ == "__main__":
    import tempfile
    test_helpers()
    test_ui_render()
    with tempfile.TemporaryDirectory() as td:
        test_gguf_roundtrip(Path(td))
        test_manifest(Path(td))
    test_server_backend_sse()
    test_ollama_backend()
    test_openai_compat_backend()
    print("\nВСЕ ТЕСТЫ ПРОШЛИ ✅")
