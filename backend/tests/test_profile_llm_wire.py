"""Wire-level check of the persona LLM guard: the real OpenAI SDK against a localhost stub.

The unit tests in test_profile_llm_timeout.py script ``create_chat_completion``. This file keeps
the SDK, httpx and the response hook real, and only fakes the provider (a 127.0.0.1 server, no
network), because the two failure shapes differ exactly at the socket:

* **silent** — the provider accepts the request and never writes a byte. The socket (read)
  timeout catches this one.
* **drip** — the provider answers 200 and then keeps the connection busy without ever finishing
  the body. Bytes keep arriving, so the socket timeout never fires; only the wall-clock deadline
  bounds it.
"""
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

os.environ.setdefault("LLM_API_KEY", "test-key")

from app.services import oasis_profile_generator as opg  # noqa: E402

PERSONA = {"bio": "Search and cloud giant.", "persona": "Alphabet speaks through one account."}
CAP = 0.4


class _Provider(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_POST(self):  # noqa: N802 (http.server API)
        self.rfile.read(int(self.headers.get("Content-Length") or 0))
        self.server.requests += 1
        mode = self.server.mode
        try:
            if mode == "silent":
                self.server.release.wait(20)
                self.close_connection = True
            elif mode == "drip":
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()
                while not self.server.release.wait(0.05):
                    self.wfile.write(b"1\r\n \r\n")
                    self.wfile.flush()
                self.close_connection = True
            else:
                body = json.dumps({
                    "id": "cmpl-1", "object": "chat.completion", "created": 0, "model": "m",
                    "choices": [{"index": 0, "finish_reason": "stop",
                                 "message": {"role": "assistant", "content": json.dumps(PERSONA)}}],
                }).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
        except OSError:
            pass                      # the client gave up on us, which is the point

    def log_message(self, *args):
        pass


@pytest.fixture
def provider():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Provider)
    server.daemon_threads = True
    server.mode = "ok"
    server.requests = 0
    server.release = threading.Event()
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server
    server.release.set()
    server.shutdown()
    server.server_close()


class _Log:
    def __init__(self):
        self.lines = []

    def _add(self, msg, *args, **kwargs):
        self.lines.append(msg % args if args else msg)

    debug = info = warning = error = _add


def _gen(monkeypatch, provider, mode):
    provider.mode = mode
    monkeypatch.setenv("PROFILE_LLM_TIMEOUT_SECONDS", str(CAP))
    monkeypatch.setattr(opg, "backoff_delay", lambda attempt: 0.0)
    log = _Log()
    monkeypatch.setattr(opg, "logger", log)
    gen = opg.OasisProfileGenerator(
        api_key="k", base_url=f"http://127.0.0.1:{provider.server_address[1]}/v1", model_name="m")
    return gen, log


def _llm(gen):
    return gen._generate_profile_with_llm(
        entity_name="GOOGL", entity_type="Company", entity_summary="GOOGL summary",
        entity_attributes={}, context="", node_degree=55)


def test_a_healthy_provider_answers_and_the_first_byte_is_logged(monkeypatch, provider):
    gen, log = _gen(monkeypatch, provider, "ok")

    result = _llm(gen)

    assert result["bio"] == PERSONA["bio"]
    assert gen.profile_degradations == []
    events = [line.split()[1] for line in log.lines if line.startswith("PROFILE_LLM ")]
    assert events == ["start", "first_byte", "end"]
    assert any("first_byte" in line and "status=200" in line for line in log.lines)


def test_a_silent_provider_costs_three_bounded_attempts_then_a_flagged_stub(monkeypatch, provider):
    gen, log = _gen(monkeypatch, provider, "silent")

    started = time.monotonic()
    result = _llm(gen)
    elapsed = time.monotonic() - started

    assert elapsed < 3 * CAP + 2.0
    assert provider.requests == 3                         # no hidden SDK retries on top of ours
    assert result["persona"] == "GOOGL summary"
    assert [d["reason"] for d in gen.profile_degradations] == ["timeout"]
    assert not any("first_byte" in line for line in log.lines)   # it never answered at all


def test_the_socket_timeout_alone_does_not_bound_a_dripping_provider(monkeypatch, provider):
    """Why the wall-clock deadline exists: with bytes arriving every 50 ms the read timeout
    (0.4 s here) never fires, so the bare SDK call is still blocked long after it."""
    gen, _ = _gen(monkeypatch, provider, "drip")
    finished = threading.Event()

    def bare_call():
        try:
            gen.client.chat.completions.create(
                model="m", messages=[{"role": "user", "content": "hi"}])
        except Exception:
            pass
        finished.set()

    threading.Thread(target=bare_call, daemon=True).start()

    assert not finished.wait(4 * CAP), "the socket timeout unexpectedly ended a dripping call"


def test_a_dripping_provider_is_bounded_by_the_wall_clock_deadline(monkeypatch, provider):
    gen, log = _gen(monkeypatch, provider, "drip")

    started = time.monotonic()
    result = _llm(gen)
    elapsed = time.monotonic() - started

    assert elapsed < 3 * CAP + 2.0
    assert result["persona"] == "GOOGL summary"
    assert gen.profile_degradations == [
        {"entity": "GOOGL", "entity_type": "Company", "node_degree": 55, "reason": "timeout"}]
    # headers did arrive: the log now tells a dripping body apart from a request never answered
    assert sum("first_byte" in line for line in log.lines) == 3
