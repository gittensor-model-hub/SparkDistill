"""Unit tests for the persistent validator worker protocol (no GPU)."""

from __future__ import annotations

import io
import json
from types import SimpleNamespace

from tritonbench.core import validator as validator_mod
from tritonbench.core.validator import TritonValidator
from tritonbench.core.validator_worker import wrap_code


def test_wrap_code_includes_pass_marker():
    wrapped = wrap_code("x = 1")
    assert "TRITONBENCH_PASS" in wrapped
    assert "x = 1" in wrapped
    assert "import torch" in wrapped


def test_session_disabled_skips_worker(monkeypatch):
    monkeypatch.setenv("TRITONBENCH_VALIDATOR_WORKER", "0")
    v = TritonValidator()
    started = []
    monkeypatch.setattr(v, "start_worker", lambda: started.append(True))
    with v.session():
        pass
    assert started == []


def test_session_starts_and_stops_worker(monkeypatch):
    monkeypatch.setenv("TRITONBENCH_VALIDATOR_WORKER", "1")
    v = TritonValidator()
    events = []
    monkeypatch.setattr(v, "start_worker", lambda: events.append("start"))
    monkeypatch.setattr(v, "stop_worker", lambda: events.append("stop"))
    with v.session():
        events.append("body")
    assert events == ["start", "body", "stop"]


def test_execute_via_worker_round_trip(monkeypatch):
    v = TritonValidator()
    monkeypatch.setattr(validator_mod, "require_blackwell_gpu", lambda *_a, **_k: None)

    class FakeStdout(io.StringIO):
        def __init__(self, lines: list[str]):
            super().__init__("".join(lines))

    responses = [
        json.dumps({"id": 1, "ok": True, "output": "TRITONBENCH_PASS\n"}) + "\n",
    ]
    fake_proc = SimpleNamespace(
        stdin=io.StringIO(),
        stdout=FakeStdout(responses),
        stderr=io.StringIO(""),
        poll=lambda: None,
        kill=lambda: None,
        wait=lambda timeout=None: 0,
    )
    # StringIO write goes to a buffer we can inspect after flush.
    written: list[str] = []

    class FakeStdin:
        def write(self, data: str) -> int:
            written.append(data)
            return len(data)

        def flush(self) -> None:
            pass

    fake_proc.stdin = FakeStdin()
    v._worker = fake_proc

    ok, output = v._execute_via_worker("print(1)", timeout=5)
    assert ok is True
    assert "TRITONBENCH_PASS" in output
    assert json.loads(written[0])["code"] == "print(1)"


def test_execute_via_worker_timeout_kills(monkeypatch):
    v = TritonValidator()
    monkeypatch.setattr(validator_mod, "require_blackwell_gpu", lambda *_a, **_k: None)
    killed = []

    fake_proc = SimpleNamespace(
        stdin=SimpleNamespace(write=lambda data: len(data), flush=lambda: None),
        stdout=SimpleNamespace(readline=lambda: ""),
        stderr=io.StringIO(""),
        poll=lambda: None,
        kill=lambda: killed.append(True),
        wait=lambda timeout=None: 0,
    )
    v._worker = fake_proc
    monkeypatch.setattr(v, "_readline_with_timeout", lambda timeout: None)

    ok, output = v._execute_via_worker("hang()", timeout=1)
    assert ok is False
    assert output == "TIMEOUT"
    assert killed == [True]
    assert v._worker is None


def test_execute_uses_worker_when_session_active(monkeypatch):
    v = TritonValidator()
    monkeypatch.setattr(validator_mod, "require_blackwell_gpu", lambda *_a, **_k: None)
    v._worker = SimpleNamespace(poll=lambda: None)
    monkeypatch.setattr(v, "_execute_via_worker", lambda code, timeout: (True, "via-worker"))
    monkeypatch.setattr(
        v,
        "_execute_subprocess",
        lambda code, timeout: (_ for _ in ()).throw(AssertionError("subprocess should not run")),
    )
    ok, output = v.execute("x=1", timeout=1)
    assert ok is True
    assert output == "via-worker"


def test_execute_falls_back_to_subprocess_without_session(monkeypatch):
    v = TritonValidator()
    monkeypatch.setattr(validator_mod, "require_blackwell_gpu", lambda *_a, **_k: None)
    monkeypatch.setattr(v, "_execute_subprocess", lambda code, timeout: (False, "subprocess"))
    ok, output = v.execute("x=1", timeout=1)
    assert ok is False
    assert output == "subprocess"
