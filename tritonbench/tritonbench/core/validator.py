"""Validate generated Triton kernel responses."""

from __future__ import annotations

import ast
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bench_config import PY_INTERPRETER, require_blackwell_gpu  # noqa: E402

PASS_MARKER = "TRITONBENCH_PASS"
_WORKER_READY_TIMEOUT_S = 600.0


def _worker_enabled() -> bool:
    raw = os.environ.get("TRITONBENCH_VALIDATOR_WORKER", "1").strip().lower()
    return raw not in {"0", "false", "no", "off"}


class TritonValidator:
    def __init__(self, triton_version: str = "3.7.1", *, gpu_index: int = 0):
        self.version = triton_version
        self.gpu_index = gpu_index
        self.deprecated_apis = [
            "tl.make_block_ptr",
            "tl.advance",
        ]
        self.modern_apis: dict[str, list[str]] = {
            "tensor_descriptors": ["tl.make_tensor_descriptor", "desc.load", "desc.store"],
            "fp8_types": ["tl.float8e4nv", "tl.float8e5m2"],
            "scan_reduce": ["tl.associative_scan", "tl.reduce"],
            "compiler_hints": ["num_stages", "num_warps", "num_ctas"],
        }
        self._worker: subprocess.Popen[str] | None = None
        self._next_job_id = 0

    def extract_code(self, response: str) -> str:
        for pattern in (r"```python\n(.*?)```", r"```\n(.*?)```"):
            matches = re.findall(pattern, response, re.DOTALL)
            if matches:
                return "\n\n".join(matches)
        return response

    def check_syntax(self, code: str) -> bool:
        try:
            ast.parse(code)
            return True
        except SyntaxError:
            return False

    def check_triton_api(self, code: str) -> dict[str, Any]:
        result: dict[str, Any] = {
            "modern": True,
            "issues": [],
            "features_used": [],
            "deprecated_used": [],
        }
        for dep in self.deprecated_apis:
            if dep in code:
                result["deprecated_used"].append(dep)
                result["issues"].append(f"Uses deprecated API: {dep}")
        for category, apis in self.modern_apis.items():
            for api in apis:
                if api in code:
                    result["features_used"].append(f"{category}: {api}")
        for pattern, msg in {
            "@triton.jit": "Missing @triton.jit decorator",
            "tl.program_id": "Missing tl.program_id",
        }.items():
            if pattern not in code:
                result["issues"].append(msg)
                result["modern"] = False
        if result["deprecated_used"]:
            result["modern"] = False
        return result

    @contextmanager
    def session(self) -> Iterator["TritonValidator"]:
        """Keep a warm GPU worker alive across many ``execute`` calls.

        Falls back to the legacy per-kernel subprocess path when the worker is
        disabled via ``TRITONBENCH_VALIDATOR_WORKER=0``.
        """
        if not _worker_enabled():
            yield self
            return
        self.start_worker()
        try:
            yield self
        finally:
            self.stop_worker()

    def start_worker(self) -> None:
        if self._worker is not None and self._worker.poll() is None:
            return
        require_blackwell_gpu(self.gpu_index)
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(self.gpu_index)
        env["TRITON_PRINT_AUTOTUNING"] = "0"
        # Prefer ``-m`` so the package import path matches the rest of TritonBench.
        command = [
            PY_INTERPRETER,
            "-m",
            "tritonbench.core.validator_worker",
            "--gpu-index",
            "0",  # already remapped via CUDA_VISIBLE_DEVICES
        ]
        self._worker = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=env,
            cwd=str(REPO_ROOT),
        )
        self._await_worker_ready()

    def stop_worker(self) -> None:
        worker = self._worker
        self._worker = None
        if worker is None:
            return
        try:
            if worker.poll() is None and worker.stdin is not None:
                worker.stdin.write(json.dumps({"shutdown": True}) + "\n")
                worker.stdin.flush()
                try:
                    worker.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    worker.kill()
                    worker.wait(timeout=5)
            elif worker.poll() is None:
                worker.kill()
                worker.wait(timeout=5)
        except (OSError, BrokenPipeError):
            try:
                worker.kill()
            except OSError:
                pass

    def _await_worker_ready(self) -> None:
        assert self._worker is not None and self._worker.stdout is not None
        deadline = time.monotonic() + _WORKER_READY_TIMEOUT_S
        while time.monotonic() < deadline:
            if self._worker.poll() is not None:
                stderr = ""
                if self._worker.stderr is not None:
                    stderr = self._worker.stderr.read() or ""
                raise RuntimeError(f"validator worker exited early (code {self._worker.returncode}): {stderr[-2000:]}")
            line = self._worker.stdout.readline()
            if not line:
                time.sleep(0.05)
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if payload.get("ready"):
                return
        self.stop_worker()
        raise TimeoutError(f"validator worker not ready after {_WORKER_READY_TIMEOUT_S}s")

    def _execute_via_worker(self, code: str, timeout: int) -> tuple[bool, str]:
        if self._worker is None or self._worker.poll() is not None:
            self.start_worker()
        assert self._worker is not None and self._worker.stdin is not None and self._worker.stdout is not None

        self._next_job_id += 1
        job_id = self._next_job_id
        request = json.dumps({"id": job_id, "code": code, "timeout_s": timeout})
        try:
            self._worker.stdin.write(request + "\n")
            self._worker.stdin.flush()
        except (BrokenPipeError, OSError):
            self.stop_worker()
            self.start_worker()
            assert self._worker is not None and self._worker.stdin is not None and self._worker.stdout is not None
            self._worker.stdin.write(request + "\n")
            self._worker.stdin.flush()

        line = self._readline_with_timeout(timeout)
        if line is None:
            self._kill_worker_hard()
            return False, "TIMEOUT"
        if self._worker is not None and self._worker.poll() is not None and not line:
            stderr = ""
            if self._worker.stderr is not None:
                stderr = self._worker.stderr.read() or ""
            self.stop_worker()
            return False, f"WORKER_CRASH: {stderr[-2000:]}"
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            self._kill_worker_hard()
            return False, f"WORKER_BAD_RESPONSE: {line[:500]}"
        if payload.get("id") != job_id:
            self._kill_worker_hard()
            return False, f"WORKER_ID_MISMATCH: expected {job_id}, got {payload.get('id')!r}"
        return bool(payload.get("ok")), str(payload.get("output") or "")

    def _readline_with_timeout(self, timeout: int) -> str | None:
        """Read one stdout line from the worker, or None on timeout."""
        assert self._worker is not None and self._worker.stdout is not None
        bucket: list[str | None] = []

        def _read() -> None:
            try:
                bucket.append(self._worker.stdout.readline() if self._worker and self._worker.stdout else None)
            except Exception:
                bucket.append(None)

        thread = threading.Thread(target=_read, daemon=True)
        thread.start()
        thread.join(timeout=timeout)
        if thread.is_alive():
            return None
        if not bucket:
            return None
        line = bucket[0]
        return line if line else None

    def _kill_worker_hard(self) -> None:
        worker = self._worker
        self._worker = None
        if worker is None:
            return
        try:
            worker.kill()
            worker.wait(timeout=5)
        except (OSError, subprocess.TimeoutExpired):
            pass

    def _execute_subprocess(self, code: str, timeout: int) -> tuple[bool, str]:
        wrapped = f"""
import torch
import triton
import triton.language as tl
import sys

torch.manual_seed(42)
if not torch.cuda.is_available():
    raise RuntimeError("CUDA required")

try:
{self._indent(code, 4)}
    print("{PASS_MARKER}")
except Exception as e:
    print(f"TRITONBENCH_FAIL: {{type(e).__name__}}: {{e}}")
    sys.exit(1)
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(wrapped)
            tmpfile = f.name
        try:
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(self.gpu_index)
            env["TRITON_PRINT_AUTOTUNING"] = "0"
            proc = subprocess.run(
                [PY_INTERPRETER, tmpfile],
                capture_output=True,
                text=True,
                timeout=timeout,
                env=env,
                cwd=str(REPO_ROOT),
            )
            output = proc.stdout + proc.stderr
            return PASS_MARKER in proc.stdout, output
        except subprocess.TimeoutExpired:
            return False, "TIMEOUT"
        finally:
            os.unlink(tmpfile)

    def execute(self, code: str, timeout: int = 120) -> tuple[bool, str]:
        require_blackwell_gpu(self.gpu_index)
        if self._worker is not None:
            try:
                return self._execute_via_worker(code, timeout)
            except Exception as exc:
                self.stop_worker()
                ok, output = self._execute_subprocess(code, timeout)
                return ok, f"{output}\n(worker fallback after: {exc})"
        return self._execute_subprocess(code, timeout)

    @staticmethod
    def _indent(code: str, spaces: int) -> str:
        prefix = " " * spaces
        return "\n".join(prefix + line for line in code.split("\n"))
