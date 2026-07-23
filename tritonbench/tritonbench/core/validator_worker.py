"""Long-lived GPU worker for TritonValidator.execute().

Imports torch/triton once, keeps the CUDA context warm, and executes kernel
snippets over a JSON-line stdin/stdout protocol. Parent processes restart this
worker on crash or timeout so a poisoned CUDA context cannot stick forever.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from typing import Any

PASS_MARKER = "TRITONBENCH_PASS"


def wrap_code(code: str) -> str:
    indented = "\n".join("    " + line for line in code.split("\n"))
    return f"""
import torch
import triton
import triton.language as tl
import sys

torch.manual_seed(42)
if not torch.cuda.is_available():
    raise RuntimeError("CUDA required")

try:
{indented}
    print("{PASS_MARKER}")
except Exception as e:
    print(f"TRITONBENCH_FAIL: {{type(e).__name__}}: {{e}}")
    raise
"""


def _clear_cuda() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
    except Exception:
        pass


def run_job(code: str) -> dict[str, Any]:
    """Execute one kernel snippet in-process; return ok/output payload."""
    namespace: dict[str, Any] = {"__name__": "__main__"}
    output_chunks: list[str] = []

    class _Capture:
        def write(self, data: str) -> int:
            output_chunks.append(data)
            return len(data)

        def flush(self) -> None:
            pass

    capture = _Capture()
    old_out, old_err = sys.stdout, sys.stderr
    try:
        sys.stdout = capture  # type: ignore[assignment]
        sys.stderr = capture  # type: ignore[assignment]
        exec(compile(wrap_code(code), "<tritonbench-worker>", "exec"), namespace)
        output = "".join(output_chunks)
        return {"ok": PASS_MARKER in output, "output": output}
    except Exception:
        output = "".join(output_chunks)
        if not output.strip():
            output = f"TRITONBENCH_FAIL: {traceback.format_exc()}"
        return {"ok": False, "output": output}
    finally:
        sys.stdout, sys.stderr = old_out, old_err
        _clear_cuda()


def warm_imports() -> None:
    import torch
    import triton  # noqa: F401
    import triton.language as tl  # noqa: F401

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    # Touch the device so the CUDA context is created before the first job.
    _ = torch.empty(1, device="cuda")
    torch.cuda.synchronize()


def serve() -> int:
    warm_imports()
    # Signal readiness so the parent does not race the first job against imports.
    sys.stdout.write(json.dumps({"ready": True}) + "\n")
    sys.stdout.flush()

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except json.JSONDecodeError as exc:
            sys.stdout.write(json.dumps({"id": None, "ok": False, "output": f"bad request: {exc}"}) + "\n")
            sys.stdout.flush()
            continue

        if request.get("shutdown"):
            return 0

        job_id = request.get("id")
        code = str(request.get("code") or "")
        result = run_job(code)
        payload = {"id": job_id, "ok": bool(result["ok"]), "output": str(result["output"])}
        sys.stdout.write(json.dumps(payload) + "\n")
        sys.stdout.flush()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-index", type=int, default=0)
    args = parser.parse_args(argv)
    # Parent already sets CUDA_VISIBLE_DEVICES; keep this for standalone debugging.
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(args.gpu_index))
    os.environ.setdefault("TRITON_PRINT_AUTOTUNING", "0")
    return serve()


if __name__ == "__main__":
    raise SystemExit(main())
