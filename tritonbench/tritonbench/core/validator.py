"""Validate generated Triton kernel responses."""

from __future__ import annotations

import ast
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bench_config import PY_INTERPRETER, require_blackwell_gpu  # noqa: E402

PASS_MARKER = "TRITONBENCH_PASS"


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

    def execute(self, code: str, timeout: int = 120) -> tuple[bool, str]:
        require_blackwell_gpu(self.gpu_index)
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

    @staticmethod
    def _indent(code: str, spaces: int) -> str:
        prefix = " " * spaces
        return "\n".join(prefix + line for line in code.split("\n"))
