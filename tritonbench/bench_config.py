"""Shared paths and runtime config for TritonBench (Triton 3.7.x, Blackwell-only)."""

from __future__ import annotations

import ast
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
TRITON_VERSION = "3.7.1"
TORCH_MIN_VERSION = "2.6.0"
TARGET_GPU_FAMILY = "blackwell"

# Blackwell spans two CUDA major families (not binary-compatible):
#   SM10x — datacenter (B200, B300, GB200): TMEM, tcgen05
#   SM12x — workstation/consumer (RTX 50, RTX PRO 6000 Blackwell, DGX Spark)
BLACKWELL_SM_MAJOR = frozenset({10, 12})

BLACKWELL_PROFILES: dict[str, dict[str, float | str]] = {
    "datacenter": {
        "label": "SM10x datacenter Blackwell (B200/B300)",
        "peak_gbps": 8000.0,  # ~8 TB/s HBM3e (B200 class)
        "peak_tflops": 2250.0,  # dense tensor FP16 order-of-magnitude; override per SKU
    },
    "workstation": {
        "label": "SM12x workstation Blackwell (RTX 50 / RTX PRO 6000)",
        "peak_gbps": 1800.0,
        "peak_tflops": 838.0,
    },
}

# Override with: export TRITONBENCH_PYTHON=/path/to/python
PY_INTERPRETER = os.environ.get("TRITONBENCH_PYTHON", sys.executable)

DATA_G_DIR = REPO_ROOT / "data" / "TritonBench_G_v1"
DATA_T_DIR = REPO_ROOT / "data" / "TritonBench_T_v1"
STATS_G_PATH = REPO_ROOT / "data" / "TritonBench_G_v1.json"
STATS_T_PATH = REPO_ROOT / "data" / "TritonBench_T_v1.jsonl"
TEST_SEPARATOR = "#" * 146


def repo_path(*parts: str) -> str:
    return str(REPO_ROOT.joinpath(*parts))


def blackwell_profile() -> str:
    name = os.environ.get("TRITONBENCH_BLACKWELL_PROFILE", "workstation").strip().lower()
    if name not in BLACKWELL_PROFILES:
        raise ValueError(
            f"unknown TRITONBENCH_BLACKWELL_PROFILE={name!r}; "
            f"expected one of {sorted(BLACKWELL_PROFILES)}"
        )
    return name


def peak_efficiency_limits() -> tuple[float, float]:
    """Return (peak_GB/s, peak_TFLOPS) for efficiency scoring on Blackwell."""
    profile = BLACKWELL_PROFILES[blackwell_profile()]
    gbps = float(os.environ.get("TRITONBENCH_PEAK_GBPS", profile["peak_gbps"]))
    tflops = float(os.environ.get("TRITONBENCH_PEAK_TFLOPS", profile["peak_tflops"]))
    return gbps, tflops


def is_blackwell_capability(major: int, minor: int) -> bool:
    return major in BLACKWELL_SM_MAJOR


def require_blackwell_gpu(device_index: int = 0) -> dict[str, str | int]:
    """Fail fast unless the selected CUDA device is Blackwell (SM10x or SM12x)."""
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU required — TritonBench in SparkDistill targets Blackwell only"
        )
    if device_index >= torch.cuda.device_count():
        raise RuntimeError(f"CUDA device {device_index} not found")

    name = torch.cuda.get_device_name(device_index)
    major, minor = torch.cuda.get_device_capability(device_index)
    if not is_blackwell_capability(major, minor):
        raise RuntimeError(
            f"GPU {name!r} (sm_{major}{minor}) is not Blackwell. "
            f"This fork only supports SM10x datacenter and SM12x workstation Blackwell. "
            f"Hopper (sm_9x), Ada/Ampere (sm_8x), and older GPUs are rejected."
        )

    profile = blackwell_profile()
    if major == 10 and profile == "workstation":
        raise RuntimeError(
            f"GPU {name!r} is datacenter Blackwell (sm_{major}{minor}) but "
            f"TRITONBENCH_BLACKWELL_PROFILE=workstation — set TRITONBENCH_BLACKWELL_PROFILE=datacenter"
        )
    if major == 12 and profile == "datacenter":
        raise RuntimeError(
            f"GPU {name!r} is workstation Blackwell (sm_{major}{minor}) but "
            f"TRITONBENCH_BLACKWELL_PROFILE=workstation (default) — set TRITONBENCH_BLACKWELL_PROFILE=datacenter"
        )

    return {
        "name": name,
        "capability": f"sm_{major}{minor}",
        "major": major,
        "minor": minor,
        "profile": profile,
        "target": TARGET_GPU_FAMILY,
    }


def parse_gpus(value: str | list[int]) -> list[int]:
    if isinstance(value, list):
        return [int(x) for x in value]
    text = str(value).strip()
    if text.startswith("["):
        return [int(x) for x in ast.literal_eval(text)]
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def load_json_records(path: Path) -> list[dict]:
    """Load TritonBench stats (JSON array; some files use a .jsonl extension)."""
    raw = path.read_text(encoding="utf-8")
    data = json.loads(raw)
    if not isinstance(data, list):
        raise ValueError(f"expected JSON array in {path}")
    return data


def check_runtime_versions(*, enforce_blackwell: bool = False) -> dict[str, str]:
    import triton
    import torch

    out: dict[str, str] = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "triton": triton.__version__,
        "cuda_available": str(torch.cuda.is_available()),
        "target_gpu": TARGET_GPU_FAMILY,
        "blackwell_profile": blackwell_profile(),
    }
    if torch.cuda.is_available():
        if enforce_blackwell:
            gpu = require_blackwell_gpu(0)
        else:
            major, minor = torch.cuda.get_device_capability(0)
            gpu = {
                "name": torch.cuda.get_device_name(0),
                "capability": f"sm_{major}{minor}",
            }
        out["gpu_name"] = str(gpu["name"])
        out["gpu_capability"] = str(gpu["capability"])
    return out
