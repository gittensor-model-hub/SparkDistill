"""GPU/CUDA-aware vLLM serving stack for TritonBench evaluation.

The generic ``pip install vllm`` path often pulls CPU-only PyTorch or a CUDA build
that does not match the host GPU. That forces kernel JIT at engine startup and makes
``sparkproof-triton-generate`` / ``eval.triton_bench --serve`` painfully slow.

Install the pinned stack with ``scripts/install_serve.sh`` (dedicated venv).
"""

from __future__ import annotations

import os
import platform
import shutil
from pathlib import Path

from eval.gpu_architecture import GpuArchitecture, normalize_gpu_architecture

VLLM_VERSION = "0.25.0"
DEFAULT_CUDA_TAG = "129"
SERVE_MAX_MODEL_LEN = 4096


def detect_cuda_wheel_tag() -> str:
    """CUDA wheel tag for vLLM + PyTorch (e.g. ``129`` → cu129)."""
    raw = os.environ.get("SPARKDISTILL_VLLM_CUDA", DEFAULT_CUDA_TAG).strip().lower()
    return raw.removeprefix("cu")


def detect_cpu_arch() -> str:
    machine = platform.machine().lower()
    if machine in {"x86_64", "amd64"}:
        return "x86_64"
    if machine in {"aarch64", "arm64"}:
        return "aarch64"
    return machine


def vllm_manylinux_platform(*, cuda_tag: str, cpu_arch: str) -> str:
    if cpu_arch == "aarch64" and cuda_tag == "130":
        return "manylinux_2_35_aarch64"
    return f"manylinux_2_28_{cpu_arch}"


def vllm_wheel_url(
    *,
    version: str = VLLM_VERSION,
    cuda_tag: str | None = None,
    cpu_arch: str | None = None,
) -> str:
    cuda = cuda_tag or detect_cuda_wheel_tag()
    arch = cpu_arch or detect_cpu_arch()
    platform_tag = vllm_manylinux_platform(cuda_tag=cuda, cpu_arch=arch)
    return (
        f"https://github.com/vllm-project/vllm/releases/download/v{version}/"
        f"vllm-{version}+cu{cuda}-cp38-abi3-{platform_tag}.whl"
    )


def pytorch_extra_index_url(cuda_tag: str | None = None) -> str:
    cuda = cuda_tag or detect_cuda_wheel_tag()
    return f"https://download.pytorch.org/whl/cu{cuda}"


def serve_venv_path() -> Path:
    return Path(os.environ.get("SPARKDISTILL_SERVE_VENV", Path.home() / ".sparkdistill-serve")).expanduser()


def resolve_vllm_executable() -> str:
    """Prefer the pinned serve venv's ``vllm`` binary when present."""
    candidate = serve_venv_path() / "bin" / "vllm"
    if candidate.is_file():
        return str(candidate)
    found = shutil.which("vllm")
    if found:
        return found
    return "vllm"


def serve_path_env() -> dict[str, str]:
    """PATH prefix so vLLM's engine JIT can find ``ninja`` from the serve venv."""
    env = os.environ.copy()
    bindir = serve_venv_path() / "bin"
    if bindir.is_dir():
        env["PATH"] = f"{bindir}{os.pathsep}{env.get('PATH', '')}"
    return env


def _gpu_architecture() -> GpuArchitecture | None:
    override = os.environ.get("SPARKDISTILL_GPU_ARCHITECTURE")
    if override:
        from eval.gpu_architecture import normalize_gpu_architecture

        return normalize_gpu_architecture(override)
    try:
        import subprocess

        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0 or not result.stdout.strip():
        return None
    name = result.stdout.strip().splitlines()[0]
    return normalize_gpu_architecture(name)


def vllm_serve_argv(
    model_path: str,
    *,
    port: int = 8000,
    served_model_name: str | None = None,
    max_model_len: int = SERVE_MAX_MODEL_LEN,
) -> list[str]:
    """Build a deterministic ``vllm serve`` command for eval-sized generations."""
    command = [
        resolve_vllm_executable(),
        "serve",
        model_path,
        "--port",
        str(port),
        "--seed",
        "0",
        "--no-enable-prefix-caching",
        "--dtype",
        "bfloat16",
        "--max-model-len",
        str(max_model_len),
        "--disable-log-requests",
    ]
    if served_model_name:
        command += ["--served-model-name", served_model_name]

    arch = _gpu_architecture()
    # Blackwell (SM120+) on vLLM 0.25: skip cooperative top-k paths that regressed
    # decode on some SM120 builds; Hopper uses the default fast path.
    if arch == "blackwell":
        command += ["--compilation-config", '{"cudagraph_mode": "NONE"}']

    return command
