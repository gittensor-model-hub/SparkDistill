import os

import pytest

from eval.serve_stack import (
    DEFAULT_CUDA_TAG,
    VLLM_VERSION,
    pytorch_extra_index_url,
    resolve_vllm_executable,
    serve_venv_path,
    vllm_manylinux_platform,
    vllm_serve_argv,
    vllm_wheel_url,
)


def test_vllm_wheel_url_x86_64_cu129():
    url = vllm_wheel_url(version=VLLM_VERSION, cuda_tag="129", cpu_arch="x86_64")
    assert url.endswith("vllm-0.25.0+cu129-cp38-abi3-manylinux_2_28_x86_64.whl")


def test_vllm_wheel_url_aarch64_cu130():
    url = vllm_wheel_url(version=VLLM_VERSION, cuda_tag="130", cpu_arch="aarch64")
    assert "manylinux_2_35_aarch64" in url
    assert "+cu130-" in url


def test_pytorch_extra_index_defaults_to_cu129():
    assert pytorch_extra_index_url() == f"https://download.pytorch.org/whl/cu{DEFAULT_CUDA_TAG}"


def test_vllm_manylinux_platform_mapping():
    assert vllm_manylinux_platform(cuda_tag="129", cpu_arch="aarch64") == "manylinux_2_28_aarch64"
    assert vllm_manylinux_platform(cuda_tag="130", cpu_arch="aarch64") == "manylinux_2_35_aarch64"


def test_resolve_vllm_executable_prefers_serve_venv(tmp_path, monkeypatch):
    venv = tmp_path / "serve"
    bindir = venv / "bin"
    bindir.mkdir(parents=True)
    vllm_bin = bindir / "vllm"
    vllm_bin.write_text("#!/bin/sh\necho vllm\n", encoding="utf-8")
    vllm_bin.chmod(0o755)
    monkeypatch.setenv("SPARKDISTILL_SERVE_VENV", str(venv))
    monkeypatch.delenv("PATH", raising=False)
    assert resolve_vllm_executable() == str(vllm_bin)


def test_vllm_serve_argv_includes_eval_defaults(monkeypatch):
    monkeypatch.setenv("SPARKDISTILL_GPU_ARCHITECTURE", "hopper-h100")
    argv = vllm_serve_argv("/models/ckpt", port=8001, served_model_name="ckpt")
    assert argv[0] == resolve_vllm_executable()
    assert "--served-model-name" in argv
    assert "ckpt" in argv
    assert "--seed" in argv
    assert "--no-enable-prefix-caching" in argv
    assert "--dtype" in argv
    assert "bfloat16" in argv
    assert "--max-model-len" in argv
    assert "--disable-log-requests" in argv
    assert "--compilation-config" not in argv


def test_vllm_serve_argv_blackwell_disables_cudagraph(monkeypatch):
    monkeypatch.setenv("SPARKDISTILL_GPU_ARCHITECTURE", "blackwell")
    argv = vllm_serve_argv("/models/ckpt")
    assert '--compilation-config' in argv
    assert '{"cudagraph_mode": "NONE"}' in argv
