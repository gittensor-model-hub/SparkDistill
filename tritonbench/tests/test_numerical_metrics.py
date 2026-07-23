"""Numerical + performance metric unit tests (no GPU)."""

from __future__ import annotations

from pathlib import Path

from tritonbench.core.numerical import (
    classify_failure,
    extract_error_metrics,
    parse_allclose_from_output,
    score_numerical,
)
from tritonbench.core.performance import (
    NsightConfig,
    format_nsight_command,
    gbps,
    parse_bench_output,
    score_performance,
    static_perf_signals,
    tflops,
)


def test_classify_failure_taxonomy():
    assert classify_failure("TIMEOUT") == "timeout"
    assert classify_failure("CUDA out of memory") == "oom"
    assert classify_failure("triton.compiler.CompilationError: boom") == "compile_error"
    assert classify_failure("AssertionError: tensors not close") in {"numerical_mismatch", "assertion"}
    assert classify_failure("RuntimeError: illegal memory access") == "runtime_error"


def test_extract_and_parse_metrics():
    out = "max_abs_err=1.2e-7 max_rel_err=3e-6 n_elements=1024 allclose=True"
    metrics = extract_error_metrics(out)
    assert metrics["max_abs_err"] == 1.2e-7
    parsed = parse_allclose_from_output(out)
    assert parsed is not None
    assert parsed.passed is True
    assert parsed.compared is True


def test_score_numerical_tiers():
    code_close = "import torch\nassert torch.allclose(a, b)\n"
    s, r = score_numerical(True, "TRITONBENCH_PASS\nallclose=True", code_close)
    assert s == 1.0 and r.passed
    s2, _ = score_numerical(True, "TRITONBENCH_PASS", "assert x == y")
    assert 0.8 <= s2 <= 0.9
    s3, _ = score_numerical(True, "TRITONBENCH_PASS", "print('hi')")
    assert abs(s3 - 0.45) < 1e-9
    s4, _ = score_numerical(False, "TIMEOUT", "")
    assert s4 == 0.0
    s5, _ = score_numerical(False, "max_abs_err=0.5 allclose=False", code_close)
    assert s5 == 0.1


def test_static_perf_signals_and_score():
    code = """
@triton.autotune(configs=[], key=["M"])
@triton.jit
def k(...):
    acc = tl.dot(a, b, acc)
    desc = tl.make_tensor_descriptor(...)
    # Blackwell SM120
"""
    signals = static_perf_signals(code)
    assert signals["autotune"]
    assert signals["tl_dot"]
    assert signals["tensor_descriptor"]
    assert signals["blackwell_hint"]
    score, perf = score_performance(code, "elapsed=0.12 ms  512.0 GB/s  1.5 TFLOPS")
    assert 0.0 < score <= 1.0
    assert perf.gbps == 512.0
    assert perf.tflops == 1.5


def test_parse_bench_and_helpers():
    parsed = parse_bench_output("latency=1.25 ms throughput=100 GB/s")
    assert parsed.latency_ms == 1.25 or parsed.gbps == 100.0
    assert gbps(1_000_000_000, 1.0) == 1000.0
    assert tflops(1_000_000_000_000, 1.0) == 1000.0
    cmd = format_nsight_command(Path("k.py"), Path("/tmp/out"), NsightConfig())
    assert cmd[0] == "ncu"
    assert "k.py" in cmd[-1]
