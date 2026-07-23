"""Performance scoring: static signals + do_bench parse + optional Nsight hooks."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class PerfResult:
    gbps: float | None = None
    tflops: float | None = None
    latency_ms: float | None = None
    peak_util: float | None = None
    score: float = 0.0
    signals_found: dict[str, bool] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class NsightConfig:
    """Optional Nsight Compute command builder (does not execute)."""

    ncu_bin: str = "ncu"
    set_full: bool = False
    metrics: tuple[str, ...] = (
        "sm__throughput.avg.pct_of_peak_sustained_elapsed",
        "dram__throughput.avg.pct_of_peak_sustained_elapsed",
        "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
    )
    kernel_name_filter: str = ""
    replay_mode: str = "kernel"


_LATENCY_RE = re.compile(
    r"(?:latency|time|elapsed|median|ms)\s*[=:]\s*([0-9.eE+-]+)\s*(?:ms)?",
    re.IGNORECASE,
)
_GBPS_RE = re.compile(r"([0-9.eE+-]+)\s*(?:GB/?s|gbps)", re.IGNORECASE)
_TFLOPS_RE = re.compile(r"([0-9.eE+-]+)\s*(?:TFLOPS|tflops)", re.IGNORECASE)
_DO_BENCH_RE = re.compile(r"do_bench|triton\.testing\.do_bench", re.IGNORECASE)


STATIC_SIGNAL_CHECKS: list[tuple[str, str]] = [
    ("autotune", r"@triton\.autotune"),
    ("block_m", r"BLOCK_M\s*[=:]"),
    ("block_n", r"BLOCK_N\s*[=:]"),
    ("block_k", r"BLOCK_K\s*[=:]"),
    ("block_size", r"BLOCK_SIZE\s*[=:]"),
    ("num_stages", r"num_stages"),
    ("num_warps", r"num_warps"),
    ("num_ctas", r"num_ctas"),
    ("tl_dot", r"tl\.dot\s*\("),
    ("tensor_descriptor", r"tl\.make_tensor_descriptor"),
    ("blackwell_hint", r"(?i)blackwell|sm_120|sm_121|sm120|rtx\s*50"),
    ("coalesced_load", r"tl\.load\([^)]*stride|offs.*stride|arange\(0,\s*BLOCK"),
    ("do_bench", r"triton\.testing\.do_bench|do_bench\s*\("),
    ("fp8", r"tl\.float8e4nv|tl\.float8e5m2|float8"),
    ("pipeline_cache", r"tl\.dot\([^)]*acc|num_stages\s*="),
]


def static_perf_signals(code: str) -> dict[str, bool]:
    """Return named boolean performance-awareness signals found in source."""
    code = code or ""
    found: dict[str, bool] = {}
    for name, pattern in STATIC_SIGNAL_CHECKS:
        found[name] = bool(re.search(pattern, code))
    return found


def parse_bench_output(exec_output: str) -> PerfResult:
    """Parse latency / bandwidth / TFLOPS figures from execution logs."""
    result = PerfResult()
    text = exec_output or ""
    if not text.strip():
        result.notes.append("empty exec_output")
        return result

    m = _GBPS_RE.search(text)
    if m:
        try:
            result.gbps = float(m.group(1))
        except ValueError:
            result.notes.append("failed to parse GB/s")

    m = _TFLOPS_RE.search(text)
    if m:
        try:
            result.tflops = float(m.group(1))
        except ValueError:
            result.notes.append("failed to parse TFLOPS")

    # Prefer explicit latency labels; fall back to first ms-looking number near do_bench.
    for m in _LATENCY_RE.finditer(text):
        try:
            val = float(m.group(1))
        except ValueError:
            continue
        # Heuristic: latency in ms is usually < 1e5
        if 0.0 < val < 1e5:
            result.latency_ms = val
            break

    if _DO_BENCH_RE.search(text):
        result.notes.append("do_bench mention in output/source log")
    return result


def estimate_bytes_moved(numel: int, dtype_bytes: int, reads: int, writes: int) -> int:
    """Rough traffic estimate for bandwidth calculations."""
    numel = max(0, int(numel))
    dtype_bytes = max(1, int(dtype_bytes))
    reads = max(0, int(reads))
    writes = max(0, int(writes))
    return numel * dtype_bytes * (reads + writes)


def gbps(bytes_moved: int, latency_ms: float) -> float:
    if latency_ms <= 0:
        return 0.0
    return (bytes_moved / 1e9) / (latency_ms / 1e3)


def tflops(flops: int, latency_ms: float) -> float:
    if latency_ms <= 0:
        return 0.0
    return (flops / 1e12) / (latency_ms / 1e3)


def _util_from_parsed(parsed: PerfResult, gpu_peak_tflops: float | None) -> float | None:
    if parsed.tflops is not None and gpu_peak_tflops and gpu_peak_tflops > 0:
        return max(0.0, min(1.0, parsed.tflops / gpu_peak_tflops))
    if parsed.gbps is not None:
        # Assume ~1 TB/s class peak for workstation Blackwell as a soft prior.
        return max(0.0, min(1.0, parsed.gbps / 1000.0))
    return None


def score_performance(
    code: str,
    exec_output: str = "",
    *,
    gpu_peak_tflops: float | None = None,
) -> tuple[float, PerfResult]:
    """Combine static perf-awareness signals with parsed bench metrics."""
    signals = static_perf_signals(code)
    parsed = parse_bench_output(exec_output)
    parsed.signals_found = signals

    static_score = sum(1.0 for v in signals.values() if v) / max(1, len(signals))
    util = _util_from_parsed(parsed, gpu_peak_tflops)
    parsed.peak_util = util

    if util is not None:
        # Blend: static structure still matters for distillation pedagogy.
        score = 0.55 * static_score + 0.45 * util
        parsed.notes.append(f"blended with peak_util={util:.3f}")
    else:
        score = static_score
        parsed.notes.append("static-only score (no parsed bench util)")

    # Small bonus if code actually calls do_bench.
    if signals.get("do_bench"):
        score = min(1.0, score + 0.05)

    parsed.score = max(0.0, min(1.0, score))
    return parsed.score, parsed


def format_nsight_command(kernel_script: Path, out_dir: Path, config: NsightConfig | None = None) -> list[str]:
    """Build an `ncu` command line for optional Nsight Compute collection."""
    cfg = config or NsightConfig()
    out_dir = Path(out_dir)
    out_base = out_dir / "nsight_report"
    cmd = [
        cfg.ncu_bin,
        "--target-processes",
        "all",
        "--replay-mode",
        cfg.replay_mode,
        "-o",
        str(out_base),
    ]
    if cfg.set_full:
        cmd.extend(["--set", "full"])
    else:
        for metric in cfg.metrics:
            cmd.extend(["--metrics", metric])
    if cfg.kernel_name_filter:
        cmd.extend(["--kernel-name", cfg.kernel_name_filter])
    cmd.append(str(kernel_script))
    return cmd
