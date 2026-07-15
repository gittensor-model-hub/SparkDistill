"""Score a candidate checkpoint's benchmark results against the frontier.

Mirrors `sparkinfer`'s speedup tiering (XL/L/M/S/XS bands over the frontier)
but applied to quality-benchmark deltas instead of decode speed.

Quality tiers (`eval:XS`–`eval:XL`) come from TritonBench only. The general basket
regression-guards GSM8K and the rest; GSM8K may regress up to 2% when Triton
improves by at least 2% vs frontier. Per-benchmark frontier highs merge on any
verified non-REJECT run (see `eval.frontier.merge_frontier_scores`).

    python -m eval.score --candidate eval/results/candidate.json --frontier eval/results/frontier.json --out eval/results/report.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from eval.benchmarks import BENCHMARKS, assert_fraction_scores
from eval.frontier import (
    is_regression,
    merge_frontier_scores,
    pct_delta,
    regression_floor_pct,
    triton_pct_delta,
)
from eval.gpu_architecture import DEFAULT_GPU_ARCHITECTURE, GpuArchitecture, tier_benchmark_for_arch

# (lower_bound_pct, label) — first match wins, checked highest-to-lowest.
_TIER_BANDS = [
    (18.0, "XL"),
    (10.0, "L"),
    (6.0, "M"),
    (3.5, "S"),
    (2.0, "XS"),
]


def _tier_for(pct: float) -> str:
    for lower_bound, label in _TIER_BANDS:
        if pct >= lower_bound:
            return label
    return "none"


def score(
    candidate: dict[str, float],
    frontier: dict[str, float],
    *,
    gpu_architecture: GpuArchitecture = DEFAULT_GPU_ARCHITECTURE,
) -> dict:
    assert_fraction_scores(candidate, "candidate")
    assert_fraction_scores(frontier, "frontier")
    tier_key = tier_benchmark_for_arch(gpu_architecture)
    per_benchmark: dict[str, dict] = {}
    regressions: list[str] = []
    triton_pct = triton_pct_delta(candidate, frontier)

    for key, benchmark in BENCHMARKS.items():
        if key not in candidate or key not in frontier:
            continue
        cand = float(candidate[key])
        front = float(frontier[key])
        pct = pct_delta(cand, front)
        floor = regression_floor_pct(key, triton_pct=triton_pct)
        per_benchmark[key] = {
            "candidate": cand,
            "frontier": front,
            "pct_delta": pct,
            "regression_floor_pct": floor,
        }
        if is_regression(key, cand, front, triton_pct=triton_pct):
            regressions.append(key)

    if regressions:
        label = "REJECT"
        best_key = None
        best_pct = None
        merged_frontier = dict(frontier)
        frontier_updates: list[str] = []
    elif tier_key not in candidate or tier_key not in frontier:
        label = "REJECT"
        best_key = None
        best_pct = None
        merged_frontier = dict(frontier)
        frontier_updates = []
    else:
        best_key = tier_key
        best_pct = per_benchmark[tier_key]["pct_delta"]
        label = _tier_for(best_pct)
        merged_frontier, frontier_updates = merge_frontier_scores(frontier, candidate)

    return {
        "label": f"eval:{label}",
        "best_benchmark": best_key,
        "best_pct_delta": None if best_key is None else best_pct,
        "regressions": [f"regression-{BENCHMARKS[key].label_slug}" for key in regressions],
        "per_benchmark": per_benchmark,
        "frontier_updates": frontier_updates,
        "frontier_scores": merged_frontier,
        "gsm8k_regression_floor_pct": regression_floor_pct("gsm8k", triton_pct=triton_pct),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--candidate", type=Path, required=True, help="scores json from eval.harness for the candidate")
    parser.add_argument("--frontier", type=Path, required=True, help="scores json from eval.harness for the frontier")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    candidate = json.loads(args.candidate.read_text())["scores"]
    frontier = json.loads(args.frontier.read_text())["scores"]
    report = score(candidate, frontier)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(f"{report['label']} (best: {report['best_benchmark']} {report['best_pct_delta']})", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
