"""Per-GPU-architecture frontier records (Blackwell vs Hopper)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from eval.benchmarks import BENCHMARKS
from eval.frontier import merge_frontier_scores
from eval.gpu_architecture import (
    DEFAULT_GPU_ARCHITECTURE,
    GPU_ARCHITECTURES,
    GpuArchitecture,
    normalize_gpu_architecture,
)

FRONTIERS_PATH = Path("runs/frontiers.json")
LEGACY_FRONTIER_PATH = Path("runs/frontier.json")


def _empty_record(arch: GpuArchitecture) -> dict[str, Any]:
    return {
        "gpu_architecture": arch,
        "run_id": None,
        "proof_bundle": None,
        "scores": {},
    }


def candidate_scores_from_report(report: dict[str, Any]) -> dict[str, float]:
    """Extract candidate benchmark scores from an ``eval.verify`` report."""
    per = report.get("per_benchmark")
    if isinstance(per, dict) and per:
        out: dict[str, float] = {}
        for key, row in per.items():
            if isinstance(row, dict) and "candidate" in row:
                try:
                    out[str(key)] = float(row["candidate"])
                except (TypeError, ValueError):
                    continue
        if out:
            return out
    scores = report.get("scores")
    if isinstance(scores, dict):
        out = {}
        for key, value in scores.items():
            try:
                out[str(key)] = float(value)
            except (TypeError, ValueError):
                continue
        return out
    return {}


def write_frontiers(frontiers: dict[str, dict[str, Any]], path: Path = FRONTIERS_PATH) -> None:
    """Persist ``runs/frontiers.json`` (and sibling legacy ``frontier.json``)."""
    payload: dict[str, Any] = {}
    for arch in GPU_ARCHITECTURES:
        record = frontiers.get(arch) or _empty_record(arch)
        payload[arch] = {
            "gpu_architecture": arch,
            "run_id": record.get("run_id"),
            "proof_bundle": record.get("proof_bundle"),
            "scores": record.get("scores") if isinstance(record.get("scores"), dict) else {},
        }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    # Keep legacy single-file Blackwell frontier in sync for older tooling.
    if path.name == "frontiers.json":
        blackwell = payload.get("blackwell") or _empty_record("blackwell")
        path.with_name("frontier.json").write_text(
            json.dumps(
                {
                    "run_id": blackwell.get("run_id"),
                    "proof_bundle": blackwell.get("proof_bundle"),
                    "scores": blackwell.get("scores") or {},
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )


def apply_verified_report_to_frontiers(
    report: dict[str, Any],
    *,
    proof_bundle: str | None,
    path: Path = FRONTIERS_PATH,
) -> list[str]:
    """Raise per-benchmark highs for a verified non-REJECT training run.

    Called at merge time (ledger workflow). Safe to re-run: only updates when a
    candidate score beats the current bucket high (or seeds an empty bucket).
    """
    if not report.get("verified"):
        return []
    label = str(report.get("label") or "")
    if label == "eval:REJECT" or label.endswith(":REJECT"):
        return []

    candidate = candidate_scores_from_report(report)
    if not candidate:
        return []

    arch = resolve_gpu_architecture(report.get("gpu_architecture"))
    frontiers = load_frontiers(path)
    current = dict(frontiers.get(arch) or _empty_record(arch))
    current_scores = current.get("scores") if isinstance(current.get("scores"), dict) else {}
    # Official basket keys via merge_frontier_scores; also raise diagnostic TritonBench
    # breakdown keys (triton_syntax_pass_rate, …) that seed alongside the basket.
    basket = {key: value for key, value in candidate.items() if key in BENCHMARKS}
    extras = {key: value for key, value in candidate.items() if key not in BENCHMARKS}
    merged_scores, updates = merge_frontier_scores(current_scores, basket)
    for key, value in extras.items():
        if key not in merged_scores or value > float(merged_scores[key]):
            merged_scores[key] = value
            updates.append(key)

    seeding = not current.get("run_id") and not current_scores
    if not updates and not seeding:
        return []

    current["gpu_architecture"] = arch
    current["scores"] = merged_scores
    if report.get("run_id") is not None and (updates or seeding):
        current["run_id"] = report.get("run_id")
    if proof_bundle is not None and (updates or seeding or not current.get("proof_bundle")):
        current["proof_bundle"] = proof_bundle
    frontiers = dict(frontiers)
    frontiers[arch] = current
    write_frontiers(frontiers, path=path)
    return updates


def load_frontiers(path: Path = FRONTIERS_PATH) -> dict[str, dict[str, Any]]:
    """Load all architecture frontiers from `runs/frontiers.json`."""
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"{path} must contain a JSON object")
        out: dict[str, dict[str, Any]] = {}
        for arch in GPU_ARCHITECTURES:
            record = data.get(arch)
            if isinstance(record, dict):
                out[arch] = record
            else:
                out[arch] = _empty_record(arch)
        return out

    # Legacy single-file frontier seeds Blackwell only.
    if LEGACY_FRONTIER_PATH.exists():
        legacy = json.loads(LEGACY_FRONTIER_PATH.read_text(encoding="utf-8"))
        scores = legacy.get("scores") if isinstance(legacy.get("scores"), dict) else {}
        return {
            "blackwell": {
                "gpu_architecture": "blackwell",
                "run_id": legacy.get("run_id"),
                "proof_bundle": legacy.get("proof_bundle"),
                "scores": scores,
            },
            "hopper": _empty_record("hopper"),
        }

    return {arch: _empty_record(arch) for arch in GPU_ARCHITECTURES}


def load_frontier_scores(
    gpu_architecture: GpuArchitecture,
    *,
    path: Path = FRONTIERS_PATH,
) -> dict[str, float] | None:
    """Return frontier scores for an architecture, or None when unset (BASELINE)."""
    record = load_frontiers(path).get(gpu_architecture) or _empty_record(gpu_architecture)
    scores = record.get("scores")
    if not isinstance(scores, dict) or not scores:
        return None
    return {key: float(value) for key, value in scores.items()}


def load_frontier_record(
    gpu_architecture: GpuArchitecture,
    *,
    path: Path = FRONTIERS_PATH,
) -> dict[str, Any]:
    return load_frontiers(path).get(gpu_architecture) or _empty_record(gpu_architecture)


def merge_frontier_record(
    frontiers: dict[str, dict[str, Any]],
    gpu_architecture: GpuArchitecture,
    candidate_scores: dict[str, float],
    *,
    run_id: str | None = None,
    proof_bundle: str | None = None,
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    """Merge per-benchmark highs into one architecture bucket."""
    record = dict(frontiers.get(gpu_architecture) or _empty_record(gpu_architecture))
    current_scores = record.get("scores") if isinstance(record.get("scores"), dict) else {}
    merged_scores, updates = merge_frontier_scores(current_scores, candidate_scores)
    record["gpu_architecture"] = gpu_architecture
    record["scores"] = merged_scores
    if run_id is not None:
        record["run_id"] = run_id
    if proof_bundle is not None:
        record["proof_bundle"] = proof_bundle
    frontiers = dict(frontiers)
    frontiers[gpu_architecture] = record
    return frontiers, updates


def resolve_gpu_architecture(value: str | None, *, default: GpuArchitecture = DEFAULT_GPU_ARCHITECTURE) -> GpuArchitecture:
    arch = normalize_gpu_architecture(value)
    if arch is None:
        return default
    return arch
