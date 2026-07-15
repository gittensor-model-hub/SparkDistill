"""Per-GPU-architecture frontier records (Blackwell vs Hopper)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

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
