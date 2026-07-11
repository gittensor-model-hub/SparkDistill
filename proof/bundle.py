"""Assemble a proof-of-training bundle: checkpoint + eval scores, ready to publish
to Hugging Face as the artifact a PR's proof link points to.

Per the project's proof-bundle scope, the bundle holds only the checkpoint and its
eval scores — no training-provenance metadata and no attestation report inside the
bundle itself. (An attestation result, if collected, belongs in the PR's ledger entry
via `eval.ledger`, not in the published bundle.)

    python -m proof.bundle --checkpoint outputs/qwen3.5-4b-phase1 --scores eval/results/candidate.json --run-id 2026-07-09-qwen3.5-4b-001 --out proof/_bundles/2026-07-09-qwen3.5-4b-001
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path


@dataclass(frozen=True)
class ProofBundle:
    run_id: str
    bundle_dir: Path
    base_model: str
    created_at: str


def build_bundle(
    checkpoint_dir: Path,
    scores_path: Path,
    out_dir: Path,
    run_id: str,
    base_model: str,
    train_hours: float | None = None,
    train_gpu: str | None = None,
    dataset_url: str | None = None,
) -> ProofBundle:
    """Copy `checkpoint_dir`'s files and `scores_path`'s scores into `out_dir`.

    `out_dir` is created fresh; call with an out_dir that doesn't already hold an
    unrelated bundle. `train_hours`/`train_gpu`/`dataset_url` are the training-track
    claims (see docs/miner-guide.md); `eval.verify` enforces the wall-clock budget
    and GPU requirement against them.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copytree(checkpoint_dir, out_dir / "checkpoint", dirs_exist_ok=True)

    scores = json.loads(scores_path.read_text())
    (out_dir / "eval_scores.json").write_text(json.dumps(scores, indent=2))

    created_at = datetime.now(UTC).isoformat()
    manifest: dict = {"run_id": run_id, "base_model": base_model, "created_at": created_at}
    if train_hours is not None:
        manifest["train_hours"] = train_hours
    if train_gpu is not None:
        manifest["train_gpu"] = train_gpu
    if dataset_url is not None:
        manifest["dataset_url"] = dataset_url
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    return ProofBundle(run_id=run_id, bundle_dir=out_dir, base_model=base_model, created_at=created_at)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--scores", type=Path, required=True, help="scores json from eval.harness")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--base-model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--train-hours", type=float, default=None, help="claimed wall-clock training time (budget: 5h)")
    parser.add_argument("--train-gpu", default=None, help="claimed training GPU, e.g. 'NVIDIA RTX PRO 6000 Blackwell'")
    parser.add_argument("--dataset-url", default=None, help="HF datasets URL the checkpoint was trained on")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    bundle = build_bundle(
        args.checkpoint,
        args.scores,
        args.out,
        args.run_id,
        args.base_model,
        train_hours=args.train_hours,
        train_gpu=args.train_gpu,
        dataset_url=args.dataset_url,
    )
    print(f"wrote proof bundle {bundle.run_id} to {bundle.bundle_dir}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
