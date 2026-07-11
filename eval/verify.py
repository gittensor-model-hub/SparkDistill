"""Cheap verification of a submitted proof-of-training bundle.

Instead of a full retrain, a proof bundle is checked by: (1) optionally requiring a
passed GPU CC attestation, (2) re-running each claimed benchmark on a small held-out
sample against the bundle's checkpoint and comparing to the claimed scores within a
tolerance, and only if both pass, (3) scoring the (now-trusted) claimed scores against
the frontier via `eval.score`. A mismatch beyond tolerance is treated as a fabricated
or stale claim and rejected outright — cheap verification does not re-run the full
basket, so it must not silently trust an unverified number either.

    python -m eval.verify --bundle-repo <hf-repo-id> --frontier eval/results/frontier.json \\
        [--attestation runs/<run-id>/attestation.json] --limit 20 --tolerance-pct 2.0 \\
        --out eval/results/report.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from eval.harness import run_harness
from eval.score import score

# Training-track budget (see docs/miner-guide.md): a proof-of-training claim must have
# been produced within this wall-clock budget on the required CC GPU.
MAX_TRAIN_HOURS = 5.0
REQUIRED_TRAIN_GPU = "RTX PRO 6000"


def check_training_claims(
    manifest: dict,
    attestation: dict | None,
    max_train_hours: float = MAX_TRAIN_HOURS,
    required_gpu: str = REQUIRED_TRAIN_GPU,
) -> list[str]:
    """Validate the bundle's training-track claims (train_hours / train_gpu).

    Older bundles without these fields are not failed here — they simply don't
    qualify for the training track and fall back to full retrain-verification.
    """
    issues: list[str] = []
    train_hours = manifest.get("train_hours")
    if train_hours is not None and float(train_hours) > max_train_hours:
        issues.append(f"train_hours {train_hours} exceeds the {max_train_hours}h budget")

    train_gpu = manifest.get("train_gpu")
    if train_gpu is not None and required_gpu.lower() not in str(train_gpu).lower():
        issues.append(f"train_gpu {train_gpu!r} is not a {required_gpu} CC node")

    # When CC attestation is provided, the attested hardware model must corroborate
    # the claimed GPU — the claim alone is just text.
    if attestation and train_gpu is not None:
        claims_blob = json.dumps(attestation.get("claims") or {}).lower()
        if claims_blob != "{}" and "pro 6000" not in claims_blob and "gb20" not in claims_blob:
            issues.append("attestation claims do not corroborate the claimed RTX PRO 6000 GPU")
    return issues


def check_claim(claimed: dict[str, float], rerun: dict[str, float], tolerance_pct: float = 2.0) -> list[str]:
    """Return the benchmark keys where the claimed score diverges from the cheap
    re-run by more than `tolerance_pct` percentage points (absolute)."""
    mismatches = []
    for key, rerun_value in rerun.items():
        claimed_value = claimed.get(key)
        if claimed_value is None:
            continue
        if abs(claimed_value - rerun_value) * 100.0 > tolerance_pct:
            mismatches.append(key)
    return mismatches


def verify_submission(
    bundle_dir: Path,
    frontier: dict[str, float],
    limit: int = 20,
    tolerance_pct: float = 2.0,
    attestation: dict | None = None,
) -> dict:
    manifest = json.loads((bundle_dir / "manifest.json").read_text())
    claimed = json.loads((bundle_dir / "eval_scores.json").read_text())["scores"]
    checkpoint_path = bundle_dir / "checkpoint"

    if attestation is not None and not attestation.get("passed"):
        return {"verified": False, "reason": "attestation_failed", "label": "eval:REJECT", "run_id": manifest.get("run_id")}

    training_issues = check_training_claims(manifest, attestation)
    if training_issues:
        return {
            "verified": False,
            "reason": "training_claims_failed",
            "issues": training_issues,
            "label": "eval:REJECT",
            "run_id": manifest.get("run_id"),
        }

    rerun = run_harness(str(checkpoint_path), sorted(claimed), Path("eval/results/_verify"), limit=limit)
    mismatches = check_claim(claimed, rerun, tolerance_pct)
    if mismatches:
        return {
            "verified": False,
            "reason": "claim_mismatch",
            "mismatches": mismatches,
            "label": "eval:REJECT",
            "run_id": manifest.get("run_id"),
        }

    report = score(claimed, frontier)
    report["verified"] = True
    report["reason"] = None
    report["run_id"] = manifest.get("run_id")
    return report


def _resolve_bundle_dir(bundle_repo: str | None, bundle_path: Path | None) -> Path:
    if bundle_path is not None:
        return bundle_path
    if bundle_repo is None:
        raise ValueError("one of --bundle-repo or --bundle-path is required")
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(repo_id=bundle_repo))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bundle-repo", default=None, help="HF hub repo id to download the proof bundle from")
    parser.add_argument("--bundle-path", type=Path, default=None, help="local bundle dir (alternative to --bundle-repo)")
    parser.add_argument("--frontier", type=Path, required=True, help="scores json from eval.harness for the frontier")
    parser.add_argument("--attestation", type=Path, default=None, help="attestation json from eval.attestation")
    parser.add_argument("--limit", type=int, default=20, help="examples per benchmark for the cheap re-run")
    parser.add_argument("--tolerance-pct", type=float, default=2.0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    bundle_dir = _resolve_bundle_dir(args.bundle_repo, args.bundle_path)
    frontier = json.loads(args.frontier.read_text())["scores"]
    attestation = json.loads(args.attestation.read_text()) if args.attestation else None

    report = verify_submission(bundle_dir, frontier, limit=args.limit, tolerance_pct=args.tolerance_pct, attestation=attestation)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(f"{report['label']} (verified={report['verified']}, reason={report['reason']})", file=sys.stderr)
    return 0 if report["verified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
