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
import os
import sys
from contextlib import contextmanager
from pathlib import Path

from eval.benchmarks import BENCHMARKS, assert_fraction_scores
from eval.canonical_dataset import canonical_hf_url, canonical_sft_sha256
from eval.dataset_verify import _sha256_file
from eval.harness import run_harness
from eval.mix_registry import REGISTRY_PATH, verify_mix_manifest
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


def check_canonical_dataset_claim(manifest: dict, *, bundle_dir: Path | None = None) -> list[str]:
    """Training-track bundles must cite the pinned canonical mining dataset."""
    issues: list[str] = []
    dataset_url = manifest.get("dataset_url")
    try:
        expected_url = canonical_hf_url().rstrip("/")
    except (FileNotFoundError, ValueError):
        return issues

    if not dataset_url:
        issues.append(
            f"training bundle must set dataset_url to the canonical mining dataset ({expected_url})"
        )
        return issues

    if str(dataset_url).rstrip("/") != expected_url:
        issues.append(
            f"dataset_url must be canonical {expected_url}, got {dataset_url!r}; "
            "training-track submissions may not use private or synthetic datasets"
        )

    if bundle_dir is not None:
        mix_path = bundle_dir / "mix_manifest.json"
        if mix_path.exists():
            mix_data = json.loads(mix_path.read_text(encoding="utf-8"))
            remote_sft_sha = mix_data.get("sft_sha256")
            try:
                expected_sft_sha = canonical_sft_sha256()
            except ValueError:
                return issues
            if remote_sft_sha != expected_sft_sha:
                issues.append(
                    "bundle mix_manifest.sft_sha256 does not match datasets/canonical.json pin"
                )
    return issues


def check_mix_provenance(
    bundle_dir: Path,
    manifest: dict,
    *,
    registry_path: Path = REGISTRY_PATH,
) -> list[str]:
    """Validate a cross-miner mix copied into the proof bundle."""
    mix_path = bundle_dir / "mix_manifest.json"
    if not mix_path.exists():
        if manifest.get("mix_manifest_sha256"):
            return ["bundle manifest references mix_manifest_sha256 but mix_manifest.json is missing"]
        return []

    expected_sha = manifest.get("mix_manifest_sha256")
    if expected_sha and _sha256_file(mix_path) != expected_sha:
        return ["mix_manifest.json sha256 does not match bundle manifest"]

    report = verify_mix_manifest(mix_path, registry_path=registry_path)
    return list(report.get("issues") or [])


def check_claim(claimed: dict[str, float], rerun: dict[str, float], tolerance_pct: float = 2.0) -> list[str]:
    """Return the benchmark keys where the claimed score diverges from the cheap
    re-run by more than the tolerance (percentage points, absolute).

    A benchmark's `claim_tolerance_pct` overrides the global `tolerance_pct`
    (e.g. triton's tiny problem set drifts more across serving instances than
    sample-based benchmarks do). The `triton` re-run is level-1-only, so it is
    compared against the claim's `triton_quick` (the same problem subset) when
    present — a full-run composite covers harder levels and would mismatch an
    honest claim systematically.

    Both score maps must be fractions in [0, 1]; the `* 100.0` below turns the gap into
    percentage points, so a 0-100 percentage would make the tolerance 100x too tight.
    """
    assert_fraction_scores(claimed, "claimed (eval_scores.json)")
    assert_fraction_scores(rerun, "re-run (harness)")
    mismatches = []
    for key, rerun_value in rerun.items():
        claimed_value = claimed.get(key)
        if key == "triton" and "triton_quick" in claimed:
            claimed_value = claimed["triton_quick"]
        if claimed_value is None:
            continue
        benchmark = BENCHMARKS.get(key)
        tolerance = tolerance_pct if benchmark is None or benchmark.claim_tolerance_pct is None else benchmark.claim_tolerance_pct
        if abs(claimed_value - rerun_value) * 100.0 > tolerance:
            mismatches.append(key)
    return mismatches


@contextmanager
def _no_student_endpoint_env():
    """Force the re-run to serve the bundle's own checkpoint.

    SPARKDISTILL_STUDENT_ENDPOINT is a miner convenience; during verification a
    stale value would silently score whatever model that endpoint serves instead
    of the checkpoint under verification.
    """
    saved = os.environ.pop("SPARKDISTILL_STUDENT_ENDPOINT", None)
    try:
        yield
    finally:
        if saved is not None:
            os.environ["SPARKDISTILL_STUDENT_ENDPOINT"] = saved


def check_claim_binding(bundle_dir: Path, attestation: dict | None) -> bool | None:
    """Whether the attestation's `eat_nonce` commits to this exact bundle.

    Returns True when the NRAS-signed nonce equals the bundle's `claim_sha256`
    (see `proof.bundle`), False when an attestation is present but unbound
    (legacy random-nonce attestations), and None when there is no attestation.

    The nonce lives in the per-device submodule tokens (where NRAS also asserts
    `x-nvidia-gpu-attestation-report-nonce-match`), not necessarily the overall
    JWT — observed live on NRAS v3.
    """
    if attestation is None:
        return None
    from proof.bundle import claim_sha256

    claims = attestation.get("claims") or {}
    nonces = [claims.get("eat_nonce")]
    nonces += [device.get("eat_nonce") for device in (claims.get("devices") or {}).values()]
    expected = claim_sha256(bundle_dir)
    return any(str(nonce).lower().removeprefix("0x") == expected for nonce in nonces if nonce)


def check_tdx_binding(bundle_dir: Path, attestation: dict | None) -> bool | None:
    """Whether the attestation's Intel TDX quote commits to this exact bundle.

    The TDX quote's 64-byte REPORTDATA must be the bundle's `claim_sha256`
    (zero-padded) — the measured-VM counterpart of `check_claim_binding`'s GPU
    nonce. Returns None when no TDX quote was captured (non-TDX host or
    unprovisioned configfs-tsm node); GPU binding remains the minimum bar.

    Honest scope: this checks the binding, not the quote's Intel signature
    chain — full DCAP/Trust Authority verification is the validator follow-up.
    """
    if attestation is None or not attestation.get("tdx"):
        return None
    from eval.attestation import tdx_report_data
    from proof.bundle import claim_sha256

    expected = tdx_report_data(claim_sha256(bundle_dir)).hex()
    return str(attestation["tdx"].get("report_data") or "").lower() == expected


def check_gpu_signature(attestation: dict | None) -> dict | None:
    """Verify the attestation's NRAS-signed GPU tokens against NVIDIA's JWKS.

    The GPU counterpart of `check_tdx_signature`: without it, the committed
    attestation JSON's `passed` flag and claims are taken on the miner's word.
    Returns None when there is no attestation.
    """
    if attestation is None or not attestation.get("token"):
        return None
    from eval.attestation import verify_gpu_token

    return verify_gpu_token(attestation["token"])


def check_tdx_signature(attestation: dict | None, pccs_url: str | None = None) -> dict | None:
    """DCAP-verify the attestation's TDX quote against Intel PCS.

    Complements `check_tdx_binding` (which proves the quote commits to this
    bundle): this proves the quote itself is genuine — ECDSA signature, PCK
    chain to Intel's root CA, QE identity, and TCB status. Without it, a
    fabricated `tdx` blob with a matching report_data would pass binding.
    Returns None when no TDX quote is present.
    """
    if attestation is None or not attestation.get("tdx"):
        return None
    from eval.attestation import verify_tdx_quote

    return verify_tdx_quote(attestation["tdx"].get("quote_b64") or "", pccs_url)


def check_checkpoint_manifest(manifest: dict, checkpoint_path: Path) -> bool | None:
    """Compare a local checkpoint against the bundle's per-file sha256 manifest.

    Returns None when the bundle predates checkpoint manifests. A mismatch on a
    locally reproduced checkpoint is informational (bit-identical retrains are
    not guaranteed across driver/stack revisions) — the score re-run stays the
    decisive check.
    """
    expected = manifest.get("checkpoint_manifest")
    if not expected:
        return None
    from proof.bundle import checkpoint_manifest

    return checkpoint_manifest(checkpoint_path) == expected


def verify_submission(
    bundle_dir: Path,
    frontier: dict[str, float] | None,
    limit: int = 20,
    tolerance_pct: float = 2.0,
    attestation: dict | None = None,
    *,
    registry_path: Path = REGISTRY_PATH,
    checkpoint: Path | None = None,
) -> dict:
    """Verify a proof bundle; `frontier=None` is the BASELINE case.

    When no frontier exists yet (first verified run on a student/phase, per
    `.gittensor/weights.json`), every proof and claim check still runs, but
    instead of tier scoring the submission is labeled `eval:BASELINE` — its
    scores then seed `runs/frontier.json` for the next submission to beat.
    """
    manifest = json.loads((bundle_dir / "manifest.json").read_text())
    claimed = json.loads((bundle_dir / "eval_scores.json").read_text())["scores"]

    # Proof-only bundles carry no weights: the validator reproduces the checkpoint
    # locally (recipe + dataset, see docs/miner-guide.md) and passes it here.
    checkpoint_path = bundle_dir / "checkpoint"
    if not checkpoint_path.is_dir():
        if checkpoint is None:
            return {
                "verified": False,
                "reason": "checkpoint_required",
                "issues": [
                    "proof-only bundle: reproduce the checkpoint from the recipe + dataset "
                    "and pass it via --checkpoint"
                ],
                "label": "eval:REJECT",
                "run_id": manifest.get("run_id"),
            }
        checkpoint_path = checkpoint

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

    mix_issues = check_mix_provenance(bundle_dir, manifest, registry_path=registry_path)
    if mix_issues:
        return {
            "verified": False,
            "reason": "mix_provenance_failed",
            "issues": mix_issues,
            "label": "eval:REJECT",
            "run_id": manifest.get("run_id"),
        }

    canonical_issues = check_canonical_dataset_claim(manifest, bundle_dir=bundle_dir)
    if canonical_issues:
        return {
            "verified": False,
            "reason": "canonical_dataset_failed",
            "issues": canonical_issues,
            "label": "eval:REJECT",
            "run_id": manifest.get("run_id"),
        }

    # Re-run only registered benchmarks — claimed score files may carry extra detail
    # keys (e.g. eval.triton_bench's triton_* sub-metrics) that have no harness entry.
    claimed_benchmarks = sorted(key for key in claimed if key in BENCHMARKS)
    with _no_student_endpoint_env():
        rerun = run_harness(str(checkpoint_path), claimed_benchmarks, Path("eval/results/_verify"), limit=limit)
    mismatches = check_claim(claimed, rerun, tolerance_pct)
    if mismatches:
        return {
            "verified": False,
            "reason": "claim_mismatch",
            "mismatches": mismatches,
            "label": "eval:REJECT",
            "run_id": manifest.get("run_id"),
        }

    if frontier:
        report = score(claimed, frontier)
    else:
        report = {
            "label": "eval:BASELINE",
            "best_benchmark": None,
            "best_pct_delta": None,
            "regressions": [],
            "per_benchmark": {key: {"candidate": claimed[key], "frontier": None} for key in claimed},
        }
    report["verified"] = True
    report["reason"] = None
    report["run_id"] = manifest.get("run_id")
    # Informational trust signals: claim_bound distinguishes an attestation that
    # cryptographically commits to this bundle from a legacy unbound one, and
    # checkpoint_hash_match records local-reproduction fidelity.
    report["claim_bound"] = check_claim_binding(bundle_dir, attestation)
    report["gpu_signature"] = check_gpu_signature(attestation)
    report["tdx_bound"] = check_tdx_binding(bundle_dir, attestation)
    report["tdx_signature"] = check_tdx_signature(attestation)
    report["checkpoint_hash_match"] = check_checkpoint_manifest(manifest, checkpoint_path)
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
    parser.add_argument(
        "--frontier",
        type=Path,
        default=Path("runs/frontier.json"),
        help="frontier scores json (default: the canonical runs/frontier.json; "
        "a missing file means no frontier exists yet -> eval:BASELINE)",
    )
    parser.add_argument("--attestation", type=Path, default=None, help="attestation json from eval.attestation")
    parser.add_argument("--limit", type=int, default=20, help="examples per benchmark for the cheap re-run")
    parser.add_argument("--tolerance-pct", type=float, default=2.0)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="locally reproduced checkpoint dir, required for proof-only bundles (no weights on HF)",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    bundle_dir = _resolve_bundle_dir(args.bundle_repo, args.bundle_path)
    frontier = json.loads(args.frontier.read_text())["scores"] if args.frontier.exists() else None
    attestation = json.loads(args.attestation.read_text()) if args.attestation else None

    report = verify_submission(
        bundle_dir,
        frontier,
        limit=args.limit,
        tolerance_pct=args.tolerance_pct,
        attestation=attestation,
        checkpoint=args.checkpoint,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(f"{report['label']} (verified={report['verified']}, reason={report['reason']})", file=sys.stderr)
    return 0 if report["verified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
