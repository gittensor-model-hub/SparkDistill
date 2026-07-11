"""Validator check for a dataset-track submission (SparkProof bundle on Hugging Face).

The dataset track is the first half of the SN74 economy: a miner runs SparkProof on a
Blackwell CC VM, publishes the verified dataset to Hugging Face (rows + `proof/`
artifacts), and opens a text-only PR here linking that HF repo plus the bundle's
`trajectories_sha256`. The validator runs this tool, which checks:

1. The `proof/` artifacts exist in the HF repo (dataset_manifest.json from SparkProof's
   release gate, manifest.json, gpu_attestation.json, trajectories.jsonl, ...).
2. The GPU CC attestation passed — the data really came from an attested Blackwell node.
3. The release gate passed (decontamination + provenance) and `trajectories.jsonl`
   matches the sha256 the release gate recorded, so the rows can't be swapped after
   gating.
4. Optionally re-runs full `sparkproof-verify` policy checks (unmodified-code / pinned
   teacher / merkle) when a SparkProof checkout is available.
5. Sizes the dataset into a reward label from verified row count:

   | label | rows |
   |---|---|
   | `dataset:l` | >= 10000 |
   | `dataset:m` | >= 1000 |
   | `dataset:s` | >= 100 |
   | `dataset:none` | < 100 (mergeable, below reward threshold) |
   | `dataset:REJECT` | any check above failed |

    python -m eval.dataset_verify --hf-repo <user>/sparkproof-triton-v0 \\
        [--claimed-sha256 <trajectories_sha256 from the PR>] \\
        [--sparkproof-root ../SparkProof] --out eval/results/dataset_report.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

# (min_rows, label) — first match wins, checked largest-to-smallest.
_SIZE_BANDS = [
    (10_000, "dataset:l"),
    (1_000, "dataset:m"),
    (100, "dataset:s"),
]

REQUIRED_PROOF_FILES = ("manifest.json", "dataset_manifest.json", "gpu_attestation.json", "trajectories.jsonl")


def size_label(rows: int) -> str:
    for min_rows, label in _SIZE_BANDS:
        if rows >= min_rows:
            return label
    return "dataset:none"


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_proof_dir(proof_dir: Path, claimed_sha256: str | None = None) -> tuple[list[str], int]:
    """Return (issues, verified_row_count) for a bundle's proof directory."""
    issues: list[str] = []
    for name in REQUIRED_PROOF_FILES:
        if not (proof_dir / name).exists():
            issues.append(f"missing proof artifact: {name}")
    if issues:
        return issues, 0

    attestation = json.loads((proof_dir / "gpu_attestation.json").read_text())
    if not attestation.get("passed"):
        issues.append("gpu_attestation.passed is false")

    dataset_manifest = json.loads((proof_dir / "dataset_manifest.json").read_text())
    if not dataset_manifest.get("passed"):
        issues.append("release gate did not pass (dataset_manifest.passed is false)")
    if dataset_manifest.get("blocked_rows"):
        issues.append(f"release gate blocked {dataset_manifest['blocked_rows']} rows")

    actual_sha = _sha256_file(proof_dir / "trajectories.jsonl")
    gated_sha = dataset_manifest.get("trajectories_sha256")
    if gated_sha and actual_sha != gated_sha:
        issues.append("trajectories.jsonl sha256 does not match dataset_manifest — rows changed after release gate")
    if claimed_sha256 and actual_sha != claimed_sha256:
        issues.append("trajectories.jsonl sha256 does not match the hash claimed in the PR")

    rows = int(dataset_manifest.get("rows_total") or 0)
    actual_rows = sum(1 for line in (proof_dir / "trajectories.jsonl").read_text().splitlines() if line.strip())
    if actual_rows != rows:
        issues.append(f"rows_total mismatch: manifest={rows} trajectories.jsonl={actual_rows}")

    return issues, rows


def run_sparkproof_verify(proof_dir: Path, sparkproof_root: Path) -> list[str]:
    """Re-run full SparkProof policy verification (pinned teachers, unmodified request
    hashes, merkle root, GPU profile) via the sibling SparkProof checkout."""
    result = subprocess.run(
        ["uv", "run", "sparkproof-verify", "--bundle", str(proof_dir)],
        cwd=sparkproof_root,
        capture_output=True,
        text=True,
        timeout=600,
    )
    if result.returncode != 0:
        tail = (result.stdout + result.stderr).strip().splitlines()[-5:]
        return [f"sparkproof-verify failed: {' | '.join(tail)}"]
    return []


def verify_dataset_submission(
    proof_dir: Path,
    *,
    claimed_sha256: str | None = None,
    sparkproof_root: Path | None = None,
) -> dict:
    issues, rows = check_proof_dir(proof_dir, claimed_sha256)
    if not issues and sparkproof_root is not None:
        issues.extend(run_sparkproof_verify(proof_dir, sparkproof_root))

    label = "dataset:REJECT" if issues else size_label(rows)
    return {
        "verified": not issues,
        "label": label,
        "rows_total": rows,
        "issues": issues,
    }


def _resolve_proof_dir(hf_repo: str | None, proof_path: Path | None) -> Path:
    if proof_path is not None:
        return proof_path
    if hf_repo is None:
        raise ValueError("one of --hf-repo or --proof-path is required")
    from huggingface_hub import snapshot_download

    snapshot = Path(snapshot_download(repo_id=hf_repo, repo_type="dataset", allow_patterns=["proof/*"]))
    return snapshot / "proof"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hf-repo", default=None, help="HF datasets repo id from the miner's PR")
    parser.add_argument("--proof-path", type=Path, default=None, help="local proof/bundle dir (alternative to --hf-repo)")
    parser.add_argument("--claimed-sha256", default=None, help="trajectories_sha256 claimed in the PR text")
    parser.add_argument(
        "--sparkproof-root",
        type=Path,
        default=None,
        help="path to a SparkProof checkout to re-run full policy verification (recommended)",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    proof_dir = _resolve_proof_dir(args.hf_repo, args.proof_path)
    report = verify_dataset_submission(
        proof_dir, claimed_sha256=args.claimed_sha256, sparkproof_root=args.sparkproof_root
    )
    if args.hf_repo:
        report["hf_repo"] = args.hf_repo

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(f"{report['label']} (rows={report['rows_total']}, verified={report['verified']})", file=sys.stderr)
    return 0 if report["verified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
