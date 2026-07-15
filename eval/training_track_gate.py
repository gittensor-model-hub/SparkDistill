"""Automated gate for training/evaluation improvement pull requests.

Training-track PRs must compete on recipe/hyperparameter changes against the
single pinned canonical mining dataset. Local generators, private blends, and
non-registry dataset paths are rejected.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from fnmatch import fnmatch
from pathlib import Path
from typing import Any

import yaml

from eval.canonical_dataset import (
    CANONICAL_PATH,
    CANONICAL_TRAINING_DATASET_PATH,
    assert_recipe_uses_canonical_dataset,
    canonical_hf_url,
    canonical_sft_sha256,
    load_canonical,
    sft_sha256_from_canonical_text,
    verify_remote_matches_pin,
)
from eval.verify import check_canonical_dataset_claim

_TRAINING_TRACK_CHECKBOX_RE = re.compile(
    r"^\s*-\s*\[[xX]\]\s+\*{0,2}Training/evaluation improvement\*{0,2}\s*$",
    re.MULTILINE,
)
_DATASET_TRACK_CHECKBOX_RE = re.compile(
    r"^\s*-\s*\[[xX]\]\s+\*{0,2}Dataset track submission\*{0,2}\s*$",
    re.MULTILINE,
)
_CANONICAL_SHA_IN_BODY_RE = re.compile(r"`([0-9a-f]{64})`")
_PROOF_BUNDLE_LINE_RE = re.compile(
    r"^\s*(?:[-*]\s*)?Proof[- ]bundle URL(?:\s*\([^)]*\))?\s*:\s*(.+?)\s*$",
    re.MULTILINE | re.IGNORECASE,
)
_HF_MODEL_REPO_URL_RE = re.compile(
    r"https://huggingface\.co/(?!datasets/)([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)"
)
_PROOF_BUNDLE_PLACEHOLDER_RE = re.compile(
    r"^(?:n/a|na|none|pending(?:\s+after.*)?|tbd|todo|optional|-|\.)$",
    re.IGNORECASE,
)
# attestation.json lives in the PR itself (runs/<run_id>/attestation.json), never on HF —
# the proof bundle only carries claim-bound artifacts, not the attestation that binds them.
_ATTESTATION_PATH_RE = re.compile(r"^runs/[^/]+/attestation\.json$")
_FORBIDDEN_CHANGED_GLOBS = (
    "eval/gen_*.py",
    "scripts/prepare_triton*.sh",
    "scripts/prepare_sft_data.sh",
    "data/processed/*",
)
_ALLOWED_ALWAYS = frozenset(
    {
        "datasets/canonical.json",
    }
)
TRAINING_LABELS = frozenset({"training:valid", "training:REJECT", "training:skipped"})
_LABEL_COLORS = {
    "training:valid": "0e8a16",
    "training:REJECT": "b60205",
}

# eval:* is the reward-tier label eval.verify computes (separate from training:* above,
# which only gates format/provenance) — applied only when verify_remote_proof_bundle_scores
# could actually compute one (i.e. the bundle's claims were CPU-verifiable; see its docstring).
EVAL_LABELS = frozenset(
    {"eval:XL", "eval:L", "eval:M", "eval:S", "eval:XS", "eval:none", "eval:BASELINE", "eval:REJECT"}
)
_EVAL_LABEL_COLORS = {
    "eval:XL": "1d76db",
    "eval:L": "0e8a16",
    "eval:M": "2cbe4e",
    "eval:S": "7bd88f",
    "eval:XS": "c5def5",
    "eval:none": "d4c5f9",
    "eval:BASELINE": "fbca04",
    "eval:REJECT": "b60205",
}


def is_training_track_pr(pr_body: str | None) -> bool:
    body = pr_body or ""
    if _DATASET_TRACK_CHECKBOX_RE.search(body):
        return False
    return bool(_TRAINING_TRACK_CHECKBOX_RE.search(body))


def is_dataset_track_pr(pr_body: str | None) -> bool:
    return bool(_DATASET_TRACK_CHECKBOX_RE.search(pr_body or ""))


def _matches_forbidden(path: str) -> str | None:
    for pattern in _FORBIDDEN_CHANGED_GLOBS:
        if fnmatch(path, pattern):
            return pattern
    return None


def validate_changed_paths(changed_paths: list[str] | None) -> list[str]:
    if changed_paths is None:
        return []
    issues: list[str] = []
    for path in changed_paths:
        if path in _ALLOWED_ALWAYS:
            continue
        if path == "datasets/registry.jsonl":
            issues.append(
                "training-track PRs must not modify datasets/registry.jsonl; use the dataset track"
            )
            continue
        forbidden = _matches_forbidden(path)
        if forbidden:
            issues.append(
                f"training-track PRs may not change {path!r} "
                f"(matches forbidden pattern {forbidden!r}); "
                "contribute new rows through SparkProof + the dataset registry first"
            )
    return issues


def _git_show(ref: str, path: str) -> str | None:
    result = subprocess.run(
        ["git", "show", f"{ref}:{path}"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    return result.stdout


def validate_recipe_paths_in_ref(head_ref: str, recipe_paths: list[str]) -> list[str]:
    issues: list[str] = []
    for recipe_path in recipe_paths:
        if not recipe_path.startswith("recipes/") or not recipe_path.endswith((".yaml", ".yml")):
            continue
        text = _git_show(head_ref, recipe_path)
        if text is None:
            continue
        try:
            recipe = yaml.safe_load(text)
        except yaml.YAMLError as exc:
            issues.append(f"{recipe_path}: invalid YAML: {exc}")
            continue
        if not isinstance(recipe, dict):
            issues.append(f"{recipe_path}: root must be a mapping")
            continue
        issues.extend(f"{recipe_path}: {issue}" for issue in assert_recipe_uses_canonical_dataset(recipe))
    return issues


def _canonical_sft_sha256s_for_pr_window(
    *,
    merge_base_ref: str | None,
    head_ref: str = "HEAD",
) -> set[str]:
    """Canonical pins valid for a training PR while dataset-track merges advance HEAD."""
    shas: set[str] = set()

    def add_from_ref(ref: str) -> None:
        text = _git_show(ref, CANONICAL_PATH.as_posix())
        if not text:
            return
        sha = sft_sha256_from_canonical_text(text)
        if sha:
            shas.add(sha)

    add_from_ref(head_ref)
    if not merge_base_ref:
        return shas

    add_from_ref(merge_base_ref)
    log = subprocess.run(
        [
            "git",
            "log",
            "--format=%H",
            f"{merge_base_ref}..{head_ref}",
            "--",
            CANONICAL_PATH.as_posix(),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if log.returncode == 0:
        for commit in log.stdout.splitlines():
            commit = commit.strip()
            if commit:
                add_from_ref(commit)
    return shas


def validate_pr_body_canonical_pin(
    pr_body: str | None,
    *,
    acceptable_sft_shas: set[str] | None = None,
) -> list[str]:
    if not pr_body:
        return ["training-track PR body must cite the pinned canonical dataset URL and sft_sha256"]
    issues: list[str] = []
    expected_url = canonical_hf_url()
    if expected_url not in pr_body:
        issues.append(f"PR body must cite canonical dataset URL {expected_url}")

    allowed = acceptable_sft_shas
    if allowed is None:
        try:
            allowed = {canonical_sft_sha256()}
        except ValueError as exc:
            issues.append(str(exc))
            return issues

    cited_shas = set(_CANONICAL_SHA_IN_BODY_RE.findall(pr_body))
    if not cited_shas:
        issues.append(
            "PR body must cite the canonical sft_sha256 used for training "
            f"(one of: {', '.join(sorted(allowed))})"
        )
    elif not cited_shas & allowed:
        issues.append(
            "PR body sft_sha256 must match a canonical pin from this PR's merge-base window "
            f"(allowed: {', '.join(sorted(allowed))})"
        )
    return issues


def parse_proof_bundle_hf_repo(pr_body: str | None) -> str | None:
    """Return org/repo for the cited Hugging Face proof bundle model repo, if any."""
    body = pr_body or ""
    field_match = _PROOF_BUNDLE_LINE_RE.search(body)
    if field_match:
        value = field_match.group(1).strip().strip("`")
        if value and not _PROOF_BUNDLE_PLACEHOLDER_RE.match(value):
            repo_match = _HF_MODEL_REPO_URL_RE.search(value)
            if repo_match:
                return repo_match.group(1)
    return None


def validate_pr_body_proof_bundle(pr_body: str | None) -> list[str]:
    """Training-track PRs must publish and cite a Hugging Face proof bundle."""
    if not pr_body:
        return ["training-track PR body must cite a Hugging Face proof-bundle URL"]

    issues: list[str] = []
    field_match = _PROOF_BUNDLE_LINE_RE.search(pr_body)
    if not field_match:
        issues.append(
            "PR body must include a Proof-bundle URL field with a published "
            "https://huggingface.co/<user>/<repo> model repo link"
        )
        return issues

    value = field_match.group(1).strip().strip("`")
    if not value or _PROOF_BUNDLE_PLACEHOLDER_RE.match(value):
        issues.append(
            "Proof-bundle URL must be a published Hugging Face model repo URL "
            "(not pending, n/a, or empty)"
        )
        return issues

    if not parse_proof_bundle_hf_repo(pr_body):
        issues.append(
            "Proof-bundle URL must be a Hugging Face model repo "
            "(https://huggingface.co/<user>/<repo>), not a datasets URL"
        )
    return issues


def verify_remote_proof_bundle(
    repo_id: str,
    *,
    hf_token: str | None = None,
    acceptable_sft_shas: set[str] | None = None,
) -> list[str]:
    """Download the cited bundle manifest and verify canonical dataset claims."""
    from huggingface_hub import hf_hub_download

    issues: list[str] = []
    try:
        manifest_path = hf_hub_download(
            repo_id=repo_id,
            repo_type="model",
            filename="manifest.json",
            token=hf_token,
        )
        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    except Exception as exc:
        return [f"failed to download proof bundle manifest from {repo_id}: {exc}"]

    if not isinstance(manifest, dict):
        return [f"{repo_id}/manifest.json must be a JSON object"]

    issues.extend(check_canonical_dataset_claim(manifest))

    allowed = acceptable_sft_shas
    if allowed is None:
        try:
            allowed = {canonical_sft_sha256()}
        except ValueError:
            allowed = set()

    try:
        mix_manifest_path = hf_hub_download(
            repo_id=repo_id,
            repo_type="model",
            filename="mix_manifest.json",
            token=hf_token,
        )
        mix_data = json.loads(Path(mix_manifest_path).read_text(encoding="utf-8"))
        remote_sft_sha = mix_data.get("sft_sha256")
        if remote_sft_sha not in allowed:
            issues.append(
                "proof bundle mix_manifest.sft_sha256 does not match an accepted canonical pin "
                f"for this PR window (allowed {len(allowed)} pin(s))"
            )
    except Exception:
        pass

    try:
        hf_hub_download(
            repo_id=repo_id,
            repo_type="model",
            filename="eval_scores.json",
            token=hf_token,
        )
    except Exception as exc:
        issues.append(f"proof bundle missing eval_scores.json on {repo_id}: {exc}")

    return issues


def find_attestation_path(changed_paths: list[str] | None) -> str | None:
    for path in changed_paths or []:
        if _ATTESTATION_PATH_RE.match(path):
            return path
    return None


def verify_remote_proof_bundle_scores(
    repo_id: str,
    *,
    head_ref: str,
    changed_paths: list[str] | None,
    hf_token: str | None = None,
) -> tuple[list[str], str | None]:
    """Re-run eval.verify's no-GPU attested checks against the cited proof bundle.

    This is the exact CPU-only check a validator otherwise has to run by hand:
    claim_sha256/TDX binding, GPU-attestation JWKS signature, and recomputing any
    attested gsm8k/triton samples from their own bundled responses. It catches
    claim/attested-sample mismatches and GPU-corroboration issues (e.g. an H200
    run whose attestation reports hwmodel=GH100, the same die as H100) at PR time
    instead of only when a human happens to check manually — that gap is what let
    gittensor-model-hub/SparkDistill#120 sit with a stale gsm8k claim (and no
    eval:* label at all) until review.

    Never needs a GPU: proof bundles are weights-free (no checkpoint on HF), so
    any claimed benchmark that isn't covered by an attested sample always ends in
    verify_submission's "checkpoint_required" reason rather than an actual harness
    re-run — that reason is intentionally not gated here, since it just means part
    of the claim is deferred to off-CI validator verification, not that anything
    checkable here failed.

    Returns `(issues, eval_label)` — `eval_label` is the reward-tier label
    eval.verify computed (e.g. "eval:BASELINE", "eval:XL", "eval:REJECT"), or
    None when nothing could be computed (download failure, or the deferred
    "checkpoint_required" case above).
    """
    from huggingface_hub import snapshot_download

    from eval.frontiers import load_frontier_scores
    from eval.verify import resolve_bundle_gpu_architecture, verify_submission

    attestation = None
    attestation_path = find_attestation_path(changed_paths)
    if attestation_path is not None:
        text = _git_show(head_ref, attestation_path)
        if text:
            try:
                attestation = json.loads(text)
            except json.JSONDecodeError:
                return [f"{attestation_path}: invalid JSON"], None

    try:
        bundle_dir = Path(snapshot_download(repo_id=repo_id, repo_type="model", token=hf_token))
    except Exception as exc:
        return [f"failed to download proof bundle from {repo_id} for score verification: {exc}"], None

    if not (bundle_dir / "manifest.json").exists() or not (bundle_dir / "eval_scores.json").exists():
        return [], None  # already flagged by verify_remote_proof_bundle

    manifest = json.loads((bundle_dir / "manifest.json").read_text(encoding="utf-8"))
    frontier = load_frontier_scores(resolve_bundle_gpu_architecture(manifest))

    report = verify_submission(bundle_dir, frontier, attestation=attestation)
    if report.get("reason") == "checkpoint_required":
        return [], None
    if report.get("verified"):
        return [], report.get("label")
    detail = report.get("issues") or report.get("mismatches") or []
    reason = report.get("reason") or "verification failed"
    issues = [f"eval.verify {reason}: {issue}" for issue in detail] if detail else [f"eval.verify: {reason}"]
    return issues, report.get("label")


def should_enforce_training_gate(
    pr_body: str | None,
    changed_paths: list[str] | None,
) -> bool:
    if pr_body is not None and is_dataset_track_pr(pr_body):
        return False
    if pr_body is not None and is_training_track_pr(pr_body):
        return True
    if not changed_paths:
        return False
    sensitive_prefixes = ("recipes/", "eval/gen_", "scripts/prepare_triton", "scripts/prepare_sft_data")
    return any(path.startswith(sensitive_prefixes) for path in changed_paths)


def gate_training_pr(
    *,
    head_ref: str,
    changed_paths: list[str] | None,
    pr_body: str | None,
    merge_base_ref: str | None = None,
    verify_hf_pin: bool = True,
    verify_proof_bundle: bool = True,
    hf_token: str | None = None,
) -> dict[str, Any]:
    if not should_enforce_training_gate(pr_body, changed_paths):
        return {
            "verified": True,
            "label": "training:skipped",
            "issues": [],
            "canonical": load_canonical(),
            "training_dataset_path": CANONICAL_TRAINING_DATASET_PATH,
        }

    acceptable_sft_shas = _canonical_sft_sha256s_for_pr_window(
        merge_base_ref=merge_base_ref,
        head_ref="HEAD",
    )

    issues: list[str] = []
    if pr_body is not None and not is_training_track_pr(pr_body):
        issues.append("check 'Training/evaluation improvement' in the pull request template")
    issues.extend(validate_changed_paths(changed_paths))

    recipe_paths = sorted({path for path in (changed_paths or []) if path.startswith("recipes/")})
    issues.extend(validate_recipe_paths_in_ref(head_ref, recipe_paths))

    issues.extend(validate_pr_body_canonical_pin(pr_body, acceptable_sft_shas=acceptable_sft_shas))
    issues.extend(validate_pr_body_proof_bundle(pr_body))

    if verify_hf_pin:
        issues.extend(verify_remote_matches_pin(hf_token=hf_token))

    eval_label: str | None = None
    if verify_proof_bundle:
        repo_id = parse_proof_bundle_hf_repo(pr_body)
        if repo_id is not None:
            bundle_issues = verify_remote_proof_bundle(
                repo_id,
                hf_token=hf_token,
                acceptable_sft_shas=acceptable_sft_shas,
            )
            issues.extend(bundle_issues)
            if not bundle_issues:
                score_issues, eval_label = verify_remote_proof_bundle_scores(
                    repo_id,
                    head_ref=head_ref,
                    changed_paths=changed_paths,
                    hf_token=hf_token,
                )
                issues.extend(score_issues)

    label = "training:valid" if not issues else "training:REJECT"
    return {
        "verified": not issues,
        "label": label,
        "eval_label": eval_label,
        "issues": issues,
        "canonical": load_canonical(),
        "acceptable_sft_shas": sorted(acceptable_sft_shas),
        "training_dataset_path": CANONICAL_TRAINING_DATASET_PATH,
    }


def update_pr_training_label(pr_number: int, label: str) -> list[str]:
    if label not in TRAINING_LABELS:
        return [f"refusing to apply unknown training label {label!r}"]

    subprocess.run(
        ["gh", "label", "create", label, "--color", _LABEL_COLORS[label], "--force"],
        capture_output=True,
        text=True,
        check=False,
    )
    for existing in TRAINING_LABELS:
        if existing != label:
            subprocess.run(
                ["gh", "pr", "edit", str(pr_number), "--remove-label", existing],
                capture_output=True,
                text=True,
                check=False,
            )
    result = subprocess.run(
        ["gh", "pr", "edit", str(pr_number), "--add-label", label],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return [result.stderr.strip() or result.stdout.strip() or "failed to apply label"]
    return []


def update_pr_eval_label(pr_number: int, label: str) -> list[str]:
    """Apply the eval.verify reward-tier label, replacing any prior eval:* label."""
    if label not in EVAL_LABELS:
        return [f"refusing to apply unknown eval label {label!r}"]

    subprocess.run(
        ["gh", "label", "create", label, "--color", _EVAL_LABEL_COLORS[label], "--force"],
        capture_output=True,
        text=True,
        check=False,
    )
    for existing in EVAL_LABELS:
        if existing != label:
            subprocess.run(
                ["gh", "pr", "edit", str(pr_number), "--remove-label", existing],
                capture_output=True,
                text=True,
                check=False,
            )
    result = subprocess.run(
        ["gh", "pr", "edit", str(pr_number), "--add-label", label],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return [result.stderr.strip() or result.stdout.strip() or "failed to apply label"]
    return []


def close_training_pr(pr_number: int, issues: list[str]) -> list[str]:
    body = "\n".join(f"- {issue}" for issue in issues) or "- training-track gate rejected this PR"
    comment = subprocess.run(
        ["gh", "pr", "comment", str(pr_number), "--body", f"## training:REJECT\n\n{body}"],
        capture_output=True,
        text=True,
        check=False,
    )
    if comment.returncode != 0:
        return [comment.stderr.strip() or comment.stdout.strip() or "failed to comment"]
    close = subprocess.run(
        ["gh", "pr", "close", str(pr_number)],
        capture_output=True,
        text=True,
        check=False,
    )
    if close.returncode != 0:
        return [close.stderr.strip() or close.stdout.strip() or "failed to close PR"]
    return []


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--head-ref", default="HEAD")
    parser.add_argument(
        "--merge-base-ref",
        default=None,
        help="git ref for merge-base with the PR base; enables canonical-pin grace window",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--pr-body-file", type=Path, default=None)
    parser.add_argument("--changed-paths-file", type=Path, default=None)
    parser.add_argument("--skip-hf-pin-check", action="store_true")
    parser.add_argument("--skip-proof-bundle-check", action="store_true")
    parser.add_argument("--apply-label", action="store_true")
    parser.add_argument("--close-on-reject", action="store_true")
    parser.add_argument("--pr-number", type=int, default=None)
    args = parser.parse_args(argv)

    pr_body = args.pr_body_file.read_text(encoding="utf-8") if args.pr_body_file else None
    changed_paths = None
    if args.changed_paths_file:
        changed_paths = [
            line.strip()
            for line in args.changed_paths_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    report = gate_training_pr(
        head_ref=args.head_ref,
        changed_paths=changed_paths,
        pr_body=pr_body,
        merge_base_ref=args.merge_base_ref,
        verify_hf_pin=not args.skip_hf_pin_check,
        verify_proof_bundle=not args.skip_proof_bundle_check,
        hf_token=os.environ.get("HF_TOKEN"),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(
        f"{report['label']} verified={report['verified']} issues={len(report.get('issues') or [])}",
        file=sys.stderr,
    )
    for issue in report.get("issues") or []:
        print(f"  - {issue}", file=sys.stderr)

    if args.apply_label:
        if args.pr_number is None:
            print("--apply-label requires --pr-number", file=sys.stderr)
            return 1
        if report["label"] == "training:skipped":
            print("training gate skipped", file=sys.stderr)
        else:
            label_issues = update_pr_training_label(args.pr_number, report["label"])
            if label_issues:
                for issue in label_issues:
                    print(f"  - {issue}", file=sys.stderr)
                return 1
        eval_label = report.get("eval_label")
        if eval_label is not None:
            eval_label_issues = update_pr_eval_label(args.pr_number, eval_label)
            if eval_label_issues:
                for issue in eval_label_issues:
                    print(f"  - {issue}", file=sys.stderr)
                return 1

    if (
        args.close_on_reject
        and report.get("label") == "training:REJECT"
        and args.pr_number is not None
    ):
        close_issues = close_training_pr(args.pr_number, list(report.get("issues") or []))
        if close_issues:
            for issue in close_issues:
                print(f"  - {issue}", file=sys.stderr)
            return 1

    return 0 if report.get("verified") else 1


if __name__ == "__main__":
    raise SystemExit(main())
