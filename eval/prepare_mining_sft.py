"""Export the canonical Hugging Face mining dataset to local Axolotl jsonl."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from eval.mining_dataset import DEFAULT_MINING_DATASET_REPO, mining_dataset_repo
from eval.canonical_dataset import (
    CANONICAL_TRAINING_DATASET_PATH,
    canonical_repo_id,
    sha256_matches_canonical_export,
    verify_remote_matches_pin,
)


def export_mining_sft(
    *,
    out_path: Path,
    repo_id: str | None = None,
    hf_token: str | None = None,
    verify_pin: bool = True,
) -> dict[str, Any]:
    """Download HF mining split and write messages-only jsonl for Axolotl."""
    from datasets import load_dataset

    repo = (repo_id or mining_dataset_repo()).strip()
    if not repo:
        raise ValueError("mining dataset repo id is empty")
    if repo != canonical_repo_id():
        raise ValueError(
            f"training exports must use the canonical mining repo {canonical_repo_id()!r}, got {repo!r}"
        )

    if verify_pin:
        pin_issues = verify_remote_matches_pin(repo_id=repo, hf_token=hf_token)
        if pin_issues:
            raise ValueError("; ".join(pin_issues))

    ds = load_dataset(repo, split="train", token=hf_token)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows_written = 0
    with out_path.open("w", encoding="utf-8") as handle:
        for row in ds:
            messages = row.get("messages")
            if not isinstance(messages, list) or not messages:
                raise ValueError(f"{repo} row missing non-empty messages list")
            handle.write(json.dumps({"messages": messages}, ensure_ascii=False) + "\n")
            rows_written += 1

    if rows_written == 0:
        raise ValueError(f"{repo} train split is empty")

    resolved_out = out_path.resolve()
    if verify_pin and resolved_out.as_posix().endswith(CANONICAL_TRAINING_DATASET_PATH):
        sha_issues = sha256_matches_canonical_export(resolved_out)
        if sha_issues:
            raise ValueError("; ".join(sha_issues))

    return {
        "repo_id": repo,
        "rows_written": rows_written,
        "out_path": str(resolved_out),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/processed/sparkproof-mining_sft.jsonl"),
        help="output messages jsonl",
    )
    parser.add_argument(
        "--repo-id",
        default=None,
        help=f"HF datasets repo (default: {DEFAULT_MINING_DATASET_REPO})",
    )
    parser.add_argument(
        "--skip-pin-check",
        action="store_true",
        help="skip verification against datasets/canonical.json (local dev only)",
    )
    args = parser.parse_args(argv)

    import os

    try:
        result = export_mining_sft(
            out_path=args.out,
            repo_id=args.repo_id,
            hf_token=os.environ.get("HF_TOKEN"),
            verify_pin=not args.skip_pin_check,
        )
    except Exception as exc:
        print(f"prepare mining sft failed: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
