"""CLI to refresh datasets/canonical.json from Hugging Face."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from eval.canonical_dataset import CANONICAL_PATH, write_pin_from_remote


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=CANONICAL_PATH,
        help="path to write datasets/canonical.json",
    )
    parser.add_argument("--repo-id", default=None, help="HF datasets repo (default: canonical default)")
    args = parser.parse_args(argv)

    try:
        payload = write_pin_from_remote(
            repo_id=args.repo_id,
            hf_token=os.environ.get("HF_TOKEN"),
            out_path=args.out,
        )
    except Exception as exc:
        print(f"update canonical pin failed: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
