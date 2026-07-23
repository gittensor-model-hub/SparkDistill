#!/usr/bin/env python3
"""Generate the TritonBench YAML problem corpus from upstream G/T sources."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tritonbench.converters.generate_corpus import generate_corpus  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, default=ROOT / "data")
    p.add_argument("--problems-root", type=Path, default=ROOT / "tritonbench" / "problems")
    p.add_argument("--channels", nargs="+", default=["G", "T"])
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--no-bugfix", action="store_true")
    args = p.parse_args(argv)
    stats = generate_corpus(
        data_dir=args.data_dir,
        problems_root=args.problems_root,
        channels=tuple(args.channels),
        include_bugfix=not args.no_bugfix,
        dry_run=args.dry_run,
        limit=args.limit,
    )
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
