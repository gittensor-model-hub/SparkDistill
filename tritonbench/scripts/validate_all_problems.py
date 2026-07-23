#!/usr/bin/env python3
"""Validate every YAML problem in the TritonBench corpus."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tritonbench.converters.validate_problem import validate_corpus  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--problems-root", type=Path, default=ROOT / "tritonbench" / "problems")
    p.add_argument("--json", action="store_true")
    args = p.parse_args(argv)
    report = validate_corpus(args.problems_root)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"ok={report['ok']} failed={report['failed']} total={report['total_files']}")
        print(report["by_level_dir"])
        for path, errs in list(report["errors"].items())[:30]:
            print(f"FAIL {path}: {errs}")
    return 1 if report["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
