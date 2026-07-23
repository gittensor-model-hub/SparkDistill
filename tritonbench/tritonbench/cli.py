"""CLI for TritonBench."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from tritonbench.converters.generate_corpus import generate_corpus
from tritonbench.converters.validate_problem import validate_corpus
from tritonbench.core.reporter import Reporter
from tritonbench.core.runner import BenchConfig, TritonBench
from tritonbench.features.triton_371 import DEFAULT_GPU_TARGET, TRITON_VERSION


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="TritonBench — Triton 3.7.1 kernel programming benchmark")
    sub = parser.add_subparsers(dest="command", required=True)

    eval_p = sub.add_parser("eval", help="Run evaluation against an OpenAI-compatible model")
    eval_p.add_argument("--config", default="configs/eval_quick.yaml")
    eval_p.add_argument("--model", help="Override model name")
    eval_p.add_argument("--endpoint", help="Override API endpoint")
    eval_p.add_argument("--levels", nargs="+", type=int, default=None)
    eval_p.add_argument("--output", default=None, help="Override output directory")
    eval_p.add_argument("--repair", action="store_true", help="Enable multi-turn repair agent mode")
    eval_p.add_argument("--max-repair-turns", type=int, default=3)

    rep_p = sub.add_parser("report", help="Print summary from a results JSON file")
    rep_p.add_argument("--results", required=True)

    gen_p = sub.add_parser("generate-problems", help="Generate YAML corpus from upstream G/T sources")
    gen_p.add_argument("--data-dir", type=Path, default=None)
    gen_p.add_argument("--problems-root", type=Path, default=None)
    gen_p.add_argument("--channels", nargs="+", default=["G", "T"])
    gen_p.add_argument("--limit", type=int, default=None)
    gen_p.add_argument("--dry-run", action="store_true")
    gen_p.add_argument("--no-bugfix", action="store_true")

    val_p = sub.add_parser("validate-problems", help="Validate the YAML problem corpus schema")
    val_p.add_argument("--problems-root", type=Path, default=None)
    val_p.add_argument("--json", action="store_true", help="Print full JSON report")

    args = parser.parse_args(argv)
    root = _repo_root()

    if args.command == "eval":
        cfg_path = Path(args.config)
        if not cfg_path.is_absolute() and not cfg_path.exists():
            alt = root / args.config
            if alt.exists():
                cfg_path = alt
        cfg_dict = yaml.safe_load(cfg_path.read_text()) if cfg_path.exists() else {}
        if args.model:
            cfg_dict["model_name"] = args.model
        if args.endpoint:
            cfg_dict["model_endpoint"] = args.endpoint
        if args.levels is not None:
            cfg_dict["levels"] = args.levels
        if args.output:
            cfg_dict["output_dir"] = args.output
        # Forward optional repair flags via BenchConfig if supported.
        known = {f.name for f in BenchConfig.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        if "enable_repair" in known:
            cfg_dict["enable_repair"] = bool(args.repair)
        if "max_repair_turns" in known:
            cfg_dict["max_repair_turns"] = int(args.max_repair_turns)
        filtered = {k: v for k, v in cfg_dict.items() if k in known}
        config = BenchConfig(**filtered)
        bench = TritonBench(config)
        bench.run()
        return 0

    if args.command == "report":
        data = json.loads(Path(args.results).read_text())
        Reporter(".").print_summary(data)
        return 0

    if args.command == "generate-problems":
        data_dir = args.data_dir or (root / "data")
        problems_root = args.problems_root or (root / "tritonbench" / "problems")
        stats = generate_corpus(
            data_dir=data_dir,
            problems_root=problems_root,
            channels=tuple(args.channels),
            include_bugfix=not args.no_bugfix,
            dry_run=args.dry_run,
            limit=args.limit,
        )
        print(json.dumps(stats, indent=2))
        return 0

    if args.command == "validate-problems":
        problems_root = args.problems_root or (root / "tritonbench" / "problems")
        report = validate_corpus(problems_root)
        if args.json:
            print(json.dumps(report, indent=2))
        else:
            print(
                f"validated {report['ok']}/{report['total_files']} problems under {report['problems_root']}"
            )
            print(f"by_level_dir: {report['by_level_dir']}")
            if report["failed"]:
                print(f"FAILED files: {report['failed']}")
                for path, errs in list(report["errors"].items())[:20]:
                    print(f"  - {path}: {errs[0]}")
        return 1 if report["failed"] else 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
