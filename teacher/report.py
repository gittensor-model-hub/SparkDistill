"""CLI: summarize a teacher-trajectory jsonl into dataset-quality stats.

`teacher.generate` writes trajectories, but nothing reports on the *set* it
produced. This reads a trajectory jsonl and prints the numbers a miner uses to
decide whether a run is worth training on — above all the **reasoning-capture
rate**, overall and per provider. The Phase-1 recipe notes that not every
teacher exposes a reasoning trace and suggests weighting the `--provider` mix
toward the ones that do; this makes that rate measurable instead of guessed.

    python -m teacher.report --in data/processed/phase1_trajectories.jsonl
    python -m teacher.report --in data/processed/phase1_trajectories.jsonl \\
        --out eval/results/trajectory_report.json

The report is read-only — it never rewrites the dataset (see `teacher.filter`
for the pass that drops rows).
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def _text_len(value: Any) -> int:
    """Stripped character length of a string field, or 0 for anything else."""
    return len(value.strip()) if isinstance(value, str) else 0


def _has_reasoning(record: dict[str, Any]) -> bool:
    reasoning = record.get("reasoning")
    return isinstance(reasoning, str) and bool(reasoning.strip())


def _length_stats(lengths: list[int]) -> dict[str, float] | None:
    """min/mean/median/max over a list of lengths, or None when empty."""
    if not lengths:
        return None
    return {
        "count": len(lengths),
        "min": min(lengths),
        "mean": round(statistics.fmean(lengths), 1),
        "median": int(statistics.median(lengths)),
        "max": max(lengths),
    }


@dataclass
class TrajectoryReport:
    """Aggregate stats over a set of teacher-trajectory records."""

    total: int = 0
    malformed: int = 0
    empty_prompts: int = 0
    empty_responses: int = 0
    with_system: int = 0
    with_reasoning: int = 0
    by_provider: Counter[str] = field(default_factory=Counter)
    by_model: Counter[str] = field(default_factory=Counter)
    # provider -> {"with": <reasoning count>, "total": <record count>}
    reasoning_by_provider: dict[str, dict[str, int]] = field(default_factory=dict)
    response_chars: list[int] = field(default_factory=list, repr=False)
    reasoning_chars: list[int] = field(default_factory=list, repr=False)

    @property
    def valid(self) -> int:
        """Records that parsed into an object (the rate denominators)."""
        return self.total - self.malformed

    @property
    def reasoning_capture_rate(self) -> float:
        return round(self.with_reasoning / self.valid, 4) if self.valid else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "valid": self.valid,
            "malformed": self.malformed,
            "empty_prompts": self.empty_prompts,
            "empty_responses": self.empty_responses,
            "with_system": self.with_system,
            "with_reasoning": self.with_reasoning,
            "reasoning_capture_rate": self.reasoning_capture_rate,
            "by_provider": dict(self.by_provider),
            "by_model": dict(self.by_model),
            "reasoning_capture_by_provider": {
                provider: {
                    "with_reasoning": counts["with"],
                    "total": counts["total"],
                    "rate": round(counts["with"] / counts["total"], 4) if counts["total"] else 0.0,
                }
                for provider, counts in sorted(self.reasoning_by_provider.items())
            },
            "response_chars": _length_stats(self.response_chars),
            "reasoning_chars": _length_stats(self.reasoning_chars),
        }


def summarize(records: Iterable[Any]) -> TrajectoryReport:
    """Aggregate an iterable of trajectory records into a `TrajectoryReport`.

    Records that are not JSON objects (e.g. an unparseable input line surfaced
    as raw text) count toward `total` and `malformed` only, so the rate
    denominators stay meaningful.
    """
    report = TrajectoryReport()
    for record in records:
        report.total += 1
        if not isinstance(record, dict):
            report.malformed += 1
            continue

        if _text_len(record.get("prompt")) == 0:
            report.empty_prompts += 1
        response_len = _text_len(record.get("response"))
        if response_len == 0:
            report.empty_responses += 1
        else:
            report.response_chars.append(response_len)
        if _text_len(record.get("system")) > 0:
            report.with_system += 1

        provider = str(record.get("provider") or "unknown")
        report.by_provider[provider] += 1
        model = record.get("model")
        if isinstance(model, str) and model:
            report.by_model[model] += 1

        slot = report.reasoning_by_provider.setdefault(provider, {"with": 0, "total": 0})
        slot["total"] += 1
        if _has_reasoning(record):
            report.with_reasoning += 1
            slot["with"] += 1
            report.reasoning_chars.append(_text_len(record.get("reasoning")))

    return report


def read_trajectories(path: Path) -> Iterator[Any]:
    """Yield parsed records from a jsonl file.

    Blank lines are skipped; a line that is not valid JSON is yielded as its raw
    text so it is counted (as `malformed`) instead of aborting the whole report.
    """
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                yield line


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--in", dest="in_path", type=Path, required=True, help="jsonl of trajectory records (teacher.generate)"
    )
    parser.add_argument("--out", type=Path, default=None, help="write the report json here (default: stdout only)")
    args = parser.parse_args(argv)

    report = summarize(read_trajectories(args.in_path))
    text = json.dumps(report.to_dict(), indent=2)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)
    print(
        f"{report.valid}/{report.total} valid trajectories, "
        f"reasoning-capture {report.reasoning_capture_rate:.0%}, "
        f"providers={dict(report.by_provider)}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
