"""CLI: filter raw teacher trajectories into a cleaner SFT-ready set.

`teacher.generate` writes whatever the teacher returned — including empty
responses, refusals, and the occasional malformed row. Training on those
dilutes the distillation signal, so this optional pass drops them *before*
`teacher.format` folds reasoning into `<think>` blocks:

    teacher.generate -> teacher.filter -> teacher.format

    python -m teacher.filter \\
        --in data/processed/phase1_trajectories.jsonl \\
        --out data/processed/phase1_trajectories.filtered.jsonl \\
        --min-response-chars 16 --dedupe-prompts

Filtering is conservative by default: only structurally broken rows (a
missing/empty prompt or response, or a non-object record) and teacher
refusals are dropped. Length bounds and prompt de-duplication are opt-in so
a default run never silently discards a legitimate short answer.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Conservative refusal markers: substrings that, at the very start of a
# teacher response, reliably indicate a non-answer. Kept short and anchored to
# the response head so a legitimate answer that merely discusses refusals or
# safety caveats mid-text is not dropped.
REFUSAL_PREFIXES: tuple[str, ...] = (
    "i cannot help",
    "i can't help",
    "i cannot assist",
    "i can't assist",
    "i cannot provide",
    "i can't provide",
    "i cannot fulfill",
    "i can't fulfill",
    "i'm sorry, but i can't",
    "i'm sorry, but i cannot",
    "i am sorry, but i can't",
    "i am sorry, but i cannot",
    "i'm unable to",
    "i am unable to",
    "sorry, i can't",
    "sorry, i cannot",
    "as an ai language model, i cannot",
    "as an ai, i cannot",
)

# Number of leading response characters scanned for a refusal marker.
_REFUSAL_SCAN_CHARS = 80


@dataclass
class FilterStats:
    """Tally of a filtering run, for logging and provenance."""

    total: int = 0
    kept: int = 0
    dropped_by_reason: Counter[str] = field(default_factory=Counter)

    @property
    def dropped(self) -> int:
        return self.total - self.kept

    def to_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "kept": self.kept,
            "dropped": self.dropped,
            "dropped_by_reason": dict(self.dropped_by_reason),
        }


def _normalized_prompt(record: dict[str, Any]) -> str:
    """Whitespace- and case-normalized prompt, for exact de-duplication."""
    return " ".join(str(record.get("prompt") or "").split()).lower()


def looks_like_refusal(response: str) -> bool:
    """Whether a teacher response opens with a known refusal marker."""
    head = response.strip().lower()[:_REFUSAL_SCAN_CHARS]
    return any(head.startswith(prefix) for prefix in REFUSAL_PREFIXES)


def drop_reason(
    record: Any,
    *,
    min_response_chars: int = 0,
    max_response_chars: int | None = None,
    drop_refusals: bool = True,
) -> str | None:
    """Return why a trajectory record should be dropped, or None to keep it.

    Checks run cheapest-first and the first failure wins, so the returned
    reason is stable and countable. `min_response_chars` / `max_response_chars`
    bound the *stripped* response length; ``0`` / ``None`` disable that bound.
    A record that is not a JSON object (e.g. an unparseable input line surfaced
    as its raw text) is reported as ``"malformed"`` rather than raising.
    """
    if not isinstance(record, dict):
        return "malformed"
    prompt = record.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        return "empty_prompt"
    response = record.get("response")
    if not isinstance(response, str) or not response.strip():
        return "empty_response"

    stripped = response.strip()
    if min_response_chars and len(stripped) < min_response_chars:
        return "too_short"
    if max_response_chars is not None and len(stripped) > max_response_chars:
        return "too_long"
    if drop_refusals and looks_like_refusal(response):
        return "refusal"
    return None


def filter_trajectories(
    records: Iterable[Any],
    *,
    min_response_chars: int = 0,
    max_response_chars: int | None = None,
    drop_refusals: bool = True,
    dedupe_prompts: bool = False,
) -> tuple[list[dict[str, Any]], FilterStats]:
    """Filter an iterable of trajectory records, returning ``(kept, stats)``.

    Order is preserved. When `dedupe_prompts` is set, the first record for a
    given normalized prompt is kept and later duplicates are dropped (counted
    under ``duplicate_prompt``), after all content checks have passed.
    """
    stats = FilterStats()
    kept: list[dict[str, Any]] = []
    seen_prompts: set[str] = set()

    for record in records:
        stats.total += 1
        reason = drop_reason(
            record,
            min_response_chars=min_response_chars,
            max_response_chars=max_response_chars,
            drop_refusals=drop_refusals,
        )
        if reason is not None:
            stats.dropped_by_reason[reason] += 1
            continue
        if dedupe_prompts:
            key = _normalized_prompt(record)
            if key in seen_prompts:
                stats.dropped_by_reason["duplicate_prompt"] += 1
                continue
            seen_prompts.add(key)
        kept.append(record)
        stats.kept += 1

    return kept, stats


def read_trajectories(path: Path) -> Iterator[Any]:
    """Yield parsed records from a jsonl file.

    Blank lines are skipped. A line that is not valid JSON is yielded as its
    raw text so the filter counts it as ``malformed`` instead of crashing the
    whole run on one bad row.
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
        "--in", dest="in_path", type=Path, required=True, help="jsonl of raw trajectories (teacher.generate)"
    )
    parser.add_argument("--out", type=Path, required=True, help="output jsonl of kept trajectory records")
    parser.add_argument(
        "--min-response-chars",
        type=int,
        default=0,
        help="drop responses shorter than this many characters (0 disables)",
    )
    parser.add_argument(
        "--max-response-chars",
        type=int,
        default=None,
        help="drop responses longer than this many characters (unset disables)",
    )
    parser.add_argument("--keep-refusals", action="store_true", help="keep teacher refusals instead of dropping them")
    parser.add_argument(
        "--dedupe-prompts",
        action="store_true",
        help="drop later rows whose normalized prompt already appeared",
    )
    parser.add_argument("--stats-out", type=Path, default=None, help="also write the filter stats json here")
    args = parser.parse_args(argv)

    kept, stats = filter_trajectories(
        read_trajectories(args.in_path),
        min_response_chars=args.min_response_chars,
        max_response_chars=args.max_response_chars,
        drop_refusals=not args.keep_refusals,
        dedupe_prompts=args.dedupe_prompts,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as out_f:
        for record in kept:
            out_f.write(json.dumps(record) + "\n")

    if args.stats_out is not None:
        args.stats_out.parent.mkdir(parents=True, exist_ok=True)
        args.stats_out.write_text(json.dumps(stats.to_dict(), indent=2), encoding="utf-8")

    print(
        f"kept {stats.kept}/{stats.total} trajectories -> {args.out} "
        f"(dropped {stats.dropped}: {dict(stats.dropped_by_reason)})",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
