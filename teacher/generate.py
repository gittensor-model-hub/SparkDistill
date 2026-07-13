"""CLI: prompt set -> teacher trajectories -> jsonl dataset.

    python -m teacher.generate \
        --prompts data/prompts/phase1.jsonl \
        --out data/processed/phase1_trajectories.jsonl \
        --provider anthropic --provider openai
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from teacher.providers import ANTHROPIC_TEACHER_MODEL, OPENAI_TEACHER_MODEL, Trajectory, get_teacher


def _iter_prompts(path: Path, limit: int | None) -> Iterator[dict]:
    yielded = 0
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if limit is not None and yielded >= limit:
                break
            yield json.loads(line)
            yielded += 1


def generate_trajectories(
    prompts_path: Path,
    providers: list[str],
    model: str | None,
    max_tokens: int,
    temperature: float,
    limit: int | None,
    concurrency: int,
    thinking_budget: int | None,
) -> Iterator[Trajectory]:
    teachers = [get_teacher(name, model=model) for name in providers]
    prompts = list(_iter_prompts(prompts_path, limit))

    def _run(teacher, record: dict) -> Trajectory:
        return teacher.generate(
            record["prompt"],
            system=record.get("system"),
            max_tokens=max_tokens,
            temperature=temperature,
            thinking_budget=thinking_budget,
        )

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(_run, teacher, record) for teacher in teachers for record in prompts]
        for future in as_completed(futures):
            yield future.result()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prompts", type=Path, required=True, help="jsonl file of {prompt, system?} records")
    parser.add_argument("--out", type=Path, required=True, help="output jsonl of trajectory records")
    parser.add_argument(
        "--provider",
        dest="providers",
        action="append",
        choices=["anthropic", "openai"],
        default=None,
        help=(
            "teacher provider (repeatable). Supported: "
            f"anthropic/{ANTHROPIC_TEACHER_MODEL}, openai/{OPENAI_TEACHER_MODEL}"
        ),
    )
    parser.add_argument(
        "--model",
        default=None,
        help=f"ignored — anthropic is fixed to {ANTHROPIC_TEACHER_MODEL}, openai to {OPENAI_TEACHER_MODEL}",
    )
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--limit", type=int, default=None, help="only sample the first N prompts")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument(
        "--thinking-budget",
        type=int,
        default=4096,
        help="reasoning token budget for teachers that support extended thinking (0 to disable)",
    )
    args = parser.parse_args(argv)

    providers = args.providers or ["anthropic", "openai"]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    thinking_budget = args.thinking_budget or None

    count = 0
    with args.out.open("w") as out_f:
        for trajectory in generate_trajectories(
            args.prompts,
            providers,
            args.model,
            args.max_tokens,
            args.temperature,
            args.limit,
            args.concurrency,
            thinking_budget,
        ):
            out_f.write(json.dumps(trajectory.to_record()) + "\n")
            count += 1

    print(f"wrote {count} trajectories to {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
