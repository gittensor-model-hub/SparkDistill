"""Generate teacher trajectories from TritonBench YAML problems for SparkDistill SFT."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Protocol

import yaml

from distill.export_sft import export_sft_records, to_messages_record
from distill.prompt_builder import build_teacher_prompt
from distill.quality_filter import QualityFilter
from tritonbench.core.evaluator import TritonEvaluator
from tritonbench.core.validator import TritonValidator
from tritonbench.features.triton_371 import DEFAULT_GPU_TARGET, TRITON_VERSION

_LEVEL_DIRS = {
    1: "level1_basic",
    2: "level2_intermediate",
    3: "level3_advanced",
    4: "level4_expert",
}


@dataclass
class TeacherResult:
    response: str
    model: str
    reasoning: str | None = None
    raw: dict[str, Any] | None = None


class TeacherClient(Protocol):
    def complete(self, system: str, user: str) -> TeacherResult: ...


class MockTeacherClient:
    """Deterministic offline teacher for tests / dry runs."""

    def __init__(self, model: str = "mock-teacher"):
        self.model = model

    def complete(self, system: str, user: str) -> TeacherResult:
        snippet = user[:180].replace("\n", " ")
        response = f'''```python
import torch
import triton
import triton.language as tl

# mock teacher response for: {snippet}

@triton.jit
def kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x, mask=mask)

def run(x):
    out = torch.empty_like(x)
    n = x.numel()
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK"]),)
    kernel[grid](x, out, n, BLOCK=1024)
    assert torch.allclose(out, x)
    return out
```
'''
        return TeacherResult(response=response, model=self.model, reasoning="mock reasoning trace")


class OpenAICompatibleTeacherClient:
    """Minimal urllib client mirroring tritonbench.harness.model_interface."""

    def __init__(self, endpoint: str, model: str, api_key: str = ""):
        self.endpoint = endpoint.rstrip("/")
        self.model = model
        self.api_key = api_key

    def complete(self, system: str, user: str) -> TeacherResult:
        import urllib.error
        import urllib.request

        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.2,
            "max_tokens": 4096,
        }
        req = urllib.request.Request(
            f"{self.endpoint}/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"teacher API HTTP {e.code}: {detail}") from e
        msg = data["choices"][0]["message"]
        return TeacherResult(
            response=msg.get("content") or "",
            model=self.model,
            reasoning=msg.get("reasoning") or msg.get("reasoning_content"),
            raw=data,
        )


def load_problems(
    problems_root: Path,
    *,
    levels: list[int] | None = None,
    include_bugfix: bool = True,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    levels = levels or [1, 2, 3, 4]
    problems: list[dict[str, Any]] = []
    for level in levels:
        level_dir = problems_root / _LEVEL_DIRS.get(level, f"level{level}")
        if not level_dir.is_dir():
            continue
        for path in sorted(level_dir.glob("*.yaml")):
            with path.open(encoding="utf-8") as fh:
                prob = yaml.safe_load(fh)
            if not isinstance(prob, dict):
                continue
            prob["level"] = prob.get("level", level)
            prob["id"] = prob.get("id", path.stem)
            problems.append(prob)
            if limit is not None and len(problems) >= limit:
                return problems
    if include_bugfix:
        bug_dir = problems_root / "bugfix"
        if bug_dir.is_dir():
            for path in sorted(bug_dir.glob("*.yaml")):
                if limit is not None and len(problems) >= limit:
                    break
                with path.open(encoding="utf-8") as fh:
                    prob = yaml.safe_load(fh)
                if not isinstance(prob, dict):
                    continue
                prob["level"] = "bugfix"
                prob["id"] = prob.get("id", path.stem)
                problems.append(prob)
    return problems


class DataGenerator:
    def __init__(
        self,
        teacher: TeacherClient,
        *,
        quality_filter: QualityFilter | None = None,
        triton_version: str = TRITON_VERSION,
        gpu_target: str = DEFAULT_GPU_TARGET,
        include_few_shot: bool = False,
    ):
        self.teacher = teacher
        self.quality_filter = quality_filter
        self.triton_version = triton_version
        self.gpu_target = gpu_target
        self.include_few_shot = include_few_shot

    def iter_records(self, problems: list[dict[str, Any]]) -> Iterator[dict[str, Any]]:
        for problem in problems:
            prompt = build_teacher_prompt(
                problem,
                triton_version=self.triton_version,
                gpu_target=self.gpu_target,
                include_few_shot=self.include_few_shot,
            )
            result = self.teacher.complete(prompt["system"], prompt["user"])
            if self.quality_filter is not None:
                filt = self.quality_filter.check(result.response, problem=problem)
                if not filt.accepted:
                    continue
            yield to_messages_record(
                problem,
                result.response,
                reasoning=result.reasoning,
                model=result.model,
                system=prompt["system"],
                user=prompt["user"],
            )

    def generate_to_jsonl(
        self,
        problems: list[dict[str, Any]],
        out_path: str | Path,
    ) -> dict[str, Any]:
        records = list(self.iter_records(problems))
        n = export_sft_records(records, out_path, format="messages")
        return {"written": n, "out": str(out_path), "seen_problems": len(problems)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate TritonBench teacher SFT trajectories")
    parser.add_argument("--problems-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--endpoint", default="", help="OpenAI-compatible endpoint; empty => mock teacher")
    parser.add_argument("--model", default="mock-teacher")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--levels", nargs="+", type=int, default=[1, 2, 3, 4])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip-execute", action="store_true")
    parser.add_argument("--require-exec", action="store_true")
    parser.add_argument("--min-composite", type=float, default=0.0)
    parser.add_argument("--no-bugfix", action="store_true")
    parser.add_argument("--few-shot", action="store_true")
    args = parser.parse_args(argv)

    if args.endpoint:
        teacher: TeacherClient = OpenAICompatibleTeacherClient(args.endpoint, args.model, args.api_key)
    else:
        teacher = MockTeacherClient(model=args.model)

    qf = None
    if args.require_exec or args.min_composite > 0 or args.skip_execute:
        qf = QualityFilter(
            TritonValidator(),
            TritonEvaluator(),
            require_exec=args.require_exec,
            min_composite=args.min_composite,
            skip_execute=args.skip_execute or not args.require_exec,
        )

    problems = load_problems(
        args.problems_root,
        levels=args.levels,
        include_bugfix=not args.no_bugfix,
        limit=args.limit,
    )
    gen = DataGenerator(teacher, quality_filter=qf, include_few_shot=args.few_shot)
    stats = gen.generate_to_jsonl(problems, args.out)
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    # Allow `python -m distill.data_generator` from tritonbench root.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    raise SystemExit(main())
