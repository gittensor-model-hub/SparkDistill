"""Main TritonBench runner."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from tritonbench.core.evaluator import TritonEvaluator
from tritonbench.core.reporter import Reporter
from tritonbench.core.validator import TritonValidator
from tritonbench.features.triton_371 import DEFAULT_GPU_TARGET, TRITON_VERSION, system_prompt_for_triton
from tritonbench.harness.model_interface import ModelInterface

_LEVEL_DIRS = {
    1: "level1_basic",
    2: "level2_intermediate",
    3: "level3_advanced",
    4: "level4_expert",
}


@dataclass
class BenchConfig:
    model_endpoint: str = "http://localhost:8000/v1"
    model_name: str = "qwen-triton"
    api_key: str = "token-abc123"
    levels: list[int] = field(default_factory=lambda: [1])
    include_bugfix: bool = True
    timeout_per_problem: int = 120
    gpu_target: str = DEFAULT_GPU_TARGET
    triton_version: str = TRITON_VERSION
    output_dir: str = "./results"
    max_tokens: int = 4096
    temperature: float = 0.2
    gpu_index: int = 0
    problems_dir: str = ""


class TritonBench:
    def __init__(self, config: BenchConfig):
        self.config = config
        self.model = ModelInterface(
            endpoint=config.model_endpoint,
            model=config.model_name,
            api_key=config.api_key,
        )
        self.validator = TritonValidator(config.triton_version, gpu_index=config.gpu_index)
        self.evaluator = TritonEvaluator(gpu_target=config.gpu_target)
        self.reporter = Reporter(config.output_dir)
        self.problems = self._load_problems()

    def _problems_root(self) -> Path:
        if self.config.problems_dir:
            return Path(self.config.problems_dir)
        return Path(__file__).resolve().parent.parent / "problems"

    def _load_problems(self) -> list[dict]:
        problems: list[dict] = []
        base = self._problems_root()
        for level in self.config.levels:
            level_dir = base / _LEVEL_DIRS.get(level, f"level{level}")
            if not level_dir.exists():
                continue
            for path in sorted(level_dir.glob("*.yaml")):
                with path.open() as fh:
                    prob = yaml.safe_load(fh)
                prob["level"] = prob.get("level", level)
                prob["id"] = prob.get("id", path.stem)
                problems.append(prob)
        if self.config.include_bugfix:
            bugfix_dir = base / "bugfix"
            if bugfix_dir.exists():
                for path in sorted(bugfix_dir.glob("*.yaml")):
                    with path.open() as fh:
                        prob = yaml.safe_load(fh)
                    prob["level"] = "bugfix"
                    prob["id"] = prob.get("id", path.stem)
                    problems.append(prob)
        return problems

    def run(self) -> dict:
        results = []
        total = len(self.problems)
        if total == 0:
            raise RuntimeError(f"no problems found under {self._problems_root()}")

        for i, problem in enumerate(self.problems):
            print(f"\n[{i + 1}/{total}] {problem['id']} (level {problem['level']})")
            result = self._evaluate_problem(problem)
            results.append(result)
            passed = sum(1 for r in results if r["exec_pass"])
            print(f"  composite={result['composite_score']:.2f} | running pass {passed}/{len(results)}")

        return self.reporter.generate(results, self.config)

    def _evaluate_problem(self, problem: dict) -> dict:
        start = time.time()
        prompt = self._build_prompt(problem)
        response = self.model.generate(
            prompt=prompt,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
        )
        code = self.validator.extract_code(response)
        syntax_ok = self.validator.check_syntax(code)
        api_check = self.validator.check_triton_api(code)
        exec_pass, exec_output = self.validator.execute(code, timeout=self.config.timeout_per_problem)
        scores = self.evaluator.score(problem, code, exec_pass, exec_output)
        return {
            "id": problem["id"],
            "level": problem["level"],
            "syntax_ok": syntax_ok,
            "api_modern": api_check["modern"],
            "api_issues": api_check.get("issues", []),
            "exec_pass": exec_pass,
            "exec_output": exec_output[:4000],
            "gen_time_s": time.time() - start,
            **scores,
        }

    def _build_prompt(self, problem: dict) -> dict[str, str]:
        system = system_prompt_for_triton(
            triton_version=self.config.triton_version,
            gpu_target=self.config.gpu_target,
        )
        user = problem["prompt"]
        if problem.get("input_code"):
            user += f"\n\n```python\n{problem['input_code']}\n```"
        if problem.get("constraints"):
            user += f"\n\nConstraints: {problem['constraints']}"
        return {"system": system, "user": user}
