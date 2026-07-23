"""Quality filter sharing TritonValidator with the eval harness / SparkProof path."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from tritonbench.core.evaluator import TritonEvaluator
from tritonbench.core.validator import TritonValidator


@dataclass
class FilterResult:
    accepted: bool
    reasons: list[str] = field(default_factory=list)
    syntax_ok: bool = False
    api_ok: bool = False
    exec_pass: bool = False
    scores: dict[str, float] = field(default_factory=dict)
    code: str = ""
    api_issues: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class QualityFilter:
    """Gate teacher/student kernel responses before they enter an SFT mix."""

    def __init__(
        self,
        validator: TritonValidator | None = None,
        evaluator: TritonEvaluator | None = None,
        *,
        require_exec: bool = True,
        require_modern_api: bool = False,
        require_syntax: bool = True,
        min_composite: float = 0.0,
        skip_execute: bool = False,
        problem: dict[str, Any] | None = None,
    ):
        self.validator = validator or TritonValidator()
        self.evaluator = evaluator or TritonEvaluator()
        self.require_exec = require_exec
        self.require_modern_api = require_modern_api
        self.require_syntax = require_syntax
        self.min_composite = float(min_composite)
        self.skip_execute = skip_execute
        self.default_problem = problem or {}

    def check(self, response_or_code: str, problem: dict[str, Any] | None = None) -> FilterResult:
        problem = problem or self.default_problem
        code = self.validator.extract_code(response_or_code)
        syntax_ok = self.validator.check_syntax(code)
        api = self.validator.check_triton_api(code)
        api_ok = bool(api.get("modern", False)) and not api.get("issues")
        # Soft api_ok: modern flag alone (issues may include missing patterns on bugfix stubs)
        api_ok_soft = bool(api.get("modern", False)) or (
            "@triton.jit" in code and "tl.program_id" in code and not api.get("deprecated_used")
        )

        if self.skip_execute:
            exec_pass, exec_output = syntax_ok, "SKIP_EXECUTE"
        elif not syntax_ok:
            exec_pass, exec_output = False, "SyntaxError"
        else:
            exec_pass, exec_output = self.validator.execute(code)

        scores = self.evaluator.score(problem, code, exec_pass, exec_output)
        reasons: list[str] = []
        if self.require_syntax and not syntax_ok:
            reasons.append("syntax_error")
        if self.require_modern_api and not api_ok_soft:
            reasons.append("api_not_modern")
            reasons.extend(f"api:{i}" for i in api.get("issues") or [])
        if self.require_exec and not exec_pass:
            reasons.append("exec_failed")
        if scores.get("composite_score", 0.0) < self.min_composite:
            reasons.append(f"composite_below_{self.min_composite}")

        return FilterResult(
            accepted=not reasons,
            reasons=reasons,
            syntax_ok=syntax_ok,
            api_ok=api_ok_soft,
            exec_pass=exec_pass,
            scores=scores,
            code=code,
            api_issues=list(api.get("issues") or []),
        )

    def filter_batch(
        self,
        records: list[dict[str, Any]],
        *,
        response_key: str = "response",
        problem_key: str = "problem",
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        accepted: list[dict[str, Any]] = []
        rejected: list[dict[str, Any]] = []
        for rec in records:
            problem = rec.get(problem_key) if isinstance(rec.get(problem_key), dict) else self.default_problem
            result = self.check(str(rec.get(response_key, "")), problem=problem)
            enriched = dict(rec)
            enriched["filter"] = result.to_dict()
            if result.accepted:
                accepted.append(enriched)
            else:
                rejected.append(enriched)
        return accepted, rejected
