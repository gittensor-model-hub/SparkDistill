"""Multi-turn generate → fail → repair agent for TritonBench bugfix / agentic eval."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

from tritonbench.core.numerical import classify_failure


class SupportsGenerate(Protocol):
    def generate(
        self,
        prompt: dict[str, str],
        max_tokens: int = 4096,
        temperature: float = 0.2,
    ) -> str: ...


REPAIR_SYSTEM_SUFFIX = """
You are repairing a broken Triton 3.7.1 kernel for workstation Blackwell (SM120).
Read the failure carefully. Fix the root cause. Return a complete runnable Python module
with @triton.jit kernel(s), launcher, and torch.allclose test. Prefer tl.make_tensor_descriptor
over deprecated tl.make_block_ptr. Do not omit boundary masks.
""".strip()


@dataclass
class RepairTurn:
    attempt: int
    response: str
    code: str
    syntax_ok: bool
    exec_pass: bool
    exec_output: str
    failure_class: str
    scores: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RepairEpisode:
    problem_id: str
    turns: list[RepairTurn] = field(default_factory=list)
    final_pass: bool = False
    final_scores: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "problem_id": self.problem_id,
            "turns": [t.to_dict() for t in self.turns],
            "final_pass": self.final_pass,
            "final_scores": self.final_scores,
            "n_turns": len(self.turns),
        }


def build_repair_user_prompt(
    problem: dict[str, Any],
    code: str,
    exec_output: str,
    failure_class: str,
) -> str:
    """Construct a repair turn user prompt from the failing attempt."""
    parts = [
        f"Problem id: {problem.get('id', 'unknown')}",
        f"Title: {problem.get('title', '')}",
        "",
        "Original task:",
        (problem.get("prompt") or "").strip(),
    ]
    if problem.get("input_code"):
        parts.extend(["", "Starter / buggy code from the problem:", "```python", problem["input_code"].rstrip(), "```"])
    parts.extend(
        [
            "",
            f"Failure class: {failure_class or 'unknown'}",
            "Previous attempt code:",
            "```python",
            (code or "").rstrip() or "# (empty)",
            "```",
            "",
            "Executor output (truncated):",
            "```",
            (exec_output or "")[:3500],
            "```",
            "",
            "Return a fixed, complete module.",
        ]
    )
    return "\n".join(parts)


class RepairAgent:
    """Iteratively repair kernels using a model + TritonValidator + TritonEvaluator."""

    def __init__(
        self,
        model: SupportsGenerate,
        validator: Any,
        evaluator: Any,
        *,
        max_turns: int = 3,
        skip_execute: bool = False,
        max_tokens: int = 4096,
        temperature: float = 0.2,
        system_base: str = "",
    ):
        if max_turns < 1:
            raise ValueError("max_turns must be >= 1")
        self.model = model
        self.validator = validator
        self.evaluator = evaluator
        self.max_turns = max_turns
        self.skip_execute = skip_execute
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.system_base = system_base or REPAIR_SYSTEM_SUFFIX

    def run(self, problem: dict[str, Any], initial_response: str | None = None) -> RepairEpisode:
        episode = RepairEpisode(problem_id=str(problem.get("id", "unknown")))
        response = initial_response
        code = ""
        exec_output = ""
        failure_class = "unknown"
        exec_pass = False
        scores: dict[str, float] = {}

        for attempt in range(1, self.max_turns + 1):
            if response is None:
                prompt = {
                    "system": self.system_base,
                    "user": self._initial_user(problem)
                    if attempt == 1
                    else build_repair_user_prompt(problem, code, exec_output, failure_class),
                }
                # For attempt > 1 without prior response, always use repair prompt.
                if attempt > 1:
                    prompt["user"] = build_repair_user_prompt(problem, code, exec_output, failure_class)
                response = self.model.generate(
                    prompt,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                )

            code = self.validator.extract_code(response)
            syntax_ok = self.validator.check_syntax(code)
            if self.skip_execute:
                exec_pass, exec_output = syntax_ok, "SKIP_EXECUTE"
            else:
                if not syntax_ok:
                    exec_pass, exec_output = False, "SyntaxError: generated code failed ast.parse"
                else:
                    exec_pass, exec_output = self.validator.execute(code)

            failure_class = "" if exec_pass else classify_failure(exec_output)
            scores = self.evaluator.score(problem, code, exec_pass, exec_output)
            turn = RepairTurn(
                attempt=attempt,
                response=response,
                code=code,
                syntax_ok=syntax_ok,
                exec_pass=exec_pass,
                exec_output=(exec_output or "")[:4000],
                failure_class=failure_class,
                scores=dict(scores),
            )
            episode.turns.append(turn)
            response = None  # force model call next turn unless we break
            if exec_pass:
                break

        episode.final_pass = bool(episode.turns and episode.turns[-1].exec_pass)
        episode.final_scores = dict(episode.turns[-1].scores) if episode.turns else {}
        return episode

    def _initial_user(self, problem: dict[str, Any]) -> str:
        user = (problem.get("prompt") or "").strip()
        if problem.get("input_code"):
            user += f"\n\n```python\n{problem['input_code']}\n```"
        if problem.get("constraints"):
            user += f"\n\nConstraints: {problem['constraints']}"
        return user
