"""Score generated Triton kernels."""

from __future__ import annotations

from typing import Any

from tritonbench.core.numerical import NumericalResult, score_numerical
from tritonbench.core.performance import PerfResult, score_performance
from tritonbench.core.reference_oracle import score_against_oracle


class TritonEvaluator:
    def __init__(self, gpu_target: str = "Blackwell-SM120", *, gpu_peak_tflops: float | None = None):
        self.gpu = gpu_target
        self.gpu_peak_tflops = gpu_peak_tflops
        self.weights = {
            "correctness": 0.35,
            "api_modernity": 0.20,
            "perf_awareness": 0.20,
            "completeness": 0.15,
            "code_quality": 0.10,
        }

    def score(
        self,
        problem: dict,
        generated_code: str,
        exec_pass: bool,
        exec_output: str,
    ) -> dict[str, float]:
        detailed = self.score_detailed(problem, generated_code, exec_pass, exec_output)
        return {
            "correctness": detailed["correctness"],
            "api_modernity": detailed["api_modernity"],
            "perf_awareness": detailed["perf_awareness"],
            "completeness": detailed["completeness"],
            "code_quality": detailed["code_quality"],
            "composite_score": detailed["composite_score"],
        }

    def score_detailed(
        self,
        problem: dict,
        generated_code: str,
        exec_pass: bool,
        exec_output: str,
    ) -> dict[str, Any]:
        correctness, numerical = score_numerical(exec_pass, exec_output, generated_code)
        perf_score, perf = score_performance(
            generated_code,
            exec_output,
            gpu_peak_tflops=self.gpu_peak_tflops,
        )
        scores: dict[str, Any] = {
            "correctness": correctness,
            "api_modernity": self._score_api_modernity(generated_code),
            "perf_awareness": perf_score,
            "completeness": self._score_completeness(problem, generated_code),
            "code_quality": self._score_quality(generated_code),
            "numerical": numerical.to_dict() if isinstance(numerical, NumericalResult) else numerical,
            "performance": perf.to_dict() if isinstance(perf, PerfResult) else perf,
        }
        oracle = score_against_oracle(problem, generated_code)
        if oracle is not None:
            scores["oracle_similarity"] = oracle
            # Light blend into code_quality without breaking weight sum semantics.
            scores["code_quality"] = min(1.0, 0.7 * scores["code_quality"] + 0.3 * oracle)

        scores["composite_score"] = sum(scores[k] * self.weights[k] for k in self.weights)
        return scores

    def _score_correctness(self, exec_pass: bool, code: str) -> float:
        """Legacy helper retained for compatibility with older call sites/tests."""
        score, _ = score_numerical(exec_pass, "", code)
        return score

    def _score_api_modernity(self, code: str) -> float:
        score = 0.5
        for pattern, delta in {
            "tl.make_tensor_descriptor": 0.15,
            "tl.float8e4nv": 0.10,
            "tl.associative_scan": 0.10,
            "num_stages": 0.05,
            "input_precision": 0.05,
        }.items():
            if pattern in code:
                score += delta
        for pattern, delta in {"tl.make_block_ptr": -0.15, "tl.advance": -0.10}.items():
            if pattern in code:
                score += delta
        return max(0.0, min(1.0, score))

    def _score_performance(self, code: str) -> float:
        score, _ = score_performance(code, "")
        return score

    def _score_completeness(self, problem: dict, code: str) -> float:
        score = 0.0
        for check, weight in [
            ("@triton.jit" in code, 0.25),
            ("def " in code and "grid" in code, 0.25),
            ("torch.allclose" in code or "assert" in code, 0.25),
            ("@triton.autotune" in code, 0.25),
        ]:
            if check:
                score += weight
        required = problem.get("required_patterns") or []
        if required:
            present = sum(1 for pattern in required if pattern in code)
            score = 0.5 * score + 0.5 * (present / len(required))
        return score

    def _score_quality(self, code: str) -> float:
        score = 0.5
        if '"""' in code or "'''" in code or code.count("#") >= 3:
            score += 0.2
        if ": int" in code or "tl.constexpr" in code:
            score += 0.15
        lines = len(code.strip().split("\n"))
        if 20 <= lines <= 200:
            score += 0.15
        return min(1.0, score)
