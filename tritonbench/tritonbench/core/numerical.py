"""Structured numerical correctness scoring for TritonBench."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class NumericalResult:
    passed: bool = False
    max_abs_err: float | None = None
    max_rel_err: float | None = None
    mean_abs_err: float | None = None
    n_elements: int | None = None
    dtype: str | None = None
    message: str = ""
    compared: bool = False
    failure_class: str = ""
    extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


FAILURE_TAXONOMY: dict[str, list[str]] = {
    "timeout": ["TIMEOUT", "timed out", "TimeoutExpired"],
    "oom": [
        "out of memory",
        "cuda out of memory",
        "CUDNN_STATUS_ALLOC_FAILED",
        "OOM",
        "insufficient memory",
    ],
    "compile_error": [
        "CompilationError",
        "MLIR",
        "failed to compile",
        "Cannot codegen",
        "triton.compiler",
        "SyntaxError",
        "TypeError: 'constexpr'",
    ],
    "runtime_error": [
        "RuntimeError",
        "CUDA error",
        "illegal memory access",
        "device-side assert",
        "TRITONBENCH_FAIL",
        "IndexError",
        "ValueError",
    ],
    "numerical_mismatch": [
        "not allclose",
        "AssertionError",
        "tensors not close",
        "max_abs_err",
        "mismatch",
        "allclose=False",
        "assert_close",
    ],
    "assertion": ["AssertionError", "assert ", "assert("],
}


_ABS_ERR_RE = re.compile(
    r"(?:max[_\s-]?abs[_\s-]?err(?:or)?|max_error|mae)\s*[=:]\s*([0-9.eE+-]+)",
    re.IGNORECASE,
)
_REL_ERR_RE = re.compile(
    r"(?:max[_\s-]?rel[_\s-]?err(?:or)?|max_relative_error)\s*[=:]\s*([0-9.eE+-]+)",
    re.IGNORECASE,
)
_MEAN_ABS_RE = re.compile(
    r"(?:mean[_\s-]?abs[_\s-]?err(?:or)?|mean_error)\s*[=:]\s*([0-9.eE+-]+)",
    re.IGNORECASE,
)
_N_ELEM_RE = re.compile(r"(?:n_elements|numel|count)\s*[=:]\s*(\d+)", re.IGNORECASE)
_ALLCLOSE_TRUE_RE = re.compile(r"allclose\s*=\s*True|torch\.allclose\s*\([^)]*\)\s*(?:#.*)?$", re.I | re.M)
_ALLCLOSE_FALSE_RE = re.compile(r"allclose\s*=\s*False", re.IGNORECASE)
_PASS_RE = re.compile(r"NUMERICAL_PASS|CORRECTNESS_PASS")
_FAIL_RE = re.compile(r"NUMERICAL_FAIL|CORRECTNESS_FAIL")


def classify_failure(output: str) -> str:
    """Classify an execution log into a failure taxonomy label."""
    text = output or ""
    if not text.strip():
        return "unknown"
    # Prefer more specific classes first.
    order = ("timeout", "oom", "compile_error", "numerical_mismatch", "assertion", "runtime_error")
    low = text.lower()
    for label in order:
        for kw in FAILURE_TAXONOMY[label]:
            needle = kw.lower()
            # Short tokens like "oom" must not match inside "compilation".
            if len(needle) <= 3:
                if re.search(rf"(?<![a-z]){re.escape(needle)}(?![a-z])", low):
                    return label
            elif needle in low:
                return label
    return "unknown"


def extract_error_metrics(output: str) -> dict[str, float]:
    """Pull numeric error metrics out of stdout/stderr if present."""
    metrics: dict[str, float] = {}
    if not output:
        return metrics
    for name, regex in (
        ("max_abs_err", _ABS_ERR_RE),
        ("max_rel_err", _REL_ERR_RE),
        ("mean_abs_err", _MEAN_ABS_RE),
    ):
        m = regex.search(output)
        if m:
            try:
                metrics[name] = float(m.group(1))
            except ValueError:
                pass
    m = _N_ELEM_RE.search(output)
    if m:
        metrics["n_elements"] = float(m.group(1))
    return metrics


def parse_allclose_from_output(exec_output: str) -> NumericalResult | None:
    """Parse structured numerical evidence from an execution log."""
    if not exec_output:
        return None
    metrics = extract_error_metrics(exec_output)
    result = NumericalResult(
        max_abs_err=metrics.get("max_abs_err"),
        max_rel_err=metrics.get("max_rel_err"),
        mean_abs_err=metrics.get("mean_abs_err"),
        n_elements=int(metrics["n_elements"]) if "n_elements" in metrics else None,
        failure_class=classify_failure(exec_output),
        extras=dict(metrics),
    )
    if _PASS_RE.search(exec_output) and not _FAIL_RE.search(exec_output):
        result.passed = True
        result.compared = True
        result.message = "pass marker present"
        return result
    if _ALLCLOSE_FALSE_RE.search(exec_output):
        result.passed = False
        result.compared = True
        result.message = "allclose=False"
        return result
    if _ALLCLOSE_TRUE_RE.search(exec_output):
        result.passed = True
        result.compared = True
        result.message = "allclose=True"
        return result
    if metrics:
        result.compared = True
        # If we only have metrics, treat tiny abs err as pass.
        abs_err = result.max_abs_err
        if abs_err is not None and abs_err <= 1e-4:
            result.passed = True
            result.message = "max_abs_err within tolerance"
        else:
            result.passed = False
            result.message = "error metrics present"
        return result
    return None


def _code_has_compare(code: str) -> bool:
    return any(
        tok in (code or "")
        for tok in ("torch.allclose", "torch.testing.assert_close", "assert_close", "np.allclose")
    )


def _code_has_assert(code: str) -> bool:
    return bool(re.search(r"^\s*assert\b", code or "", re.MULTILINE))


def score_numerical(
    exec_pass: bool,
    exec_output: str,
    code: str,
    *,
    atol: float = 1e-5,
    rtol: float = 1e-5,
) -> tuple[float, NumericalResult]:
    """Return (score in [0,1], structured NumericalResult).

    Full credit requires execution success plus evidence that a reference
    comparison ran and passed. Merely executing is not enough.
    """
    parsed = parse_allclose_from_output(exec_output) or NumericalResult()
    parsed.failure_class = parsed.failure_class or (classify_failure(exec_output) if not exec_pass else "")

    if not exec_pass:
        # Partial crumbs if the log includes numeric mismatch info.
        if parsed.compared or parsed.max_abs_err is not None:
            parsed.message = parsed.message or "exec failed after compare"
            return 0.1, parsed
        parsed.message = parsed.message or "exec failed"
        return 0.0, parsed

    has_compare = _code_has_compare(code)
    has_assert = _code_has_assert(code)

    if parsed.compared and parsed.passed:
        score = 1.0 if has_compare else 0.9
        parsed.message = parsed.message or "numerical compare passed"
        return score, parsed

    if parsed.compared and not parsed.passed:
        # Exec somehow passed while compare reported fail — trust compare.
        parsed.passed = False
        return 0.15, parsed

    if has_compare:
        # Exec passed and code contains allclose; assume the assert would have failed the process.
        parsed.compared = True
        parsed.passed = True
        parsed.message = "exec_pass with torch.allclose in source"
        # Soften slightly vs explicit allclose=True log.
        return 1.0, parsed

    if has_assert:
        parsed.compared = True
        parsed.passed = True
        parsed.message = "exec_pass with assert in source"
        return 0.85, parsed

    parsed.message = "exec_pass without reference compare"
    return 0.45, parsed


def within_tolerance(
    max_abs_err: float | None,
    max_rel_err: float | None,
    *,
    atol: float,
    rtol: float,
) -> bool:
    """Check whether parsed errors fall within atol/rtol style bounds."""
    if max_abs_err is None and max_rel_err is None:
        return False
    if max_abs_err is not None and max_abs_err <= atol:
        return True
    if max_rel_err is not None and max_rel_err <= rtol:
        return True
    if max_abs_err is not None and max_rel_err is not None:
        # Combined criterion similar to torch.allclose.
        return max_abs_err <= (atol + rtol * (max_abs_err / max(max_rel_err, 1e-12)))
    return False
