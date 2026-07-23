"""Distill bridge + evaluator detailed scoring tests (no GPU)."""

from __future__ import annotations

from pathlib import Path

from distill.data_generator import DataGenerator, MockTeacherClient, load_problems
from distill.export_sft import (
    dedupe_by_prompt_hash,
    export_sft_records,
    to_completion_record,
    to_messages_record,
)
from distill.prompt_builder import build_system_prompt, build_teacher_prompt
from distill.quality_filter import QualityFilter
from tritonbench.core.evaluator import TritonEvaluator
from tritonbench.core.validator import TritonValidator

ROOT = Path(__file__).resolve().parents[1]
PROBLEMS = ROOT / "tritonbench" / "problems"


def test_build_prompts():
    system = build_system_prompt()
    assert "Triton 3.7.1" in system
    assert "Blackwell" in system
    problem = {
        "id": "vector_add",
        "category": "kernel_generation",
        "prompt": "Write vector add",
        "tags": ["elementwise"],
    }
    prompt = build_teacher_prompt(problem, include_few_shot=True)
    assert prompt["system"]
    assert "vector add" in prompt["user"].lower() or "Write vector add" in prompt["user"]


def test_messages_and_completion_export(tmp_path: Path):
    problem = {"id": "p", "level": 1, "category": "kernel_generation", "prompt": "hi", "tags": []}
    msg = to_messages_record(problem, "assistant answer", reasoning="think hard", model="t1", system="sys")
    assert msg["messages"][-1]["content"].startswith("<think>")
    comp = to_completion_record(problem, "ans", reasoning=None, model="t1")
    assert "completion" in comp
    path = tmp_path / "sft.jsonl"
    n = export_sft_records([msg, msg], path)
    assert n == 2
    deduped = dedupe_by_prompt_hash([msg, msg])
    assert len(deduped) == 1


def test_quality_filter_skip_execute():
    qf = QualityFilter(
        TritonValidator(),
        TritonEvaluator(),
        require_exec=True,
        skip_execute=True,
        min_composite=0.0,
    )
    code = """```python
import triton
import triton.language as tl
@triton.jit
def k(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x, mask=mask)
```"""
    result = qf.check(code, problem={"id": "t", "required_patterns": ["@triton.jit"]})
    assert result.syntax_ok
    assert result.accepted


def test_data_generator_mock(tmp_path: Path):
    # Prefer real seed problems if present; otherwise synthesize one YAML.
    if (PROBLEMS / "level1_basic" / "vector_add.yaml").exists():
        problems = load_problems(PROBLEMS, levels=[1], include_bugfix=False, limit=2)
    else:
        problems = [{"id": "toy", "level": 1, "category": "kernel_generation", "prompt": "write add", "tags": []}]
    gen = DataGenerator(MockTeacherClient(), quality_filter=None)
    out = tmp_path / "out.jsonl"
    stats = gen.generate_to_jsonl(problems, out)
    assert stats["written"] == len(problems)
    assert out.exists() and out.stat().st_size > 0


def test_evaluator_detailed_keys():
    e = TritonEvaluator()
    code = "@triton.jit\ndef k():\n    pass\ndef launch():\n    grid=1\nassert torch.allclose(a,b)\n"
    detailed = e.score_detailed({}, code, True, "TRITONBENCH_PASS\nallclose=True")
    assert "numerical" in detailed and "performance" in detailed
    assert 0.0 <= detailed["composite_score"] <= 1.0
    # Backward compatible score() keys
    scores = e.score({}, code, True, "TRITONBENCH_PASS\nallclose=True")
    for k in ("correctness", "api_modernity", "perf_awareness", "completeness", "code_quality", "composite_score"):
        assert k in scores
