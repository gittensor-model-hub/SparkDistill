"""Reference oracle + repair agent tests (no GPU)."""

from __future__ import annotations

from pathlib import Path

from tritonbench.core.evaluator import TritonEvaluator
from tritonbench.core.reference_oracle import (
    build_oracle_index,
    compare_to_gold,
    gold_fingerprint,
    resolve_gold_kernel,
    score_against_oracle,
)
from tritonbench.core.repair_agent import RepairAgent, build_repair_user_prompt
from tritonbench.core.validator import TritonValidator

ROOT = Path(__file__).resolve().parents[1]


class _FakeModel:
    def __init__(self):
        self.calls = 0

    def generate(self, prompt, max_tokens=4096, temperature=0.2):
        self.calls += 1
        return """```python
import torch
import triton
import triton.language as tl

@triton.jit
def kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x, mask=mask)
```"""


class _FakeValidator(TritonValidator):
    def execute(self, code: str, timeout: int = 120):
        raise AssertionError("execute should be skipped in this test")


def test_gold_fingerprint_stable():
    src = "def f(x):\n    return x + 1\n"
    assert gold_fingerprint(src) == gold_fingerprint(src)


def test_compare_to_gold_similarity():
    gold = "@triton.jit\ndef add():\n    tl.load(x)\n    tl.store(y, x)\n"
    gen = "@triton.jit\ndef add():\n    tl.load(x)\n    tl.store(y, x)\n"
    sim = compare_to_gold(gen, gold)
    assert sim["similarity"] >= 0.8


def test_resolve_and_score_oracle():
    problem = {
        "id": "vector_addition",
        "gold_kernel": "data/TritonBench_G_v1/vector_addition.py",
        "source": {"channel": "G", "file": "vector_addition.py"},
    }
    path = resolve_gold_kernel(problem, data_root=ROOT)
    assert path is not None and path.exists()
    code = path.read_text(encoding="utf-8")
    score = score_against_oracle(problem, code, data_root=ROOT)
    assert score is not None and score > 0.5


def test_build_oracle_index_nonempty():
    index = build_oracle_index(ROOT / "data")
    assert len(index) > 100
    assert "G:vector_addition" in index or "vector_addition" in index


def test_repair_agent_skip_execute():
    model = _FakeModel()
    agent = RepairAgent(
        model,
        _FakeValidator(),
        TritonEvaluator(),
        max_turns=2,
        skip_execute=True,
    )
    problem = {
        "id": "toy",
        "prompt": "Write a copy kernel",
        "level": 1,
        "category": "kernel_generation",
    }
    episode = agent.run(problem)
    assert episode.turns
    assert model.calls >= 1
    assert episode.final_pass is True  # syntax ok + skip_execute


def test_build_repair_prompt_contains_failure():
    text = build_repair_user_prompt(
        {"id": "x", "title": "t", "prompt": "do thing"},
        "code here",
        "CompilationError: boom",
        "compile_error",
    )
    assert "compile_error" in text
    assert "CompilationError" in text
