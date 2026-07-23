"""Converter unit tests (no GPU)."""

from __future__ import annotations

from pathlib import Path

import yaml

from tritonbench.converters.difficulty_map import (
    difficulty_to_level,
    infer_category,
    infer_tags,
    level_dirname,
    slugify_id,
)
from tritonbench.converters.from_g_json import convert_g_channel, entry_to_problem
from tritonbench.converters.from_t_jsonl import convert_t_channel
from tritonbench.converters.generate_corpus import BUGFIX_TEMPLATES, generate_corpus, synthetic_bugfix_problems
from tritonbench.converters.validate_problem import (
    ProblemSchemaError,
    validate_corpus,
    validate_problem,
    validate_problem_strict,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def test_slugify_and_level_dirname():
    assert slugify_id("Foo/Bar-Baz.py") == "bar_baz"
    assert level_dirname(1) == "level1_basic"
    assert level_dirname("bugfix") == "bugfix"


def test_difficulty_and_tags():
    assert difficulty_to_level("1") == 1
    assert difficulty_to_level("5") == 4
    tags = infer_tags("flash attention softmax kernel with kv_cache")
    assert "attention" in tags
    assert "softmax" in tags
    boosted = difficulty_to_level("2", tags=["attention", "matmul"])
    assert boosted >= 2


def test_infer_category():
    assert infer_category({"torch_code": "x = 1"}, "T") == "kernel_translation"
    assert infer_category({"file": "wrong_mask.py", "simp_instru": "fix the bug"}, "G") == "kernel_debugging"
    assert infer_category({"file": "add.py", "simp_instru": "write add"}, "G") == "kernel_generation"


def test_g_entry_to_problem():
    entry = {
        "file": "vector_addition.py",
        "repo": "example/repo",
        "simp_instru": "Write a vector addition kernel with masking for arbitrary sizes and a launcher.",
        "comp_instru": "Write a complete vector addition Triton kernel with autotune and torch.allclose tests on CUDA.",
        "difficulty": "1",
        "star": 10,
    }
    prob = entry_to_problem(entry)
    assert prob is not None
    assert prob["id"] == "vector_addition"
    assert prob["level"] == 1
    assert "Triton 3.7.1" in prob["prompt"]
    assert prob["gold_kernel"].endswith("vector_addition.py")
    assert validate_problem(prob) == [] or all(e.startswith("unknown field:") for e in validate_problem(prob))


def test_convert_g_and_t_channels():
    g = convert_g_channel(DATA / "TritonBench_G_v1.json", limit=5)
    assert len(g) == 5
    t = convert_t_channel(DATA / "TritonBench_T_v1.jsonl", limit=5)
    assert len(t) == 5
    assert all("prompt" in p for p in g + t)


def test_bugfix_templates_schema():
    assert len(BUGFIX_TEMPLATES) >= 20
    for prob in synthetic_bugfix_problems():
        errs = [e for e in validate_problem(prob) if not e.startswith("unknown field:")]
        assert errs == [], (prob["id"], errs)


def test_validate_problem_strict_rejects_bad():
    try:
        validate_problem_strict({"id": "BAD ID", "level": 9, "category": "nope", "title": "t", "prompt": "short"})
        assert False, "expected ProblemSchemaError"
    except ProblemSchemaError:
        pass


def test_generate_corpus_dry_run(tmp_path: Path):
    stats = generate_corpus(
        data_dir=DATA,
        problems_root=tmp_path / "problems",
        channels=("G", "T"),
        dry_run=True,
        limit=8,
    )
    assert stats["written"] > 0
    assert stats["g"]["count"] + stats["t"]["count"] <= 8


def test_generate_corpus_writes_yaml(tmp_path: Path):
    problems_root = tmp_path / "problems"
    stats = generate_corpus(
        data_dir=DATA,
        problems_root=problems_root,
        channels=("G",),
        dry_run=False,
        limit=3,
        include_bugfix=True,
    )
    assert stats["written"] >= 3
    yamls = list(problems_root.rglob("*.yaml"))
    assert len(yamls) >= 3
    sample = yaml.safe_load(yamls[0].read_text(encoding="utf-8"))
    assert "id" in sample and "prompt" in sample
    report = validate_corpus(problems_root)
    assert report["failed"] == 0
