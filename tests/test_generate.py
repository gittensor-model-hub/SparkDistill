from pathlib import Path

from teacher.generate import _iter_prompts


def _write(tmp_path: Path, content: str) -> Path:
    path = tmp_path / "prompts.jsonl"
    path.write_text(content, encoding="utf-8")
    return path


def test_iter_prompts_yields_all_records_without_limit(tmp_path):
    path = _write(tmp_path, '{"prompt": "a"}\n{"prompt": "b"}\n{"prompt": "c"}\n')

    got = list(_iter_prompts(path, limit=None))

    assert [r["prompt"] for r in got] == ["a", "b", "c"]


def test_iter_prompts_skips_blank_lines(tmp_path):
    path = _write(tmp_path, '{"prompt": "a"}\n\n\n{"prompt": "b"}\n')

    got = list(_iter_prompts(path, limit=None))

    assert [r["prompt"] for r in got] == ["a", "b"]


def test_limit_counts_prompts_not_file_lines(tmp_path):
    # A blank line sits among the first prompts; --limit must still yield 3 prompts,
    # not stop early because the blank line consumed the limit budget.
    path = _write(
        tmp_path,
        '{"prompt": "a"}\n{"prompt": "b"}\n\n{"prompt": "c"}\n{"prompt": "d"}\n{"prompt": "e"}\n',
    )

    got = list(_iter_prompts(path, limit=3))

    assert [r["prompt"] for r in got] == ["a", "b", "c"]


def test_limit_larger_than_available_yields_everything(tmp_path):
    path = _write(tmp_path, '{"prompt": "a"}\n{"prompt": "b"}\n')

    got = list(_iter_prompts(path, limit=10))

    assert [r["prompt"] for r in got] == ["a", "b"]
