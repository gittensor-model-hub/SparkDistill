"""Tests for eval.train_prep."""

import json
from pathlib import Path

import yaml

from eval.train_prep import MIN_SAMPLE_PACKING_ROWS, count_jsonl_rows, prepare_train_recipe


def _write_jsonl(path: Path, rows: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for i in range(rows):
            handle.write(
                json.dumps(
                    {
                        "messages": [
                            {"role": "user", "content": f"p{i}"},
                            {"role": "assistant", "content": f"a{i}"},
                        ]
                    }
                )
                + "\n"
            )


def test_count_jsonl_rows(tmp_path: Path):
    path = tmp_path / "data.jsonl"
    _write_jsonl(path, 3)
    assert count_jsonl_rows(path) == 3


def test_prepare_train_recipe_resolves_paths_and_disables_packing(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("eval.train_prep._has_flash_attn", lambda: False)
    monkeypatch.setattr("eval.train_prep._has_flash_attn_3", lambda: False)
    monkeypatch.setattr("eval.train_prep._has_cut_cross_entropy", lambda: False)
    root = tmp_path / "distill"
    data = root / "data/processed/tiny.jsonl"
    _write_jsonl(data, MIN_SAMPLE_PACKING_ROWS - 1)
    recipe = root / "recipes/demo/sft.yaml"
    recipe.parent.mkdir(parents=True)
    recipe.write_text(
        yaml.safe_dump(
            {
                "datasets": [{"path": "data/processed/tiny.jsonl"}],
                "dataset_prepared_path": "data/prepared/demo",
                "output_dir": "outputs/demo",
                "sample_packing": True,
                "pad_to_sequence_len": True,
                "attn_implementation": "flash_attention_2",
                "plugins": ["axolotl.integrations.cut_cross_entropy.CutCrossEntropyPlugin"],
            }
        ),
        encoding="utf-8",
    )

    result = prepare_train_recipe(recipe_path=recipe, distill_root=root)
    prepared = yaml.safe_load(Path(result["prepared_recipe"]).read_text(encoding="utf-8"))

    assert prepared["datasets"][0]["path"] == str(data.resolve())
    assert prepared["dataset_prepared_path"] == str((root / "data/prepared/demo").resolve())
    assert prepared["output_dir"] == str((root / "outputs/demo").resolve())
    assert prepared["sample_packing"] is False
    assert prepared["pad_to_sequence_len"] is False
    assert prepared["attn_implementation"] == "sdpa"
    assert "plugins" not in prepared
    assert result["row_count"] == MIN_SAMPLE_PACKING_ROWS - 1
    assert any("sample_packing disabled" in note for note in result["notes"])


def test_multipack_guard_uses_total_rows_across_datasets(tmp_path: Path, monkeypatch):
    # Two shards whose sizes are each below the threshold but sum comfortably above it:
    # packing must stay enabled because Axolotl concatenates them.
    monkeypatch.setattr("eval.train_prep._has_flash_attn", lambda: False)
    monkeypatch.setattr("eval.train_prep._has_flash_attn_3", lambda: False)
    monkeypatch.setattr("eval.train_prep._has_cut_cross_entropy", lambda: False)
    root = tmp_path / "distill"
    a = root / "data/processed/a.jsonl"
    b = root / "data/processed/b.jsonl"
    _write_jsonl(a, MIN_SAMPLE_PACKING_ROWS - 12)
    _write_jsonl(b, MIN_SAMPLE_PACKING_ROWS - 12)  # each < 32, together > 32
    recipe = root / "recipes/demo/sft.yaml"
    recipe.parent.mkdir(parents=True)
    recipe.write_text(
        yaml.safe_dump(
            {
                "datasets": [
                    {"path": "data/processed/a.jsonl"},
                    {"path": "data/processed/b.jsonl"},
                ],
                "sample_packing": True,
                "pad_to_sequence_len": True,
            }
        ),
        encoding="utf-8",
    )

    result = prepare_train_recipe(recipe_path=recipe, distill_root=root)
    prepared = yaml.safe_load(Path(result["prepared_recipe"]).read_text(encoding="utf-8"))

    assert result["row_count"] == 2 * (MIN_SAMPLE_PACKING_ROWS - 12)
    assert prepared["sample_packing"] is True
    assert prepared["pad_to_sequence_len"] is True
    assert not any("sample_packing disabled" in note for note in result["notes"])


def test_multipack_guard_disables_when_total_below_threshold(tmp_path: Path, monkeypatch):
    # A large trailing shard must not mask a tiny total: 5 + 5 rows stays below 32.
    monkeypatch.setattr("eval.train_prep._has_flash_attn", lambda: False)
    monkeypatch.setattr("eval.train_prep._has_flash_attn_3", lambda: False)
    monkeypatch.setattr("eval.train_prep._has_cut_cross_entropy", lambda: False)
    root = tmp_path / "distill"
    a = root / "data/processed/a.jsonl"
    b = root / "data/processed/b.jsonl"
    _write_jsonl(a, 5)
    _write_jsonl(b, 5)
    recipe = root / "recipes/demo/sft.yaml"
    recipe.parent.mkdir(parents=True)
    recipe.write_text(
        yaml.safe_dump(
            {
                "datasets": [
                    {"path": "data/processed/a.jsonl"},
                    {"path": "data/processed/b.jsonl"},
                ],
                "sample_packing": True,
            }
        ),
        encoding="utf-8",
    )

    result = prepare_train_recipe(recipe_path=recipe, distill_root=root)
    prepared = yaml.safe_load(Path(result["prepared_recipe"]).read_text(encoding="utf-8"))

    assert result["row_count"] == 10
    assert prepared["sample_packing"] is False


def test_prepare_train_recipe_upgrades_to_flash_attention_3(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("eval.train_prep._has_flash_attn", lambda: True)
    monkeypatch.setattr("eval.train_prep._has_flash_attn_3", lambda: True)
    monkeypatch.setattr("eval.train_prep._has_cut_cross_entropy", lambda: False)
    root = tmp_path / "distill"
    data = root / "data/processed/tiny.jsonl"
    _write_jsonl(data, MIN_SAMPLE_PACKING_ROWS)
    recipe = root / "recipes/demo/sft.yaml"
    recipe.parent.mkdir(parents=True)
    recipe.write_text(
        yaml.safe_dump(
            {
                "datasets": [{"path": "data/processed/tiny.jsonl"}],
                "attn_implementation": "flash_attention_2",
            }
        ),
        encoding="utf-8",
    )

    result = prepare_train_recipe(recipe_path=recipe, distill_root=root)
    prepared = yaml.safe_load(Path(result["prepared_recipe"]).read_text(encoding="utf-8"))

    assert prepared["attn_implementation"] == "flash_attention_3"
    assert any("flash_attention_3" in note for note in result["notes"])
