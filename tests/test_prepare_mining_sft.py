"""Tests for eval.prepare_mining_sft."""

import json
from pathlib import Path

from eval.prepare_mining_sft import export_mining_sft


def test_export_mining_sft_writes_messages_only(tmp_path: Path, monkeypatch):
    rows = [
        {"messages": [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]},
        {"messages": [{"role": "user", "content": "c"}, {"role": "assistant", "content": "d"}], "metadata": {"x": 1}},
    ]

    class _Split:
        def __iter__(self):
            return iter(rows)

        def __len__(self):
            return len(rows)

    def _fake_load_dataset(repo_id, split, token=None):
        assert repo_id == "gittensor-model-hub/sparkproof-mining"
        assert split == "train"
        return _Split()

    monkeypatch.setattr("datasets.load_dataset", _fake_load_dataset)
    monkeypatch.setattr("eval.prepare_mining_sft.verify_remote_matches_pin", lambda **kwargs: [])

    out = tmp_path / "mining.jsonl"
    result = export_mining_sft(
        out_path=out,
        repo_id="gittensor-model-hub/sparkproof-mining",
        hf_token="tok",
        verify_pin=False,
    )
    assert result["rows_written"] == 2

    lines = out.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    first = json.loads(lines[0])
    second = json.loads(lines[1])
    assert "metadata" not in first
    assert "metadata" not in second
    assert first["messages"][0]["content"] == "a"
