import json
from pathlib import Path

from eval import frontiers as frontiers_mod
from eval.frontiers import FRONTIERS_PATH, load_frontier_record, load_frontier_scores, load_frontiers, merge_frontier_record


def test_load_frontiers_from_repo_file():
    frontiers = load_frontiers(FRONTIERS_PATH)
    assert set(frontiers) == {"blackwell", "hopper"}
    assert frontiers["blackwell"]["scores"]["triton"] > 0
    assert frontiers["hopper"]["scores"] == {}


def test_load_frontier_scores_returns_none_for_empty_bucket():
    assert load_frontier_scores("hopper", path=FRONTIERS_PATH) is None


def test_load_frontier_scores_blackwell_has_triton():
    scores = load_frontier_scores("blackwell", path=FRONTIERS_PATH)
    assert scores is not None
    assert scores["gsm8k"] == 0.6


def test_legacy_frontier_json_seeds_blackwell_only(tmp_path: Path):
    legacy = {
        "run_id": "legacy-run",
        "proof_bundle": "https://example.com/bundle",
        "scores": {"gsm8k": 0.55, "triton": 0.40},
    }
    frontiers_path = tmp_path / "frontiers.json"
    legacy_path = tmp_path / "frontier.json"
    legacy_path.write_text(json.dumps(legacy), encoding="utf-8")

    original_legacy = frontiers_mod.LEGACY_FRONTIER_PATH
    original_frontiers = frontiers_mod.FRONTIERS_PATH
    try:
        frontiers_mod.LEGACY_FRONTIER_PATH = legacy_path
        frontiers_mod.FRONTIERS_PATH = frontiers_path
        loaded = load_frontiers(frontiers_path)
    finally:
        frontiers_mod.LEGACY_FRONTIER_PATH = original_legacy
        frontiers_mod.FRONTIERS_PATH = original_frontiers

    assert loaded["blackwell"]["run_id"] == "legacy-run"
    assert loaded["blackwell"]["scores"]["triton"] == 0.40
    assert loaded["hopper"]["scores"] == {}


def test_merge_frontier_record_updates_arch_bucket_only():
    frontiers = load_frontiers(FRONTIERS_PATH)
    updated, updates = merge_frontier_record(
        frontiers,
        "hopper",
        {"gsm8k": 0.7, "triton": 0.5},
        run_id="hopper-baseline-001",
        proof_bundle="https://example.com/hopper",
    )
    assert updates == ["gsm8k", "triton"]
    assert updated["hopper"]["scores"]["triton"] == 0.5
    assert updated["blackwell"]["scores"]["triton"] == frontiers["blackwell"]["scores"]["triton"]


def test_load_frontier_record_preserves_metadata():
    record = load_frontier_record("blackwell", path=FRONTIERS_PATH)
    assert record["gpu_architecture"] == "blackwell"
    assert record["run_id"] == "2026-07-11-qwen3.5-4b-mining-001"
