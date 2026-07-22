import json
from pathlib import Path

from eval import frontiers as frontiers_mod
from eval.frontiers import (
    FRONTIERS_PATH,
    apply_verified_report_to_frontiers,
    candidate_scores_from_report,
    load_frontier_record,
    load_frontier_scores,
    load_frontiers,
    merge_frontier_record,
)


def test_load_frontiers_from_repo_file():
    frontiers = load_frontiers(FRONTIERS_PATH)
    assert set(frontiers) == {"blackwell", "hopper"}
    assert frontiers["blackwell"]["scores"]["triton"] > 0
    assert frontiers["hopper"]["run_id"] == "2026-07-15-magicrails-hopper-v2"
    assert frontiers["hopper"]["scores"]["gsm8k"] == 0.74


def test_load_frontier_scores_hopper_seeded():
    scores = load_frontier_scores("hopper", path=FRONTIERS_PATH)
    assert scores is not None
    assert scores["triton"] == 0.3719444444444444


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
        {"gsm8k": 0.8, "triton": 0.5},
        run_id="hopper-improve-001",
        proof_bundle="https://example.com/hopper",
    )
    assert "gsm8k" in updates and "triton" in updates
    assert updated["hopper"]["scores"]["triton"] == 0.5
    assert updated["blackwell"]["scores"]["triton"] == frontiers["blackwell"]["scores"]["triton"]


def test_load_frontier_record_preserves_metadata():
    record = load_frontier_record("blackwell", path=FRONTIERS_PATH)
    assert record["gpu_architecture"] == "blackwell"
    assert record["run_id"] == "2026-07-11-qwen3.5-4b-mining-001"


def test_apply_verified_report_seeds_empty_bucket(tmp_path: Path):
    frontiers_path = tmp_path / "frontiers.json"
    frontiers_path.write_text(
        json.dumps(
            {
                "blackwell": {"gpu_architecture": "blackwell", "run_id": None, "proof_bundle": None, "scores": {}},
                "hopper": {"gpu_architecture": "hopper", "run_id": None, "proof_bundle": None, "scores": {}},
            }
        ),
        encoding="utf-8",
    )
    report = {
        "verified": True,
        "label": "eval:BASELINE",
        "run_id": "hopper-baseline",
        "gpu_architecture": "hopper",
        "per_benchmark": {
            "gsm8k": {"candidate": 0.74, "frontier": None},
            "triton": {"candidate": 0.37, "frontier": None},
            "triton_syntax_pass_rate": {"candidate": 0.66, "frontier": None},
        },
    }
    updates = apply_verified_report_to_frontiers(
        report,
        proof_bundle="https://huggingface.co/org/hopper-proof",
        path=frontiers_path,
    )
    assert "gsm8k" in updates and "triton" in updates
    loaded = json.loads(frontiers_path.read_text(encoding="utf-8"))
    assert loaded["hopper"]["run_id"] == "hopper-baseline"
    assert loaded["hopper"]["scores"]["gsm8k"] == 0.74
    assert loaded["hopper"]["scores"]["triton_syntax_pass_rate"] == 0.66
    legacy = json.loads((tmp_path / "frontier.json").read_text(encoding="utf-8"))
    assert legacy["scores"] == {}


def test_apply_verified_report_skips_reject(tmp_path: Path):
    frontiers_path = tmp_path / "frontiers.json"
    frontiers_path.write_text(
        json.dumps(
            {
                "blackwell": {
                    "gpu_architecture": "blackwell",
                    "run_id": "b1",
                    "proof_bundle": None,
                    "scores": {"triton": 0.4},
                },
                "hopper": {"gpu_architecture": "hopper", "run_id": None, "proof_bundle": None, "scores": {}},
            }
        ),
        encoding="utf-8",
    )
    before = frontiers_path.read_text(encoding="utf-8")
    updates = apply_verified_report_to_frontiers(
        {
            "verified": True,
            "label": "eval:REJECT",
            "run_id": "bad",
            "gpu_architecture": "blackwell",
            "per_benchmark": {"triton": {"candidate": 0.9, "frontier": 0.4}},
        },
        proof_bundle="https://example.com/x",
        path=frontiers_path,
    )
    assert updates == []
    assert frontiers_path.read_text(encoding="utf-8") == before


def test_candidate_scores_from_report():
    assert candidate_scores_from_report(
        {"per_benchmark": {"gsm8k": {"candidate": 0.5}, "triton": {"candidate": 0.2}}}
    ) == {"gsm8k": 0.5, "triton": 0.2}

