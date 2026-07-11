import json

from proof.bundle import build_bundle


def test_build_bundle_copies_checkpoint_and_scores(tmp_path):
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "adapter_model.bin").write_text("fake-weights")

    scores_path = tmp_path / "candidate.json"
    scores_path.write_text(json.dumps({"checkpoint": "outputs/x", "scores": {"gsm8k": 0.88}}))

    out_dir = tmp_path / "bundle"
    bundle = build_bundle(checkpoint_dir, scores_path, out_dir, run_id="run-001", base_model="Qwen/Qwen3.5-4B")

    assert bundle.run_id == "run-001"
    assert (out_dir / "checkpoint" / "adapter_model.bin").read_text() == "fake-weights"
    assert json.loads((out_dir / "eval_scores.json").read_text())["scores"] == {"gsm8k": 0.88}
    manifest = json.loads((out_dir / "manifest.json").read_text())
    assert manifest["run_id"] == "run-001"
    assert manifest["base_model"] == "Qwen/Qwen3.5-4B"
    assert "train_hours" not in manifest


def test_build_bundle_records_training_claims(tmp_path):
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "adapter_model.bin").write_text("fake-weights")

    scores_path = tmp_path / "candidate.json"
    scores_path.write_text(json.dumps({"scores": {"gsm8k": 0.88}}))

    bundle = build_bundle(
        checkpoint_dir,
        scores_path,
        tmp_path / "bundle",
        run_id="run-002",
        base_model="Qwen/Qwen3.5-4B",
        train_hours=4.2,
        train_gpu="NVIDIA RTX PRO 6000 Blackwell",
        dataset_url="https://huggingface.co/datasets/miner/sparkproof-triton-v0",
    )

    manifest = json.loads((bundle.bundle_dir / "manifest.json").read_text())
    assert manifest["train_hours"] == 4.2
    assert manifest["train_gpu"] == "NVIDIA RTX PRO 6000 Blackwell"
    assert manifest["dataset_url"] == "https://huggingface.co/datasets/miner/sparkproof-triton-v0"
