import hashlib
import json

import pytest

from proof.bundle import build_bundle, checkpoint_manifest, claim_sha256


def test_build_bundle_is_proof_only_by_default(tmp_path):
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "adapter_model.bin").write_text("fake-weights")

    scores_path = tmp_path / "candidate.json"
    scores_path.write_text(json.dumps({"checkpoint": "outputs/x", "scores": {"gsm8k": 0.88}}))

    out_dir = tmp_path / "bundle"
    bundle = build_bundle(checkpoint_dir, scores_path, out_dir, run_id="run-001", base_model="Qwen/Qwen3.5-4B")

    assert bundle.run_id == "run-001"
    # No weights in the bundle — only the per-file hash manifest of the checkpoint.
    assert not (out_dir / "checkpoint").exists()
    manifest = json.loads((out_dir / "manifest.json").read_text())
    expected_sha = hashlib.sha256(b"fake-weights").hexdigest()
    assert manifest["checkpoint_manifest"] == {"adapter_model.bin": expected_sha}
    assert json.loads((out_dir / "eval_scores.json").read_text())["scores"] == {"gsm8k": 0.88}
    assert manifest["run_id"] == "run-001"
    assert manifest["base_model"] == "Qwen/Qwen3.5-4B"
    assert "train_hours" not in manifest
    assert bundle.claim_sha256 == claim_sha256(out_dir)


def test_build_bundle_include_checkpoint_restores_legacy_copy(tmp_path):
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "adapter_model.bin").write_text("fake-weights")
    scores_path = tmp_path / "candidate.json"
    scores_path.write_text(json.dumps({"scores": {"gsm8k": 0.88}}))

    out_dir = tmp_path / "bundle"
    build_bundle(
        checkpoint_dir, scores_path, out_dir, run_id="run-001", base_model="Qwen/Qwen3.5-4B", include_checkpoint=True
    )
    assert (out_dir / "checkpoint" / "adapter_model.bin").read_text() == "fake-weights"


def test_checkpoint_manifest_covers_nested_files(tmp_path):
    checkpoint_dir = tmp_path / "ckpt"
    (checkpoint_dir / "sub").mkdir(parents=True)
    (checkpoint_dir / "a.safetensors").write_text("aa")
    (checkpoint_dir / "sub" / "b.json").write_text("bb")
    manifest = checkpoint_manifest(checkpoint_dir)
    assert sorted(manifest) == ["a.safetensors", "sub/b.json"]


def test_claim_sha256_changes_when_scores_change(tmp_path):
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "w.bin").write_text("w")
    scores_path = tmp_path / "candidate.json"
    scores_path.write_text(json.dumps({"scores": {"gsm8k": 0.88}}))

    bundle = build_bundle(checkpoint_dir, scores_path, tmp_path / "b1", run_id="r", base_model="m")
    (tmp_path / "b1" / "eval_scores.json").write_text(json.dumps({"scores": {"gsm8k": 0.99}}))
    assert claim_sha256(tmp_path / "b1") != bundle.claim_sha256


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


def test_build_bundle_rejects_missing_checkpoint(tmp_path):
    # A typo'd / wrong --checkpoint must fail loudly, not silently emit an empty
    # checkpoint_manifest and a vacuous claim_sha256.
    scores_path = tmp_path / "candidate.json"
    scores_path.write_text(json.dumps({"scores": {"gsm8k": 0.88}}))

    with pytest.raises(NotADirectoryError):
        build_bundle(
            tmp_path / "does-not-exist",
            scores_path,
            tmp_path / "bundle",
            run_id="run-x",
            base_model="Qwen/Qwen3.5-4B",
        )
    # Nothing should be written for an invalid checkpoint.
    assert not (tmp_path / "bundle" / "manifest.json").exists()


def test_build_bundle_rejects_empty_checkpoint_dir(tmp_path):
    # An existing but empty checkpoint directory is equally vacuous.
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    scores_path = tmp_path / "candidate.json"
    scores_path.write_text(json.dumps({"scores": {"gsm8k": 0.88}}))

    with pytest.raises(ValueError):
        build_bundle(
            checkpoint_dir,
            scores_path,
            tmp_path / "bundle",
            run_id="run-x",
            base_model="Qwen/Qwen3.5-4B",
        )


def test_build_bundle_records_mix_manifest(tmp_path):
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "adapter_model.bin").write_text("fake-weights")

    scores_path = tmp_path / "candidate.json"
    scores_path.write_text(json.dumps({"scores": {"gsm8k": 0.88}}))

    mix_manifest = tmp_path / "mix_manifest.json"
    mix_manifest.write_text(
        json.dumps(
            {
                "mix_version": "sparkdistill-mix-v0",
                "mix_id": "mix-001",
                "rows_total": 42,
                "components": [{"miner": "alice", "trajectories_sha256": "a" * 64}],
            }
        ),
        encoding="utf-8",
    )

    bundle = build_bundle(
        checkpoint_dir,
        scores_path,
        tmp_path / "bundle",
        run_id="run-003",
        base_model="Qwen/Qwen3.5-4B",
        mix_manifest=mix_manifest,
    )

    manifest = json.loads((bundle.bundle_dir / "manifest.json").read_text())
    assert manifest["mix_id"] == "mix-001"
    assert manifest["mix_rows_total"] == 42
    assert manifest["mix_component_count"] == 1
    assert (bundle.bundle_dir / "mix_manifest.json").exists()
