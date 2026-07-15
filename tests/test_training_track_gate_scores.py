"""Tests for eval.training_track_gate's no-GPU eval.verify wiring (issue: #120)."""

import json
from pathlib import Path

from eval.training_track_gate import find_attestation_path, verify_remote_proof_bundle_scores


def test_find_attestation_path_matches_run_dir():
    paths = ["recipes/foo.yaml", "runs/2026-07-15-magicrails-hopper-v2/attestation.json"]
    assert find_attestation_path(paths) == "runs/2026-07-15-magicrails-hopper-v2/attestation.json"


def test_find_attestation_path_absent():
    assert find_attestation_path(["recipes/foo.yaml"]) is None
    assert find_attestation_path(None) is None


def _fake_snapshot(bundle_dir: Path):
    def fake_snapshot_download(*, repo_id, repo_type, token=None):
        return str(bundle_dir)

    return fake_snapshot_download


def test_verify_remote_proof_bundle_scores_skips_checkpoint_required(tmp_path, monkeypatch):
    from eval.canonical_dataset import canonical_hf_url

    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "manifest.json").write_text(json.dumps({"run_id": "r1", "dataset_url": canonical_hf_url()}))
    (bundle / "eval_scores.json").write_text(json.dumps({"scores": {"triton": 0.4}}))

    monkeypatch.setattr("huggingface_hub.snapshot_download", _fake_snapshot(bundle))

    issues = verify_remote_proof_bundle_scores(
        "org/repo",
        head_ref="HEAD",
        changed_paths=None,
    )
    # No attested samples and no checkpoint -> verify_submission's "checkpoint_required"
    # is deferred to off-CI validator verification, not a CI-gatable failure.
    assert issues == []


def test_verify_remote_proof_bundle_scores_surfaces_attested_mismatch(tmp_path, monkeypatch):
    import eval.verify as v

    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "manifest.json").write_text(json.dumps({"run_id": "r1"}))
    (bundle / "eval_scores.json").write_text(json.dumps({"scores": {"gsm8k": 0.64}}))

    monkeypatch.setattr("huggingface_hub.snapshot_download", _fake_snapshot(bundle))
    monkeypatch.setattr(
        v,
        "verify_submission",
        lambda *a, **k: {
            "verified": False,
            "reason": "attested_eval_samples_failed",
            "issues": ["claimed gsm8k 0.64 diverges from attested regression sample 0.74"],
            "label": "eval:REJECT",
        },
    )

    issues = verify_remote_proof_bundle_scores(
        "org/repo",
        head_ref="HEAD",
        changed_paths=None,
    )
    assert any("diverges from attested regression sample" in issue for issue in issues)


def test_verify_remote_proof_bundle_scores_passes_when_verified(tmp_path, monkeypatch):
    import eval.verify as v

    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "manifest.json").write_text(json.dumps({"run_id": "r1"}))
    (bundle / "eval_scores.json").write_text(json.dumps({"scores": {"gsm8k": 0.74}}))

    monkeypatch.setattr("huggingface_hub.snapshot_download", _fake_snapshot(bundle))
    monkeypatch.setattr(v, "verify_submission", lambda *a, **k: {"verified": True, "label": "eval:BASELINE"})

    issues = verify_remote_proof_bundle_scores(
        "org/repo",
        head_ref="HEAD",
        changed_paths=None,
    )
    assert issues == []


def test_verify_remote_proof_bundle_scores_reads_attestation_from_head_ref(tmp_path, monkeypatch):
    import eval.training_track_gate as gate
    import eval.verify as v

    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "manifest.json").write_text(json.dumps({"run_id": "r1"}))
    (bundle / "eval_scores.json").write_text(json.dumps({"scores": {"gsm8k": 0.74}}))

    monkeypatch.setattr("huggingface_hub.snapshot_download", _fake_snapshot(bundle))

    captured = {}

    def fake_show(ref, path):
        assert ref == "refs/remotes/origin/training-pr-head"
        assert path == "runs/r1/attestation.json"
        return json.dumps({"passed": True})

    def fake_verify_submission(bundle_dir, frontier, attestation=None):
        captured["attestation"] = attestation
        return {"verified": True, "label": "eval:BASELINE"}

    monkeypatch.setattr(gate, "_git_show", fake_show)
    monkeypatch.setattr(v, "verify_submission", fake_verify_submission)

    issues = verify_remote_proof_bundle_scores(
        "org/repo",
        head_ref="refs/remotes/origin/training-pr-head",
        changed_paths=["runs/r1/attestation.json"],
    )
    assert issues == []
    assert captured["attestation"] == {"passed": True}
