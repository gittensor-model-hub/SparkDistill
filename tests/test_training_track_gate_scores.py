"""Tests for eval.training_track_gate's no-GPU eval.verify wiring (issue: #120)."""

import json
from pathlib import Path

from eval.training_track_gate import (
    EVAL_LABELS,
    find_attestation_path,
    update_pr_eval_label,
    verify_remote_proof_bundle_scores,
)


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

    issues, eval_label = verify_remote_proof_bundle_scores(
        "org/repo",
        head_ref="HEAD",
        changed_paths=None,
    )
    # No attested samples and no checkpoint -> verify_submission's "checkpoint_required"
    # is deferred to off-CI validator verification, not a CI-gatable failure.
    assert issues == []
    assert eval_label is None


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

    issues, eval_label = verify_remote_proof_bundle_scores(
        "org/repo",
        head_ref="HEAD",
        changed_paths=None,
    )
    assert any("diverges from attested regression sample" in issue for issue in issues)
    assert eval_label == "eval:REJECT"


def test_verify_remote_proof_bundle_scores_passes_when_verified(tmp_path, monkeypatch):
    import eval.verify as v

    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "manifest.json").write_text(json.dumps({"run_id": "r1"}))
    (bundle / "eval_scores.json").write_text(json.dumps({"scores": {"gsm8k": 0.74}}))

    monkeypatch.setattr("huggingface_hub.snapshot_download", _fake_snapshot(bundle))
    monkeypatch.setattr(v, "verify_submission", lambda *a, **k: {"verified": True, "label": "eval:BASELINE"})

    issues, eval_label = verify_remote_proof_bundle_scores(
        "org/repo",
        head_ref="HEAD",
        changed_paths=None,
    )
    assert issues == []
    assert eval_label == "eval:BASELINE"


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

    issues, eval_label = verify_remote_proof_bundle_scores(
        "org/repo",
        head_ref="refs/remotes/origin/training-pr-head",
        changed_paths=["runs/r1/attestation.json"],
    )
    assert issues == []
    assert eval_label == "eval:BASELINE"
    assert captured["attestation"] == {"passed": True}


def test_update_pr_eval_label_rejects_unknown_label():
    assert update_pr_eval_label(1, "eval:bogus") == ["refusing to apply unknown eval label 'eval:bogus'"]


def test_update_pr_eval_label_applies_known_label(monkeypatch):
    from types import SimpleNamespace

    import eval.training_track_gate as gate

    calls: list[list[str]] = []

    def fake_run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(gate.subprocess, "run", fake_run)
    assert update_pr_eval_label(120, "eval:BASELINE") == []
    assert ["gh", "pr", "edit", "120", "--add-label", "eval:BASELINE"] in calls
    # every other eval:* label gets removed so exactly one is ever active
    for other in EVAL_LABELS - {"eval:BASELINE"}:
        assert ["gh", "pr", "edit", "120", "--remove-label", other] in calls


def test_gate_training_pr_threads_eval_label(monkeypatch):
    import eval.training_track_gate as gate

    monkeypatch.setattr(gate, "should_enforce_training_gate", lambda *a, **k: True)
    monkeypatch.setattr(gate, "_canonical_sft_sha256s_for_pr_window", lambda **k: {"a" * 64})
    monkeypatch.setattr(gate, "validate_changed_paths", lambda *a, **k: [])
    monkeypatch.setattr(gate, "validate_recipe_paths_in_ref", lambda *a, **k: [])
    monkeypatch.setattr(gate, "validate_pr_body_canonical_pin", lambda *a, **k: [])
    monkeypatch.setattr(gate, "validate_pr_body_proof_bundle", lambda *a, **k: [])
    monkeypatch.setattr(gate, "is_training_track_pr", lambda *a, **k: True)
    monkeypatch.setattr(gate, "load_canonical", lambda: {})
    monkeypatch.setattr(gate, "parse_proof_bundle_hf_repo", lambda *a, **k: "org/repo")
    monkeypatch.setattr(gate, "verify_remote_proof_bundle", lambda *a, **k: [])
    monkeypatch.setattr(gate, "verify_remote_proof_bundle_scores", lambda *a, **k: ([], "eval:XL"))

    report = gate.gate_training_pr(
        head_ref="HEAD",
        changed_paths=["recipes/foo.yaml"],
        pr_body="- [x] Training/evaluation improvement",
        verify_hf_pin=False,
    )
    assert report["verified"] is True
    assert report["label"] == "training:valid"
    assert report["eval_label"] == "eval:XL"
