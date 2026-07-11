import hashlib
import json

from eval.dataset_verify import check_proof_dir, size_label, verify_dataset_submission


def _write_proof_dir(tmp_path, *, rows=3, attested=True, gate_passed=True, tamper_rows=False):
    proof = tmp_path / "proof"
    proof.mkdir()

    traj_lines = "\n".join(json.dumps({"prompt": f"p{i}", "response": f"r{i}"}) for i in range(rows)) + "\n"
    (proof / "trajectories.jsonl").write_text(traj_lines)
    sha = hashlib.sha256(traj_lines.encode()).hexdigest()

    (proof / "manifest.json").write_text(json.dumps({"version": "sparkproof-2"}))
    (proof / "prompts.jsonl").write_text(json.dumps({"prompt": "p0"}) + "\n")
    (proof / "trajectories_raw.jsonl").write_text(traj_lines)
    (proof / "validation_report.jsonl").write_text(
        "\n".join(json.dumps({"index": i, "validation": {"passed": True}}) for i in range(rows)) + "\n"
    )
    (proof / "gpu_attestation.json").write_text(json.dumps({"passed": attested, "nonce": "n" * 64}))
    (proof / "novelty_report.json").write_text(json.dumps({"verified_rows": rows, "novel_verified_rows": rows}))
    (proof / "dataset_manifest.json").write_text(
        json.dumps(
            {
                "passed": gate_passed,
                "blocked_rows": 0 if gate_passed else 2,
                "rows_total": rows,
                "trajectories_sha256": sha,
            }
        )
    )

    if tamper_rows:
        (proof / "trajectories.jsonl").write_text(traj_lines + json.dumps({"prompt": "extra"}) + "\n")
    return proof, sha


def test_size_label_bands():
    assert size_label(200) == "dataset:xl"
    assert size_label(150) == "dataset:xl"
    assert size_label(100) == "dataset:l"
    assert size_label(75) == "dataset:m"
    assert size_label(50) == "dataset:s"
    assert size_label(25) == "dataset:xs"
    assert size_label(24) == "dataset:none"


def test_valid_proof_dir_passes(tmp_path):
    proof, sha = _write_proof_dir(tmp_path)
    issues, rows = check_proof_dir(proof, claimed_sha256=sha)
    assert issues == []
    assert rows == 3


def test_failed_attestation_rejects(tmp_path):
    proof, _ = _write_proof_dir(tmp_path, attested=False)
    report = verify_dataset_submission(proof)
    assert report["label"] == "dataset:REJECT"
    assert any("gpu_attestation" in issue for issue in report["issues"])


def test_failed_release_gate_rejects(tmp_path):
    proof, _ = _write_proof_dir(tmp_path, gate_passed=False)
    report = verify_dataset_submission(proof)
    assert report["label"] == "dataset:REJECT"


def test_tampered_rows_after_gate_rejects(tmp_path):
    proof, _ = _write_proof_dir(tmp_path, tamper_rows=True)
    report = verify_dataset_submission(proof)
    assert report["label"] == "dataset:REJECT"
    assert any("sha256" in issue for issue in report["issues"])


def test_claimed_sha_mismatch_rejects(tmp_path):
    proof, _ = _write_proof_dir(tmp_path)
    report = verify_dataset_submission(proof, claimed_sha256="deadbeef")
    assert report["label"] == "dataset:REJECT"
    assert any("claimed in the PR" in issue for issue in report["issues"])


def test_missing_sparkproof_root_rejects(tmp_path):
    proof, sha = _write_proof_dir(tmp_path)
    report = verify_dataset_submission(proof, claimed_sha256=sha, sparkproof_root=None)
    assert report["label"] == "dataset:REJECT"
    assert any("sparkproof-root is required" in issue for issue in report["issues"])


def test_missing_artifact_rejects(tmp_path):
    proof, _ = _write_proof_dir(tmp_path)
    (proof / "gpu_attestation.json").unlink()
    report = verify_dataset_submission(proof, sparkproof_root=None)
    assert report["label"] == "dataset:REJECT"
    assert any("missing proof artifact" in issue for issue in report["issues"])
