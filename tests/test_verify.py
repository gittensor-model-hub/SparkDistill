from eval.verify import _no_student_endpoint_env, check_claim, check_training_claims


def test_check_claim_within_tolerance_has_no_mismatch():
    claimed = {"gsm8k": 0.88, "humaneval": 0.80}
    rerun = {"gsm8k": 0.885, "humaneval": 0.795}
    assert check_claim(claimed, rerun, tolerance_pct=2.0) == []


def test_check_claim_triton_compares_against_quick_subset():
    # A full-run composite (levels 1-4) legitimately differs from a level-1-only
    # re-run; the claim's triton_quick (same subset as the re-run) is the fair bar.
    claimed = {"triton": 0.55, "triton_quick": 0.82}
    rerun = {"triton": 0.815}
    assert check_claim(claimed, rerun, tolerance_pct=2.0) == []
    # And a fabricated quick-subset claim still gets caught.
    assert check_claim({"triton": 0.55, "triton_quick": 0.95}, rerun, tolerance_pct=2.0) == ["triton"]


def test_check_claim_triton_uses_widened_benchmark_tolerance():
    # Observed live: honest cross-server drift of 2.1pp on the 3-problem quick
    # set — within triton's claim_tolerance_pct (5.0), beyond the 2.0 default.
    claimed = {"triton_quick": 0.4278, "triton": 0.4278}
    rerun = {"triton": 0.4489}
    assert check_claim(claimed, rerun, tolerance_pct=2.0) == []
    # gsm8k keeps the tight default.
    assert check_claim({"gsm8k": 0.60}, {"gsm8k": 0.631}, tolerance_pct=2.0) == ["gsm8k"]


def test_check_claim_triton_falls_back_to_headline_without_quick():
    claimed = {"triton": 0.815}
    rerun = {"triton": 0.82}
    assert check_claim(claimed, rerun, tolerance_pct=2.0) == []


def test_no_student_endpoint_env_hides_and_restores(monkeypatch):
    import os

    monkeypatch.setenv("SPARKDISTILL_STUDENT_ENDPOINT", "http://stale:8000/v1")
    with _no_student_endpoint_env():
        assert "SPARKDISTILL_STUDENT_ENDPOINT" not in os.environ
    assert os.environ["SPARKDISTILL_STUDENT_ENDPOINT"] == "http://stale:8000/v1"


def test_check_claim_beyond_tolerance_flags_mismatch():
    claimed = {"gsm8k": 0.88, "humaneval": 0.80}
    rerun = {"gsm8k": 0.70, "humaneval": 0.795}
    assert check_claim(claimed, rerun, tolerance_pct=2.0) == ["gsm8k"]


def test_check_claim_rejects_percentage_unit_scores():
    # The `* 100.0` pp conversion assumes fractions; a 0-100 percentage would make the
    # tolerance 100x too tight and reject honest submissions (issue #72). Fail loudly.
    import pytest

    with pytest.raises(ValueError, match=r"fractions in \[0, 1\]"):
        check_claim({"gsm8k": 88.0}, {"gsm8k": 86.0}, tolerance_pct=2.0)


def test_check_claim_ignores_benchmarks_not_claimed():
    claimed = {"gsm8k": 0.88}
    rerun = {"gsm8k": 0.88, "humaneval": 0.10}
    assert check_claim(claimed, rerun, tolerance_pct=2.0) == []


def test_training_claims_within_budget_pass():
    manifest = {"train_hours": 4.5, "train_gpu": "NVIDIA RTX PRO 6000 Blackwell Server Edition"}
    assert check_training_claims(manifest, None) == []


def test_training_claims_over_budget_fail():
    manifest = {"train_hours": 6.0, "train_gpu": "NVIDIA RTX PRO 6000 Blackwell"}
    issues = check_training_claims(manifest, None)
    assert any("budget" in issue for issue in issues)


def test_training_claims_wrong_gpu_fail():
    manifest = {"train_hours": 3.0, "train_gpu": "NVIDIA H100"}
    issues = check_training_claims(manifest, None)
    assert any("RTX PRO 6000" in issue for issue in issues)


def test_training_claims_absent_fields_do_not_fail():
    # Legacy bundles without training claims fall back to full retrain-verification.
    assert check_training_claims({}, None) == []


def test_training_claims_attestation_must_corroborate_gpu():
    manifest = {"train_hours": 3.0, "train_gpu": "NVIDIA RTX PRO 6000 Blackwell"}
    attestation = {"passed": True, "claims": {"hwmodel": "GH100 A01 GSP BROM"}}
    issues = check_training_claims(manifest, attestation)
    assert any("corroborate" in issue for issue in issues)

    corroborating = {"passed": True, "claims": {"hwmodel": "GB202 RTX PRO 6000"}}
    assert check_training_claims(manifest, corroborating) == []


def test_proof_only_bundle_requires_local_checkpoint(tmp_path):
    import json

    from eval.verify import verify_submission

    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "manifest.json").write_text(json.dumps({"run_id": "r1"}))
    (bundle / "eval_scores.json").write_text(json.dumps({"scores": {"gsm8k": 0.6}}))

    report = verify_submission(bundle, frontier={"gsm8k": 0.5})
    assert report["verified"] is False
    assert report["reason"] == "checkpoint_required"


def test_claim_binding_matches_bound_nonce(tmp_path):
    import json

    from eval.verify import check_claim_binding
    from proof.bundle import claim_sha256

    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "manifest.json").write_text(json.dumps({"run_id": "r1"}))
    (bundle / "eval_scores.json").write_text(json.dumps({"scores": {"gsm8k": 0.6}}))
    digest = claim_sha256(bundle)

    bound = {"passed": True, "claims": {"eat_nonce": digest.upper()}}
    unbound = {"passed": True, "claims": {"eat_nonce": "ab" * 32}}
    # NRAS v3 puts the nonce in the per-device submodule tokens, not the overall JWT.
    device_bound = {"passed": True, "claims": {"devices": {"GPU-0": {"eat_nonce": digest}}}}
    assert check_claim_binding(bundle, bound) is True
    assert check_claim_binding(bundle, device_bound) is True
    assert check_claim_binding(bundle, unbound) is False
    assert check_claim_binding(bundle, None) is None


def test_checkpoint_manifest_match_and_mismatch(tmp_path):
    from eval.verify import check_checkpoint_manifest
    from proof.bundle import checkpoint_manifest

    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    (ckpt / "w.bin").write_text("weights")
    manifest = {"checkpoint_manifest": checkpoint_manifest(ckpt)}
    assert check_checkpoint_manifest(manifest, ckpt) is True
    (ckpt / "w.bin").write_text("tampered")
    assert check_checkpoint_manifest(manifest, ckpt) is False
    assert check_checkpoint_manifest({}, ckpt) is None


def test_no_frontier_yields_baseline_label(tmp_path, monkeypatch):
    import json

    import eval.verify as v

    bundle = tmp_path / "bundle"
    (bundle / "checkpoint").mkdir(parents=True)
    (bundle / "checkpoint" / "w.bin").write_text("w")
    (bundle / "manifest.json").write_text(json.dumps({"run_id": "r1"}))
    (bundle / "eval_scores.json").write_text(json.dumps({"scores": {"gsm8k": 0.6}}))
    monkeypatch.setattr(v, "run_harness", lambda *a, **k: {"gsm8k": 0.6})

    report = v.verify_submission(bundle, frontier=None)
    assert report["verified"] is True
    assert report["label"] == "eval:BASELINE"
    assert report["per_benchmark"]["gsm8k"] == {"candidate": 0.6, "frontier": None}

    # With a frontier, normal tier scoring applies unchanged.
    scored = v.verify_submission(bundle, frontier={"gsm8k": 0.5})
    assert scored["label"] == "eval:XL"


def test_tdx_binding_matches_report_data(tmp_path):
    import json

    from eval.attestation import tdx_report_data
    from eval.verify import check_tdx_binding
    from proof.bundle import claim_sha256

    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "manifest.json").write_text(json.dumps({"run_id": "r1"}))
    (bundle / "eval_scores.json").write_text(json.dumps({"scores": {"gsm8k": 0.6}}))
    digest = claim_sha256(bundle)

    bound = {"passed": True, "tdx": {"report_data": tdx_report_data(digest).hex()}}
    unbound = {"passed": True, "tdx": {"report_data": "ff" * 64}}
    no_tdx = {"passed": True, "tdx": None}
    assert check_tdx_binding(bundle, bound) is True
    assert check_tdx_binding(bundle, unbound) is False
    assert check_tdx_binding(bundle, no_tdx) is None
    assert check_tdx_binding(bundle, None) is None


def test_tdx_signature_absent_without_quote():
    from eval.verify import check_tdx_signature

    assert check_tdx_signature(None) is None
    assert check_tdx_signature({"passed": True, "tdx": None}) is None


def test_gpu_signature_absent_without_token():
    from eval.verify import check_gpu_signature

    assert check_gpu_signature(None) is None
    assert check_gpu_signature({"passed": True, "token": ""}) is None
