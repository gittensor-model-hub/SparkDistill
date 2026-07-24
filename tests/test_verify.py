from eval.verify import _no_student_endpoint_env, check_claim, check_training_claims, resolve_bundle_gpu_architecture


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
    manifest = {"train_hours": 3.0, "train_gpu": "NVIDIA A100"}
    issues = check_training_claims(manifest, None)
    assert any("accepted training GPU" in issue for issue in issues)


def test_training_claims_h100_pass():
    manifest = {"train_hours": 3.0, "train_gpu": "NVIDIA H100"}
    assert check_training_claims(manifest, None) == []


def test_training_claims_b200_pass():
    manifest = {"train_hours": 3.0, "train_gpu": "NVIDIA B200"}
    assert check_training_claims(manifest, None) == []


def test_training_claims_absent_fields_do_not_fail():
    # Legacy bundles without training claims fall back to full retrain-verification.
    assert check_training_claims({}, None) == []


def _signed(hwmodel: str) -> tuple[dict, dict]:
    """(attestation, gpu_sig) whose JWKS-verified device claim reports `hwmodel`.

    The attestation's own `claims` sidecar deliberately disagrees: only the signed
    `gpu_sig` claims may decide corroboration.
    """
    attestation = {"passed": True, "token": "<signed>", "claims": {"hwmodel": "SPOOFED B300"}}
    gpu_sig = {"verified": True, "claims": {"devices": {"GPU-0": {"hwmodel": hwmodel}}}}
    return attestation, gpu_sig


def test_training_claims_attestation_must_corroborate_gpu():
    manifest = {"train_hours": 3.0, "train_gpu": "NVIDIA RTX PRO 6000 Blackwell"}
    attestation, gpu_sig = _signed("GH100 A01 GSP BROM")
    issues = check_training_claims(manifest, attestation, gpu_sig=gpu_sig)
    assert any("corroborate" in issue for issue in issues)

    corroborating, corroborating_sig = _signed("GB202 RTX PRO 6000")
    assert check_training_claims(manifest, corroborating, gpu_sig=corroborating_sig) == []

    h100_manifest = {"train_hours": 3.0, "train_gpu": "NVIDIA H100"}
    h100_attestation, h100_sig = _signed("NVIDIA H100 SXM")
    assert check_training_claims(h100_manifest, h100_attestation, gpu_sig=h100_sig) == []

    # H200 is the same GH100 die as H100 (upgraded HBM3e only) — NVIDIA's hwmodel
    # attestation claim reports the die, not the memory SKU. Confirmed live on a
    # real H200 submission (gittensor-model-hub/SparkDistill#120): hwmodel="GH100",
    # which this must corroborate rather than reject.
    h200_manifest = {"train_hours": 4.2, "train_gpu": "NVIDIA H200"}
    h200_attestation, h200_sig = _signed("GH100")
    assert check_training_claims(h200_manifest, h200_attestation, gpu_sig=h200_sig) == []


def test_training_claims_ignore_the_editable_claims_sidecar():
    """A Hopper run must not corroborate a Blackwell train_gpu by editing attestation.json.

    `attestation["claims"]` is an unsigned convenience copy the miner commits in the
    PR, so hwmodel must be read from the JWKS-verified NRAS claims instead.
    """
    manifest = {"train_hours": 3.0, "train_gpu": "NVIDIA B300"}
    _, hopper_sig = _signed("GH100 A01 GSP BROM")

    # Sidecar rewritten to a Blackwell hwmodel.
    spoofed = {"passed": True, "token": "<signed>", "claims": {"devices": {"GPU-0": {"hwmodel": "B300 A01"}}}}
    assert any("corroborate" in i for i in check_training_claims(manifest, spoofed, gpu_sig=hopper_sig))

    # Sidecar deleted or emptied — must not short-circuit the check to a pass.
    for stripped in ({"passed": True, "token": "<signed>"}, {"passed": True, "token": "<signed>", "claims": {}}):
        assert any("corroborate" in i for i in check_training_claims(manifest, stripped, gpu_sig=hopper_sig))


def test_training_claims_reject_unverifiable_gpu_token():
    """No signed hwmodel evidence at all is fail-closed, not a pass-through."""
    manifest = {"train_hours": 3.0, "train_gpu": "NVIDIA B300"}
    attestation = {"passed": True, "claims": {"devices": {"GPU-0": {"hwmodel": "B300 A01"}}}}
    issues = check_training_claims(manifest, attestation, gpu_sig=None)
    assert any("unverified" in i for i in issues)


def test_proof_only_bundle_requires_local_checkpoint(tmp_path):
    import json

    from eval.canonical_dataset import canonical_hf_url
    from eval.verify import verify_submission

    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "manifest.json").write_text(
        json.dumps({"run_id": "r1", "dataset_url": canonical_hf_url()})
    )
    (bundle / "eval_scores.json").write_text(json.dumps({"scores": {"gsm8k": 0.6, "triton": 0.4}}))

    report = verify_submission(bundle, frontier={"gsm8k": 0.5, "triton": 0.3})
    assert report["verified"] is False
    assert report["reason"] == "checkpoint_required"


def test_claim_binding_matches_bound_nonce(tmp_path, monkeypatch):
    import json

    import jwt
    from cryptography.hazmat.primitives.asymmetric import ec

    from eval.verify import check_claim_binding
    from proof.bundle import claim_sha256

    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "manifest.json").write_text(json.dumps({"run_id": "r1"}))
    (bundle / "eval_scores.json").write_text(json.dumps({"scores": {"gsm8k": 0.6}}))
    digest = claim_sha256(bundle)

    key = ec.generate_private_key(ec.SECP384R1())

    def encode(payload):
        return jwt.encode(payload, key, algorithm="ES384", headers={"kid": "k"})

    def make_token(*, device_nonce: str | None = None, platform_nonce: str | None = None) -> str:
        platform = {"iss": "https://nras.attestation.nvidia.com", "sub": "platform"}
        if platform_nonce is not None:
            platform["eat_nonce"] = platform_nonce
        device = {"iss": "https://nras.attestation.nvidia.com", "hwmodel": "GH100"}
        if device_nonce is not None:
            device["eat_nonce"] = device_nonce
        return json.dumps(
            [
                ["JWT", jwt.encode({"sub": "overall"}, "k", algorithm="HS256")],
                {"REMOTE_GPU_CLAIMS": [["JWT", encode(platform)], {"GPU-0": encode(device)}]},
            ]
        )

    class FakeKey:
        def __init__(self, k):
            self.key = k.public_key()

    class FakeJWKClient:
        def __init__(self, url):
            pass

        def get_signing_key_from_jwt(self, encoded):
            return FakeKey(key)

    monkeypatch.setattr(jwt, "PyJWKClient", FakeJWKClient)

    # Editable JSON claims alone must never bind.
    assert check_claim_binding(bundle, {"passed": True, "claims": {"eat_nonce": digest}}) is False
    assert (
        check_claim_binding(
            bundle,
            {"passed": True, "token": make_token(), "claims": {"eat_nonce": digest}},
        )
        is False
    )

    bound = {"passed": True, "token": make_token(device_nonce=digest.upper())}
    platform_bound = {"passed": True, "token": make_token(platform_nonce=digest)}
    unbound = {"passed": True, "token": make_token(device_nonce="ab" * 32)}
    assert check_claim_binding(bundle, bound) is True
    assert check_claim_binding(bundle, platform_bound) is True
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
    from eval.canonical_dataset import canonical_hf_url

    bundle = tmp_path / "bundle"
    (bundle / "checkpoint").mkdir(parents=True)
    (bundle / "checkpoint" / "w.bin").write_text("w")
    (bundle / "manifest.json").write_text(
        json.dumps({"run_id": "r1", "dataset_url": canonical_hf_url()})
    )
    (bundle / "eval_scores.json").write_text(json.dumps({"scores": {"gsm8k": 0.6, "triton": 0.5}}))
    monkeypatch.setattr(v, "run_harness", lambda *a, **k: {"gsm8k": 0.6, "triton": 0.5})

    report = v.verify_submission(bundle, frontier=None)
    assert report["verified"] is True
    assert report["label"] == "eval:BASELINE"
    assert report["per_benchmark"]["gsm8k"] == {"candidate": 0.6, "frontier": None}

    # With a frontier, tier scoring uses triton only.
    scored = v.verify_submission(bundle, frontier={"gsm8k": 0.5, "triton": 0.4})
    assert scored["label"] == "eval:XL"


def test_tdx_binding_matches_report_data(tmp_path):
    import base64
    import json

    from eval.attestation import _TDX_REPORT_DATA_OFFSET, tdx_report_data
    from eval.verify import check_tdx_binding
    from proof.bundle import claim_sha256

    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "manifest.json").write_text(json.dumps({"run_id": "r1"}))
    (bundle / "eval_scores.json").write_text(json.dumps({"scores": {"gsm8k": 0.6}}))
    digest = claim_sha256(bundle)
    quote = b"\x00" * _TDX_REPORT_DATA_OFFSET + tdx_report_data(digest) + b"\x00" * 64
    quote_b64 = base64.b64encode(quote).decode()

    bound = {"passed": True, "tdx": {"report_data": tdx_report_data(digest).hex(), "quote_b64": quote_b64}}
    unbound = {"passed": True, "tdx": {"report_data": "ff" * 64, "quote_b64": quote_b64}}
    json_only = {"passed": True, "tdx": {"report_data": tdx_report_data(digest).hex()}}
    no_tdx = {"passed": True, "tdx": None}
    assert check_tdx_binding(bundle, bound) is True
    assert check_tdx_binding(bundle, unbound) is False
    assert check_tdx_binding(bundle, json_only) is False
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


def test_attested_gsm8k_skips_harness_without_checkpoint(tmp_path, monkeypatch):
    import json

    import eval.verify as v
    from eval.attestation import tdx_report_data
    from eval.canonical_dataset import canonical_hf_url
    from eval.regression_sample import REGRESSION_SAMPLE_FILENAME, build_regression_sample, load_regression_problems
    from proof.bundle import claim_sha256

    responses = [
        {"problem_id": int(row["problem_id"]), "model_response": f"#### {row['answer'].split('####')[-1].strip()}"}
        for row in load_regression_problems()
    ]
    sample = build_regression_sample(responses)

    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "manifest.json").write_text(
        json.dumps({"run_id": "r-attest", "dataset_url": canonical_hf_url()})
    )
    (bundle / "eval_scores.json").write_text(json.dumps({"scores": {"gsm8k": sample["exact_match"]}}))
    (bundle / REGRESSION_SAMPLE_FILENAME).write_text(json.dumps(sample, indent=2))

    digest = claim_sha256(bundle)
    # Integrity is stubbed; claim_bound still goes through check_claim_binding.
    monkeypatch.setattr(v, "check_attestation_integrity", lambda *a, **k: [])
    monkeypatch.setattr(v, "check_claim_binding", lambda *_a, **_k: True)
    monkeypatch.setattr(v, "check_tdx_binding", lambda *_a, **_k: True)
    monkeypatch.setattr(v, "check_gpu_signature", lambda *_a, **_k: {"verified": True, "claims": {}})
    monkeypatch.setattr(v, "check_tdx_signature", lambda *_a, **_k: {"verified": True, "status": "UpToDate"})

    attestation = {
        "passed": True,
        "token": "nras-token-placeholder",
        "claims": {"eat_nonce": digest},
        "tdx": {"report_data": tdx_report_data(digest).hex(), "quote_b64": "AAAA"},
    }

    def fail_harness(*_args, **_kwargs):
        raise AssertionError("run_harness should not run for attested gsm8k-only bundles")

    monkeypatch.setattr(v, "run_harness", fail_harness)

    report = v.verify_submission(bundle, frontier={"gsm8k": 0.5}, attestation=attestation)
    assert report["verified"] is True
    assert report["attested_gsm8k_regression"] is True
    assert report["claim_bound"] is True


def test_attested_gsm8k_sample_without_attestation_fails(tmp_path):
    import json

    from eval.canonical_dataset import canonical_hf_url
    from eval.regression_sample import REGRESSION_SAMPLE_FILENAME, build_regression_sample, load_regression_problems
    from eval.verify import verify_submission

    responses = [
        {"problem_id": int(row["problem_id"]), "model_response": f"#### {row['answer'].split('####')[-1].strip()}"}
        for row in load_regression_problems()
    ]
    sample = build_regression_sample(responses)

    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "manifest.json").write_text(
        json.dumps({"run_id": "r1", "dataset_url": canonical_hf_url()})
    )
    (bundle / "eval_scores.json").write_text(json.dumps({"scores": {"gsm8k": sample["exact_match"]}}))
    (bundle / REGRESSION_SAMPLE_FILENAME).write_text(json.dumps(sample, indent=2))

    report = verify_submission(bundle, frontier={"gsm8k": 0.5})
    assert report["verified"] is False
    assert report["reason"] == "attested_eval_samples_failed"


def test_resolve_bundle_gpu_architecture_prefers_explicit_field():
    assert resolve_bundle_gpu_architecture({"gpu_architecture": "hopper-h100", "train_gpu": "NVIDIA B200"}) == "hopper"


def test_resolve_bundle_gpu_architecture_falls_back_to_train_gpu():
    assert resolve_bundle_gpu_architecture({"train_gpu": "NVIDIA H200"}) == "hopper"


def test_resolve_bundle_gpu_architecture_defaults_to_blackwell():
    assert resolve_bundle_gpu_architecture({}) == "blackwell"
    assert resolve_bundle_gpu_architecture({"train_gpu": "NVIDIA A100"}) == "blackwell"


def test_verify_submission_hopper_tiers_on_triton(tmp_path, monkeypatch):
    import json

    import eval.verify as v
    from eval.canonical_dataset import canonical_hf_url

    bundle = tmp_path / "bundle"
    (bundle / "checkpoint").mkdir(parents=True)
    (bundle / "checkpoint" / "w.bin").write_text("w")
    (bundle / "manifest.json").write_text(
        json.dumps({"run_id": "r-hopper", "dataset_url": canonical_hf_url(), "train_gpu": "NVIDIA H100"})
    )
    (bundle / "eval_scores.json").write_text(json.dumps({"scores": {"gsm8k": 0.6, "triton": 0.5}}))
    monkeypatch.setattr(v, "run_harness", lambda *a, **k: {"gsm8k": 0.6, "triton": 0.5})

    report = v.verify_submission(bundle, frontier={"gsm8k": 0.5, "triton": 0.4})
    assert report["gpu_architecture"] == "hopper"
    assert report["best_benchmark"] == "triton"
    assert report["label"] == "eval:XL"
