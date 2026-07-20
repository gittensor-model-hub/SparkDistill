"""CI-level GPU + TDX attestation integrity checks (no GPU required).

Covers the three fail-closed signals training CI uses for proof-only bundles:
1. GPU NRAS token vs NVIDIA JWKS (`check_gpu_signature` / `verify_gpu_token`)
2. TDX REPORTDATA binding to claim_sha256 (`check_tdx_binding`)
3. TDX quote DCAP/PCS verification (`check_tdx_signature` / `verify_tdx_quote`)
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

jwt = pytest.importorskip("jwt")


def _es384_token_fixture():
    from cryptography.hazmat.primitives.asymmetric import ec

    key = ec.generate_private_key(ec.SECP384R1())
    encode = lambda payload: jwt.encode(  # noqa: E731
        payload, key, algorithm="ES384", headers={"kid": "nv-eat-kid-ci-test"}
    )
    token = json.dumps(
        [
            ["JWT", jwt.encode({"sub": "overall"}, "k", algorithm="HS256")],
            {
                "REMOTE_GPU_CLAIMS": [
                    ["JWT", encode({"iss": "https://nras.attestation.nvidia.com", "sub": "platform"})],
                    {
                        "GPU-0": encode(
                            {
                                "iss": "https://nras.attestation.nvidia.com",
                                "hwmodel": "GH100 A01 GSP BROM",
                            }
                        )
                    },
                ]
            },
        ]
    )
    return key, token


def _patch_jwks(monkeypatch, key):
    class FakeKey:
        def __init__(self, k):
            self.key = k.public_key()

    class FakeJWKClient:
        def __init__(self, url):
            pass

        def get_signing_key_from_jwt(self, encoded):
            return FakeKey(key)

    monkeypatch.setattr(jwt, "PyJWKClient", FakeJWKClient)


def _bundle(tmp_path: Path, scores: dict | None = None) -> Path:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "manifest.json").write_text(json.dumps({"run_id": "ci-attest-1"}), encoding="utf-8")
    (bundle / "eval_scores.json").write_text(
        json.dumps({"scores": scores or {"gsm8k": 0.6, "triton": 0.5}}),
        encoding="utf-8",
    )
    return bundle


# --- 1. GPU attestation (JWKS) -------------------------------------------------


def test_ci_gpu_attestation_jwks_accepts_valid_token(monkeypatch):
    from eval.verify import check_gpu_signature

    key, token = _es384_token_fixture()
    _patch_jwks(monkeypatch, key)
    result = check_gpu_signature({"passed": True, "token": token})
    assert result is not None
    assert result["verified"] is True
    assert result["tokens_checked"] == 2


def test_ci_gpu_attestation_jwks_rejects_forged_token(monkeypatch):
    from cryptography.hazmat.primitives.asymmetric import ec

    from eval.verify import check_gpu_signature

    _, token = _es384_token_fixture()
    other = ec.generate_private_key(ec.SECP384R1())
    _patch_jwks(monkeypatch, other)
    result = check_gpu_signature({"passed": True, "token": token})
    assert result is not None
    assert result["verified"] is False


def test_ci_gpu_attestation_missing_token_is_none():
    from eval.verify import check_gpu_signature

    assert check_gpu_signature({"passed": True}) is None
    assert check_gpu_signature({"passed": True, "token": ""}) is None


# --- 2a. TDX binding (REPORTDATA ↔ claim_sha256) ------------------------------


def test_ci_tdx_binding_accepts_matching_report_data(tmp_path):
    from eval.attestation import tdx_report_data
    from eval.verify import check_tdx_binding
    from proof.bundle import claim_sha256

    bundle = _bundle(tmp_path)
    digest = claim_sha256(bundle)
    att = {
        "passed": True,
        "tdx": {"report_data": tdx_report_data(digest).hex(), "quote_b64": "AAAA"},
    }
    assert check_tdx_binding(bundle, att) is True


def test_ci_tdx_binding_rejects_mismatched_report_data(tmp_path):
    from eval.verify import check_tdx_binding

    bundle = _bundle(tmp_path)
    att = {"passed": True, "tdx": {"report_data": "00" * 64, "quote_b64": "AAAA"}}
    assert check_tdx_binding(bundle, att) is False


# --- 2b. TDX signature (DCAP / Intel PCS) -------------------------------------


def test_ci_tdx_signature_accepts_up_to_date_quote(monkeypatch):
    import sys
    import types

    from eval.verify import check_tdx_signature

    class Report:
        status = "UpToDate"
        advisory_ids: list[str] = []

    fake = types.ModuleType("dcap_qvl")

    async def ok(quote, pccs_url=None):
        return Report()

    fake.get_collateral_and_verify = ok
    monkeypatch.setitem(sys.modules, "dcap_qvl", fake)

    result = check_tdx_signature({"passed": True, "tdx": {"quote_b64": "AAAA", "report_data": "ab" * 32}})
    assert result is not None
    assert result["verified"] is True
    assert result["status"] == "UpToDate"


def test_ci_tdx_signature_rejects_stale_quote(monkeypatch):
    import sys
    import types

    from eval.verify import check_tdx_signature

    class Report:
        status = "OutOfDate"
        advisory_ids = ["INTEL-SA-00837"]

    fake = types.ModuleType("dcap_qvl")

    async def stale(quote, pccs_url=None):
        return Report()

    fake.get_collateral_and_verify = stale
    monkeypatch.setitem(sys.modules, "dcap_qvl", fake)

    result = check_tdx_signature({"passed": True, "tdx": {"quote_b64": "AAAA", "report_data": "ab" * 32}})
    assert result is not None
    assert result["verified"] is False
    assert result["status"] == "OutOfDate"


# --- Integrity gate (GPU + both TDX checks) -----------------------------------


def test_check_attestation_integrity_requires_gpu_jwks_and_claim_binding(tmp_path, monkeypatch):
    from eval.verify import check_attestation_integrity
    from proof.bundle import claim_sha256

    bundle = _bundle(tmp_path)
    key, token = _es384_token_fixture()
    _patch_jwks(monkeypatch, key)
    digest = claim_sha256(bundle)

    # Forged passed:true with no token / wrong nonce.
    assert any("JWKS" in i or "token" in i for i in check_attestation_integrity(bundle, {"passed": True}))
    assert any(
        "eat_nonce" in i
        for i in check_attestation_integrity(
            bundle, {"passed": True, "token": token, "claims": {"eat_nonce": "deadbeef"}}
        )
    )

    ok = check_attestation_integrity(
        bundle,
        {"passed": True, "token": token, "claims": {"devices": {"GPU-0": {"eat_nonce": digest}}}},
    )
    assert ok == []


def test_check_attestation_integrity_requires_both_tdx_checks_when_quote_present(tmp_path, monkeypatch):
    import sys
    import types

    from eval.attestation import tdx_report_data
    from eval.verify import check_attestation_integrity
    from proof.bundle import claim_sha256

    bundle = _bundle(tmp_path)
    key, token = _es384_token_fixture()
    _patch_jwks(monkeypatch, key)
    digest = claim_sha256(bundle)

    class Report:
        status = "UpToDate"
        advisory_ids: list[str] = []

    fake = types.ModuleType("dcap_qvl")

    async def ok(quote, pccs_url=None):
        return Report()

    fake.get_collateral_and_verify = ok
    monkeypatch.setitem(sys.modules, "dcap_qvl", fake)

    # Wrong REPORTDATA → fail TDX binding even if DCAP would pass.
    bad_bind = check_attestation_integrity(
        bundle,
        {
            "passed": True,
            "token": token,
            "claims": {"eat_nonce": digest},
            "tdx": {"report_data": "11" * 64, "quote_b64": "AAAA"},
        },
    )
    assert any("REPORTDATA" in i for i in bad_bind)

    good = check_attestation_integrity(
        bundle,
        {
            "passed": True,
            "token": token,
            "claims": {"eat_nonce": digest},
            "tdx": {"report_data": tdx_report_data(digest).hex(), "quote_b64": "AAAA"},
        },
    )
    assert good == []


def test_check_attestation_integrity_require_tdx_for_attested_samples_path(tmp_path, monkeypatch):
    from eval.verify import check_attestation_integrity
    from proof.bundle import claim_sha256

    bundle = _bundle(tmp_path)
    key, token = _es384_token_fixture()
    _patch_jwks(monkeypatch, key)
    digest = claim_sha256(bundle)
    att = {"passed": True, "token": token, "claims": {"eat_nonce": digest}}
    assert check_attestation_integrity(bundle, att, require_tdx=False) == []
    assert any("TDX quote is required" in i for i in check_attestation_integrity(bundle, att, require_tdx=True))


def test_verify_submission_rejects_forged_passed_true_attestation(tmp_path):
    """CI must not accept {"passed": true} without NRAS/JWKS evidence."""
    from eval.canonical_dataset import canonical_hf_url
    from eval.verify import verify_submission

    bundle = _bundle(tmp_path)
    (bundle / "manifest.json").write_text(
        json.dumps({"run_id": "r1", "dataset_url": canonical_hf_url()}),
        encoding="utf-8",
    )
    report = verify_submission(
        bundle,
        frontier={"gsm8k": 0.5, "triton": 0.4},
        attestation={"passed": True},
    )
    assert report["verified"] is False
    assert report["reason"] == "attestation_integrity_failed"
    assert report["label"] == "eval:REJECT"
    assert any("token" in i or "JWKS" in i for i in report["issues"])
