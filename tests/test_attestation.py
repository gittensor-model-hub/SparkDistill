import json

import pytest

from eval.attestation import _decode_overall_claims
from eval.verify import check_training_claims

jwt = pytest.importorskip("jwt")


def _token(overall: dict, devices: dict[str, dict]) -> str:
    encode = lambda payload: jwt.encode(payload, "k", algorithm="HS256")  # noqa: E731
    return json.dumps(
        [
            ["JWT", encode(overall)],
            {"REMOTE_GPU_CLAIMS": [["JWT", encode({"sub": "platform"})], {k: encode(v) for k, v in devices.items()}]},
        ]
    )


def test_decode_includes_device_hardware_claims():
    token = _token(
        {"iss": "NRAS", "x-nvidia-overall-att-result": True},
        {"GPU-0": {"hwmodel": "GB20X", "x-nvidia-gpu-driver-version": "595.71.05"}},
    )
    claims = _decode_overall_claims(token)
    assert claims["iss"] == "NRAS"
    assert claims["devices"]["GPU-0"]["hwmodel"] == "GB20X"


def test_device_claims_corroborate_training_gpu():
    # The overall JWT has no hardware fields; without device submodule claims the
    # verify-side corroboration check wrongly rejected genuinely attested bundles.
    token = _token({"iss": "NRAS"}, {"GPU-0": {"hwmodel": "GB20X"}})
    attestation = {"passed": True, "claims": _decode_overall_claims(token)}
    manifest = {"train_hours": 0.1, "train_gpu": "NVIDIA RTX PRO 6000 Blackwell Server Edition"}
    assert check_training_claims(manifest, attestation) == []


def test_garbage_token_decodes_to_empty():
    assert _decode_overall_claims("not json") == {}


def test_tdx_report_data_pads_digest():
    from eval.attestation import tdx_report_data

    digest = "ab" * 32
    data = tdx_report_data(digest)
    assert len(data) == 64
    assert data[:32] == bytes.fromhex(digest)
    assert data[32:] == b"\x00" * 32


def test_tdx_quote_via_provisioned_node(tmp_path):
    from eval.attestation import _TDX_REPORT_DATA_OFFSET, tdx_quote, tdx_report_data

    digest = "cd" * 32
    node = tmp_path / "report"
    node.mkdir()
    (node / "provider").write_text("tdx_guest\n")
    # Emulate the kernel: outblob holds a quote embedding the report data at the
    # v4 offset (in reality it is regenerated on every inblob write).
    fake_quote = b"\x00" * _TDX_REPORT_DATA_OFFSET + tdx_report_data(digest) + b"\x00" * 128
    (node / "outblob").write_bytes(fake_quote)

    result = tdx_quote(digest, report_path=node)
    assert result is not None
    assert result["provider"] == "tdx_guest"
    assert result["report_data"] == tdx_report_data(digest).hex()
    assert (node / "inblob").read_bytes() == tdx_report_data(digest)


def test_tdx_quote_absent_on_non_tdx_host(tmp_path):
    from eval.attestation import tdx_quote

    # mkdir fails inside a nonexistent parent -> None, never raises.
    assert tdx_quote("ab" * 32, report_path=tmp_path / "no" / "tsm" / "node") is None


def test_verify_tdx_quote_reports_missing_library(monkeypatch):
    import builtins
    import sys

    from eval.attestation import verify_tdx_quote

    monkeypatch.setitem(sys.modules, "dcap_qvl", None)
    real_import = builtins.__import__

    def no_dcap(name, *args, **kwargs):
        if name == "dcap_qvl":
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.delitem(sys.modules, "dcap_qvl")
    monkeypatch.setattr(builtins, "__import__", no_dcap)
    result = verify_tdx_quote("AAAA")
    assert result["verified"] is False
    assert "not installed" in result["status"]


def test_verify_tdx_quote_maps_status(monkeypatch):
    import sys
    import types

    from eval.attestation import verify_tdx_quote

    class Report:
        def __init__(self, status, advisories):
            self.status = status
            self.advisory_ids = advisories

    fake = types.ModuleType("dcap_qvl")

    async def fake_verify(quote, pccs_url=None):
        return Report("UpToDate", [])

    fake.get_collateral_and_verify = fake_verify
    monkeypatch.setitem(sys.modules, "dcap_qvl", fake)
    result = verify_tdx_quote("AAAA")
    assert result == {"verified": True, "status": "UpToDate", "advisory_ids": []}

    async def stale_verify(quote, pccs_url=None):
        return Report("OutOfDate", ["INTEL-SA-00837"])

    fake.get_collateral_and_verify = stale_verify
    result = verify_tdx_quote("AAAA")
    assert result["verified"] is False
    assert result["status"] == "OutOfDate"
    assert result["advisory_ids"] == ["INTEL-SA-00837"]


def _es384_token_fixture():
    from cryptography.hazmat.primitives.asymmetric import ec

    key = ec.generate_private_key(ec.SECP384R1())
    encode = lambda payload: jwt.encode(  # noqa: E731
        payload, key, algorithm="ES384", headers={"kid": "nv-eat-kid-test"}
    )
    token = json.dumps(
        [
            ["JWT", jwt.encode({"sub": "overall"}, "k", algorithm="HS256")],
            {
                "REMOTE_GPU_CLAIMS": [
                    ["JWT", encode({"iss": "https://nras.attestation.nvidia.com", "sub": "platform"})],
                    {"GPU-0": encode({"iss": "https://nras.attestation.nvidia.com", "hwmodel": "GB20X"})},
                ]
            },
        ]
    )
    return key, token


def test_verify_gpu_token_accepts_valid_signatures(monkeypatch):
    from eval.attestation import verify_gpu_token

    key, token = _es384_token_fixture()

    class FakeKey:
        def __init__(self, k):
            self.key = k.public_key()

    class FakeJWKClient:
        def __init__(self, url):
            pass

        def get_signing_key_from_jwt(self, encoded):
            return FakeKey(key)

    monkeypatch.setattr(jwt, "PyJWKClient", FakeJWKClient)
    result = verify_gpu_token(token)
    assert result["verified"] is True
    assert result["tokens_checked"] == 2
    assert result["issues"] == []


def test_verify_gpu_token_rejects_wrong_key(monkeypatch):
    from cryptography.hazmat.primitives.asymmetric import ec

    from eval.attestation import verify_gpu_token

    _, token = _es384_token_fixture()
    other = ec.generate_private_key(ec.SECP384R1())

    class FakeKey:
        key = other.public_key()

    class FakeJWKClient:
        def __init__(self, url):
            pass

        def get_signing_key_from_jwt(self, encoded):
            return FakeKey()

    monkeypatch.setattr(jwt, "PyJWKClient", FakeJWKClient)
    result = verify_gpu_token(token)
    assert result["verified"] is False
    assert result["tokens_checked"] == 0
    assert len(result["issues"]) == 2


def test_verify_gpu_token_garbage_is_unverified():
    from eval.attestation import verify_gpu_token

    result = verify_gpu_token("not json")
    assert result["verified"] is False
