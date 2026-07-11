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
