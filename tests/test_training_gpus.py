"""Tests for eval.training_gpus."""

from eval.training_gpus import attestation_corroborates_training_gpu, is_accepted_training_gpu


def test_accepted_training_gpus():
    assert is_accepted_training_gpu("NVIDIA RTX PRO 6000 Blackwell Server Edition")
    assert is_accepted_training_gpu("NVIDIA B200")
    assert is_accepted_training_gpu("NVIDIA B300")
    assert is_accepted_training_gpu("NVIDIA H100 SXM")
    assert is_accepted_training_gpu("NVIDIA H200")
    assert not is_accepted_training_gpu("NVIDIA A100")


def test_attestation_corroboration_by_family():
    assert attestation_corroborates_training_gpu(
        "NVIDIA H100",
        {"passed": True, "claims": {"hwmodel": "NVIDIA GH100"}},
    )
    assert not attestation_corroborates_training_gpu(
        "NVIDIA H100",
        {"passed": True, "claims": {"hwmodel": "GB202 RTX PRO 6000"}},
    )


def test_attestation_reads_per_device_hwmodel():
    """Real NRAS tokens put hwmodel under claims.devices, not the overall JWT."""
    assert attestation_corroborates_training_gpu(
        "NVIDIA H200",
        {
            "passed": True,
            "claims": {
                "devices": {"GPU-0": {"hwmodel": "GH100 A01 GSP BROM"}},
            },
        },
    )
    assert attestation_corroborates_training_gpu(
        "NVIDIA RTX PRO 6000 Blackwell",
        {
            "passed": True,
            "claims": {
                "devices": {"GPU-0": {"hwmodel": "GB202 RTX PRO 6000"}},
            },
        },
    )


def test_attestation_ignores_grindable_nonce_containing_gpu_tokens():
    """A claim_sha256 / eat_nonce containing 'b200' must not corroborate hardware (#148)."""
    att = {
        "passed": True,
        "claims": {
            "eat_nonce": "87acb1e183a2b0f74c3b2008b8ef6975a95269bc490a8886f317fa4bd714b085",
            "devices": {"GPU-0": {"hwmodel": "GH100 A01 GSP BROM"}},
        },
    }
    assert attestation_corroborates_training_gpu("NVIDIA B200", att) is False
    assert attestation_corroborates_training_gpu("NVIDIA H100", att) is True


def test_attestation_empty_claims_still_pass_through():
    assert attestation_corroborates_training_gpu(
        "NVIDIA B200",
        {"passed": True, "claims": {}},
    )


def test_attestation_without_hwmodel_fails_closed():
    assert attestation_corroborates_training_gpu(
        "NVIDIA B200",
        {"passed": True, "claims": {"eat_nonce": "deadbeefb200cafe"}},
    ) is False
