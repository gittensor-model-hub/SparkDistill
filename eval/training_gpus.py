"""Accepted training-track GPU claims for proof bundles."""

from __future__ import annotations

from typing import Any

# Substrings matched case-insensitively against manifest train_gpu claims.
ACCEPTED_TRAINING_GPU_SUBSTRINGS = (
    # Blackwell workstation (RTX PRO 6000, etc.)
    "rtx pro 6000",
    "gb20",
    # Datacenter Blackwell
    "b200",
    "b300",
    "gb200",
    # Hopper training nodes
    "h100",
    "h200",
    "gh100",
    "gh200",
)


def is_accepted_training_gpu(train_gpu: str | None) -> bool:
    if not train_gpu:
        return False
    blob = str(train_gpu).lower()
    return any(pattern in blob for pattern in ACCEPTED_TRAINING_GPU_SUBSTRINGS)


def accepted_training_gpu_label() -> str:
    return "Blackwell (RTX PRO 6000 / B200 / B300) or Hopper (H100 / H200) CC node"


def _hwmodel_claim_values(claims: dict[str, Any] | None) -> list[str]:
    """Return only hardware-model claim strings (never the whole claims blob).

    NRAS per-device submodule tokens carry `hwmodel` under `claims.devices`
    (see `eval.attestation._decode_overall_claims`). A top-level `hwmodel` is
    also accepted for tests / simplified fixtures.
    """
    if not isinstance(claims, dict) or not claims:
        return []

    values: list[str] = []
    top = claims.get("hwmodel")
    if isinstance(top, str) and top.strip():
        values.append(top)

    devices = claims.get("devices")
    if isinstance(devices, dict):
        for device_claims in devices.values():
            if not isinstance(device_claims, dict):
                continue
            hwmodel = device_claims.get("hwmodel")
            if isinstance(hwmodel, str) and hwmodel.strip():
                values.append(hwmodel)
    return values


def attestation_corroborates_training_gpu(train_gpu: str, attestation: dict | None) -> bool:
    """When attestation is present, hwmodel claims must match the declared train_gpu family.

    Matches **only** `hwmodel` fields — never the flattened claims JSON blob.
    A grindable `eat_nonce` / `claim_sha256` that happens to contain `b200` etc.
    must not corroborate a Blackwell declaration against Hopper hardware (#148).
    """
    if not attestation:
        return True
    claims = attestation.get("claims") or {}
    if not isinstance(claims, dict) or claims == {}:
        return True

    hwmodels = _hwmodel_claim_values(claims)
    if not hwmodels:
        # No hardware identity to check — fail closed so a nonce/claims grind
        # cannot stand in for hwmodel (unlike an empty claims dict above, which
        # preserves the legacy "no claims decoded" pass-through).
        return False

    claims_blob = " ".join(hwmodels).lower()
    gpu = str(train_gpu).lower()
    if any(token in gpu for token in ("h100", "gh100")):
        return "h100" in claims_blob or "gh100" in claims_blob
    if any(token in gpu for token in ("h200", "gh200")):
        # H200 is the same GH100 die as H100 (upgraded HBM3e only) — NVIDIA's
        # hwmodel attestation claim reports the die, not the memory SKU, so a
        # genuine H200 node attests as "GH100" too. Confirmed live: a real H200
        # training submission's attestation reported hwmodel="GH100" and was
        # wrongly rejected here before this fix (gittensor-model-hub/SparkDistill#120).
        return any(token in claims_blob for token in ("h200", "gh200", "gh100"))
    if any(token in gpu for token in ("b200", "b300", "gb200")):
        return any(token in claims_blob for token in ("b200", "b300", "gb200", "gb102"))
    if "pro 6000" in gpu or "gb20" in gpu:
        return "pro 6000" in claims_blob or "gb20" in claims_blob
    return is_accepted_training_gpu(train_gpu)
