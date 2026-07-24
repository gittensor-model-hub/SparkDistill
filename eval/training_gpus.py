"""Accepted training-track GPU claims for proof bundles."""

from __future__ import annotations

from eval.gpu_architecture import GpuArchitecture, normalize_gpu_architecture

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


def claimed_hwmodels(claims: dict) -> list[str]:
    """Lowercased `hwmodel` values carried by an attestation's claims.

    Hardware identity lives on the per-device submodule claims that
    `eval.attestation._decode_overall_claims` collects under `devices` — the
    overall EAT has no hardware fields. A top-level `hwmodel` is read too, so
    flattened or hand-rolled claim shapes still corroborate.
    """
    models: list[str] = []
    for source in (claims, *(claims.get("devices") or {}).values()):
        if not isinstance(source, dict):
            continue
        hwmodel = source.get("hwmodel")
        if hwmodel:
            models.append(str(hwmodel).lower())
    return models


def attested_gpu_architectures(claims: dict) -> set[GpuArchitecture]:
    """Architecture families evidenced by an attestation's `hwmodel` claims.

    Only `hwmodel` is read, for the same reason `attestation_corroborates_training_gpu`
    restricts itself to it: other claim fields are hex-shaped and miner-influenced, so
    matching the whole blob would let a nonce stand in for hardware evidence (#148).
    """
    architectures: set[GpuArchitecture] = set()
    for model in claimed_hwmodels(claims):
        architecture = normalize_gpu_architecture(model)
        if architecture is not None:
            architectures.add(architecture)
    return architectures


def attestation_corroborates_training_gpu(train_gpu: str, attestation: dict | None) -> bool:
    """When attestation is present, hwmodel claims must match the declared train_gpu family.

    Only `hwmodel` claims are considered. Matching against the whole claims blob
    would let unrelated hex fields stand in for hardware evidence: `b200` and
    `b300` are themselves valid hex, so a `claim_sha256` nonce or a measurement
    digest can contain them by chance, and the nonce is miner-controlled.
    """
    if not attestation:
        return True
    claims = attestation.get("claims")
    if not claims:
        return True
    models = claimed_hwmodels(claims) if isinstance(claims, dict) else []
    if isinstance(claims, dict) and claims and not models:
        # No hardware identity to check — fail closed so a nonce/claims grind
        # cannot stand in for hwmodel (unlike an empty claims dict above, which
        # preserves the legacy "no claims decoded" pass-through).
        return False

    def attests(*tokens: str) -> bool:
        return any(token in model for model in models for token in tokens)

    gpu = str(train_gpu).lower()
    if any(token in gpu for token in ("h100", "gh100")):
        return attests("h100", "gh100")
    if any(token in gpu for token in ("h200", "gh200")):
        # H200 is the same GH100 die as H100 (upgraded HBM3e only) — NVIDIA's
        # hwmodel attestation claim reports the die, not the memory SKU, so a
        # genuine H200 node attests as "GH100" too. Confirmed live: a real H200
        # training submission's attestation reported hwmodel="GH100" and was
        # wrongly rejected here before this fix (gittensor-model-hub/SparkDistill#120).
        return attests("h200", "gh200", "gh100")
    if any(token in gpu for token in ("b200", "b300", "gb200")):
        return attests("b200", "b300", "gb200", "gb102")
    if "pro 6000" in gpu or "gb20" in gpu:
        return attests("pro 6000", "gb20")
    return False
