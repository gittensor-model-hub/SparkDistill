"""Map upstream TritonBench difficulty / keywords onto harness levels and tags."""

from __future__ import annotations

import re
from typing import Any

CHANNEL_G = "G"
CHANNEL_T = "T"

LEVEL_DIRS: dict[int | str, str] = {
    1: "level1_basic",
    2: "level2_intermediate",
    3: "level3_advanced",
    4: "level4_expert",
    "bugfix": "bugfix",
}

# Upstream difficulty strings are "1".."5". Harness only has levels 1-4.
_DIFFICULTY_TO_LEVEL: dict[str, int] = {
    "1": 1,
    "2": 2,
    "3": 3,
    "4": 4,
    "5": 4,
}

# Keyword buckets used to refine tags and nudge level when upstream difficulty is missing.
TAG_KEYWORDS: dict[str, tuple[str, ...]] = {
    "attention": (
        "attention",
        "attn",
        "flash",
        "softmax_lse",
        "kv_cache",
        "cross_attn",
        "self_attn",
        "mha",
        "gqa",
        "mqa",
    ),
    "matmul": ("matmul", "gemm", "gemv", "bmm", "dot", "linear", "bgmv", "lora"),
    "softmax": ("softmax", "log_softmax", "logsoftmax"),
    "normalization": ("layernorm", "layer_norm", "rmsnorm", "rms_norm", "batch_norm", "group_norm"),
    "rope": ("rope", "rotary", "rbe", "pos_emb"),
    "quantization": (
        "quant",
        "dequant",
        "int4",
        "int8",
        "fp8",
        "fp4",
        "nf4",
        "awq",
        "gptq",
    ),
    "elementwise": (
        "relu",
        "gelu",
        "silu",
        "swiglu",
        "sigmoid",
        "tanh",
        "add",
        "mul",
        "div",
        "sub",
        "abs",
        "exp",
        "log",
        "sqrt",
        "vector_add",
        "elementwise",
    ),
    "reduction": ("reduce", "sum", "mean", "max", "min", "argmax", "cumsum", "scan"),
    "convolution": ("conv", "pool", "avg_pool", "max_pool", "im2col"),
    "dropout": ("dropout", "drop"),
    "embedding": ("embedding", "embed", "gather", "scatter"),
    "sparse": ("sparse", "block_sparse", "mask_mod"),
    "optimizer": ("adam", "sgd", "lion", "muon"),
    "fused": ("fused", "fusion"),
    "debug": ("bug", "fix", "incorrect", "wrong", "broken"),
}

_LEVEL_BOOST: dict[str, int] = {
    "attention": 1,
    "matmul": 1,
    "quantization": 1,
    "sparse": 1,
    "fused": 1,
    "convolution": 0,
    "rope": 0,
    "normalization": 0,
    "reduction": 0,
    "elementwise": -1,
    "dropout": -1,
}

_DEFAULT_REQUIRED = ["@triton.jit", "tl.load", "tl.store"]
_CATEGORY_REQUIRED: dict[str, list[str]] = {
    "kernel_generation": ["@triton.jit", "tl.load", "tl.store"],
    "kernel_translation": ["@triton.jit", "tl.load", "tl.store", "torch.allclose"],
    "kernel_debugging": ["@triton.jit", "mask"],
}


def level_dirname(level: int | str) -> str:
    """Return the problems/ subdirectory name for a harness level."""
    if level in LEVEL_DIRS:
        return LEVEL_DIRS[level]
    if isinstance(level, str) and level.isdigit() and int(level) in LEVEL_DIRS:
        return LEVEL_DIRS[int(level)]
    raise ValueError(f"unknown harness level: {level!r}")


def difficulty_to_level(difficulty: Any, *, tags: list[str] | None = None) -> int:
    """Map upstream difficulty (+ optional tag nudge) to harness level 1-4."""
    key = str(difficulty).strip() if difficulty is not None else ""
    base = _DIFFICULTY_TO_LEVEL.get(key)
    if base is None:
        try:
            base = max(1, min(4, int(float(key))))
        except (TypeError, ValueError):
            base = 2
    nudge = 0
    for tag in tags or []:
        nudge += _LEVEL_BOOST.get(tag, 0)
    if nudge > 0:
        base = min(4, base + 1)
    elif nudge < 0:
        base = max(1, base - 1)
    return base


def _haystack(*parts: Any) -> str:
    return " ".join(str(p) for p in parts if p).lower()


def infer_tags(*text_parts: Any) -> list[str]:
    """Infer topic tags from free-text instruction / filename fragments."""
    text = _haystack(*text_parts)
    tags: list[str] = []
    for tag, keywords in TAG_KEYWORDS.items():
        if any(kw in text for kw in keywords):
            tags.append(tag)
    # Stable order, unique.
    return list(dict.fromkeys(tags))


def infer_category(entry: dict[str, Any], channel: str) -> str:
    """Pick generation vs translation vs debugging from channel + payload."""
    blob = _haystack(
        entry.get("file"),
        entry.get("name"),
        entry.get("simp_instru"),
        entry.get("comp_instru"),
        entry.get("description"),
        entry.get("title"),
        entry.get("prompt"),
    )
    if "bug" in blob or "fix" in blob or "incorrect" in blob or "wrong_" in blob:
        return "kernel_debugging"
    if channel == CHANNEL_T or entry.get("torch_code") or entry.get("torch_cnt"):
        return "kernel_translation"
    return "kernel_generation"


def infer_required_patterns(instruction: str, category: str) -> list[str]:
    """Required substrings the evaluator checks for completeness."""
    patterns = list(_CATEGORY_REQUIRED.get(category, _DEFAULT_REQUIRED))
    low = (instruction or "").lower()
    if "mask" in low or "boundary" in low or "not a multiple" in low:
        if "mask" not in patterns:
            patterns.append("mask")
    if "autotune" in low or "block_size" in low or category != "kernel_debugging":
        if "@triton.autotune" not in patterns and category != "kernel_debugging":
            patterns.append("@triton.autotune")
    if "allclose" in low or "correctness" in low or category == "kernel_translation":
        if "torch.allclose" not in patterns:
            patterns.append("torch.allclose")
    if "tensor_descriptor" in low or "make_tensor_descriptor" in low:
        patterns.append("tl.make_tensor_descriptor")
    return list(dict.fromkeys(patterns))


def slugify_id(name: str) -> str:
    """Normalize a file / function name into a stable problem id."""
    stem = name.rsplit("/", 1)[-1]
    stem = stem.rsplit("\\", 1)[-1]
    if stem.endswith(".py"):
        stem = stem[:-3]
    stem = stem.strip().lower()
    stem = re.sub(r"[^a-z0-9]+", "_", stem)
    stem = re.sub(r"_+", "_", stem).strip("_")
    return stem or "unnamed"


def human_title(slug: str) -> str:
    """Turn a slug into a short title."""
    return slug.replace("_", " ").strip().title() or "Untitled Kernel"


def clamp_level(level: int) -> int:
    return max(1, min(4, int(level)))
