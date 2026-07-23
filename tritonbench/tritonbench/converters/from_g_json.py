"""Convert TritonBench-G JSON entries into YAML harness problems."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tritonbench.converters.difficulty_map import (
    CHANNEL_G,
    difficulty_to_level,
    human_title,
    infer_category,
    infer_required_patterns,
    infer_tags,
    slugify_id,
)

_PROMPT_FOOTER = """
Requirements (SparkDistill / TritonBench harness):
- Target Triton 3.7.1 on workstation Blackwell (CUDA SM12x / SM120)
- Prefer tl.make_tensor_descriptor over deprecated tl.make_block_ptr
- Use @triton.jit kernels with correct boundary masking
- Include @triton.autotune with at least 3 configs when tile sizes matter
- Provide a Python launcher + grid and a torch.allclose correctness test
- Use fp32 accumulators for tl.dot reductions
- Print a short note if you intentionally skip autotune for a tiny elementwise kernel
""".strip()


def load_g_entries(path: str | Path) -> list[dict[str, Any]]:
    """Load the upstream TritonBench-G JSON list."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"expected list in {path}, got {type(data).__name__}")
    return [e for e in data if isinstance(e, dict)]


def _pick_instruction(entry: dict[str, Any], *, use_comp_instru: bool) -> str:
    if use_comp_instru:
        text = (entry.get("comp_instru") or entry.get("simp_instru") or "").strip()
    else:
        text = (entry.get("simp_instru") or entry.get("comp_instru") or "").strip()
    return text


def _gold_relpath(entry: dict[str, Any]) -> str | None:
    fname = (entry.get("file") or "").strip()
    if not fname:
        return None
    if not fname.endswith(".py"):
        fname = f"{fname}.py"
    return f"data/TritonBench_G_v1/{fname}"


def build_g_prompt(entry: dict[str, Any], instruction: str) -> str:
    """Build a teacher/student prompt for a G-channel kernel generation task."""
    fname = entry.get("file") or "kernel.py"
    repo = entry.get("repo") or "unknown"
    parts = [
        f"Write a complete Triton 3.7.1 implementation for the following GPU kernel task.",
        f"Upstream reference file: `{fname}` (repo: {repo}).",
        "",
        "Task description:",
        instruction.strip(),
        "",
        _PROMPT_FOOTER,
    ]
    return "\n".join(parts)


def entry_to_problem(
    entry: dict[str, Any],
    *,
    use_comp_instru: bool = True,
    id_suffix: str = "",
) -> dict[str, Any] | None:
    """Convert one G-channel entry into a harness problem dict, or None if empty."""
    instruction = _pick_instruction(entry, use_comp_instru=use_comp_instru)
    if not instruction or len(instruction) < 40:
        return None

    file_name = entry.get("file") or entry.get("name") or "kernel"
    slug = slugify_id(str(file_name))
    if id_suffix:
        slug = f"{slug}{id_suffix}"

    tags = infer_tags(instruction, file_name, entry.get("repo"))
    level = difficulty_to_level(entry.get("difficulty"), tags=tags)
    category = infer_category(entry, CHANNEL_G)
    required = infer_required_patterns(instruction, category)
    gold = _gold_relpath(entry)

    problem: dict[str, Any] = {
        "id": slug,
        "level": level,
        "category": category,
        "title": human_title(slug),
        "prompt": build_g_prompt(entry, instruction),
        "constraints": {
            "triton_version": "3.7.1",
            "gpu_target": "Blackwell-SM120",
            "dtype_hint": "float32 preferred unless the task requires lower precision",
        },
        "required_patterns": required,
        "tags": tags,
        "source": {
            "channel": CHANNEL_G,
            "file": entry.get("file"),
            "repo": entry.get("repo"),
            "difficulty": str(entry.get("difficulty")) if entry.get("difficulty") is not None else None,
            "star": entry.get("star"),
            "simp_instru_len": entry.get("simp_instru_len"),
            "comp_instru_len": entry.get("comp_instru_len"),
            "output_triton_len": entry.get("output_triton_len"),
        },
    }
    if gold:
        problem["gold_kernel"] = gold
    return problem


def convert_g_channel(
    path: str | Path,
    *,
    limit: int | None = None,
    use_comp_instru: bool = True,
) -> list[dict[str, Any]]:
    """Convert all (or first `limit`) G-channel entries to problems."""
    entries = load_g_entries(path)
    problems: list[dict[str, Any]] = []
    seen: set[str] = set()
    for entry in entries:
        if limit is not None and len(problems) >= limit:
            break
        prob = entry_to_problem(entry, use_comp_instru=use_comp_instru)
        if prob is None:
            continue
        base_id = prob["id"]
        if base_id in seen:
            n = 2
            while f"{base_id}_{n}" in seen:
                n += 1
            prob["id"] = f"{base_id}_{n}"
            prob["title"] = human_title(prob["id"])
        seen.add(prob["id"])
        problems.append(prob)
    return problems


def g_channel_stats(problems: list[dict[str, Any]]) -> dict[str, Any]:
    """Summary counters for converted G problems."""
    by_level: dict[int, int] = {}
    by_tag: dict[str, int] = {}
    for p in problems:
        by_level[p["level"]] = by_level.get(p["level"], 0) + 1
        for tag in p.get("tags") or []:
            by_tag[tag] = by_tag.get(tag, 0) + 1
    return {
        "count": len(problems),
        "by_level": dict(sorted(by_level.items())),
        "top_tags": sorted(by_tag.items(), key=lambda kv: (-kv[1], kv[0]))[:20],
    }
