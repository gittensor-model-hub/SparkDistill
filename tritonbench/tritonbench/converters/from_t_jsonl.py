"""Convert TritonBench-T JSON (misnamed .jsonl) entries into YAML harness problems."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tritonbench.converters.difficulty_map import (
    CHANNEL_T,
    difficulty_to_level,
    human_title,
    infer_category,
    infer_required_patterns,
    infer_tags,
    slugify_id,
)

_PROMPT_FOOTER = """
Requirements (SparkDistill / TritonBench harness):
- Translate the PyTorch reference into Triton 3.7.1 kernels for Blackwell SM120
- Preserve numerics: include torch.allclose vs the reference (atol/rtol appropriate for dtype)
- Use @triton.jit + boundary masks; prefer tl.make_tensor_descriptor over tl.make_block_ptr
- Add @triton.autotune when there are meaningful tile/block choices
- Provide a launcher and a small self-contained test
""".strip()


def load_t_entries(path: str | Path) -> list[dict[str, Any]]:
    """Load TritonBench-T data (JSON array stored under a .jsonl filename)."""
    raw = Path(path).read_text(encoding="utf-8").strip()
    if not raw:
        return []
    # Prefer JSON array; fall back to true JSONL if needed.
    if raw[0] == "[":
        data = json.loads(raw)
        if not isinstance(data, list):
            raise ValueError(f"expected list in {path}")
        return [e for e in data if isinstance(e, dict)]
    entries: list[dict[str, Any]] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        if isinstance(obj, dict):
            entries.append(obj)
    return entries


def _gold_relpath(entry: dict[str, Any]) -> str | None:
    fname = (entry.get("file") or "").strip()
    if not fname:
        name = (entry.get("name") or "").strip()
        if name:
            fname = f"{name}.py"
    if not fname:
        return None
    if not fname.endswith(".py"):
        fname = f"{fname}.py"
    return f"data/TritonBench_T_v1/{fname}"


def _clip(text: str, limit: int = 2500) -> str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 20].rstrip() + "\n... [truncated]"


def build_t_prompt(entry: dict[str, Any]) -> str:
    """Build a translation prompt from T-channel fields."""
    name = entry.get("name") or entry.get("file") or "kernel"
    description = (entry.get("description") or "").strip()
    math = (entry.get("math") or "").strip()
    func_inputs = (entry.get("func_inputs") or "").strip()
    torch_code = (entry.get("torch_code") or "").strip()
    example = (entry.get("example") or "").strip()
    other = (entry.get("other") or "").strip()

    parts = [
        f"Translate the following PyTorch operator into a Triton 3.7.1 kernel suite named `{name}`.",
        "",
        "Operator description:",
        description or "(no description provided)",
    ]
    if func_inputs:
        parts.extend(["", "Target Python signature / call pattern:", func_inputs])
    if math:
        parts.extend(["", "Mathematical definition:", _clip(math, 1800)])
    if torch_code:
        parts.extend(["", "PyTorch reference snippet:", "```python", _clip(torch_code, 2000), "```"])
    if example:
        parts.extend(["", "Example / usage notes:", _clip(example, 1500)])
    if other:
        parts.extend(["", "Additional constraints:", _clip(other, 1200)])
    parts.extend(["", _PROMPT_FOOTER])
    return "\n".join(parts)


def entry_to_problem(entry: dict[str, Any], *, id_suffix: str = "") -> dict[str, Any] | None:
    """Convert one T-channel entry into a harness problem dict."""
    name = entry.get("name") or entry.get("file")
    description = (entry.get("description") or "").strip()
    if not name and not description:
        return None
    if not description and not (entry.get("torch_code") or "").strip():
        return None

    slug = slugify_id(str(name or "kernel"))
    if id_suffix:
        slug = f"{slug}{id_suffix}"

    instruction_blob = " ".join(
        str(entry.get(k) or "")
        for k in ("description", "math", "torch_code", "other", "func_inputs", "name", "file")
    )
    tags = infer_tags(instruction_blob)
    # T-channel tasks are often fused ops — tag them.
    if "fused" in slug and "fused" not in tags:
        tags.append("fused")
    level = difficulty_to_level(entry.get("difficulty"), tags=tags)
    category = infer_category(entry, CHANNEL_T)
    required = infer_required_patterns(instruction_blob, category)
    gold = _gold_relpath(entry)

    constraints: dict[str, Any] = {
        "triton_version": "3.7.1",
        "gpu_target": "Blackwell-SM120",
        "params_cnt": entry.get("params_cnt"),
        "torch_cnt": entry.get("torch_cnt"),
    }
    if func := (entry.get("func_inputs") or "").strip():
        constraints["func_inputs"] = func[:500]

    problem: dict[str, Any] = {
        "id": slug,
        "level": level,
        "category": category,
        "title": human_title(slug),
        "prompt": build_t_prompt(entry),
        "constraints": constraints,
        "required_patterns": required,
        "tags": tags,
        "source": {
            "channel": CHANNEL_T,
            "file": entry.get("file") or f"{slug}.py",
            "name": entry.get("name"),
            "difficulty": str(entry.get("difficulty")) if entry.get("difficulty") is not None else None,
            "params_cnt": entry.get("params_cnt"),
            "torch_cnt": entry.get("torch_cnt"),
        },
    }
    if gold:
        problem["gold_kernel"] = gold
    return problem


def convert_t_channel(path: str | Path, *, limit: int | None = None) -> list[dict[str, Any]]:
    """Convert all (or first `limit`) T-channel entries to problems."""
    entries = load_t_entries(path)
    problems: list[dict[str, Any]] = []
    seen: set[str] = set()
    for entry in entries:
        if limit is not None and len(problems) >= limit:
            break
        prob = entry_to_problem(entry)
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


def t_channel_stats(problems: list[dict[str, Any]]) -> dict[str, Any]:
    by_level: dict[int, int] = {}
    for p in problems:
        by_level[p["level"]] = by_level.get(p["level"], 0) + 1
    return {"count": len(problems), "by_level": dict(sorted(by_level.items()))}
