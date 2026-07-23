"""Schema validation for TritonBench YAML problems."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

from tritonbench.converters.difficulty_map import LEVEL_DIRS

REQUIRED_FIELDS: tuple[str, ...] = (
    "id",
    "level",
    "category",
    "title",
    "prompt",
)
OPTIONAL_FIELDS: tuple[str, ...] = (
    "constraints",
    "required_patterns",
    "tags",
    "source",
    "gold_kernel",
    "input_code",
    "expected_fix",
    "hints",
    "timeout_s",
    "atol",
    "rtol",
)
VALID_CATEGORIES: frozenset[str] = frozenset(
    {
        "kernel_generation",
        "kernel_translation",
        "kernel_debugging",
    }
)
VALID_CHANNELS: frozenset[str] = frozenset({"G", "T", "synthetic"})
_ID_RE = re.compile(r"^[a-z][a-z0-9_]{1,120}$")


class ProblemSchemaError(ValueError):
    """Raised when a problem fails strict schema validation."""


def _level_ok(level: Any) -> bool:
    if level == "bugfix":
        return True
    if isinstance(level, int) and level in LEVEL_DIRS:
        return True
    if isinstance(level, str) and level.isdigit() and int(level) in (1, 2, 3, 4):
        return True
    return False


def validate_problem(prob: dict[str, Any]) -> list[str]:
    """Return a list of human-readable schema errors (empty means OK)."""
    errors: list[str] = []
    if not isinstance(prob, dict):
        return ["problem must be a mapping"]

    for field in REQUIRED_FIELDS:
        if field not in prob or prob[field] in (None, ""):
            errors.append(f"missing required field: {field}")

    pid = prob.get("id")
    if isinstance(pid, str):
        if not _ID_RE.match(pid):
            errors.append(f"id must be snake_case slug matching {_ID_RE.pattern}: {pid!r}")
    elif pid is not None:
        errors.append(f"id must be str, got {type(pid).__name__}")

    if "level" in prob and not _level_ok(prob["level"]):
        errors.append(f"invalid level: {prob['level']!r}")

    cat = prob.get("category")
    if cat is not None and cat not in VALID_CATEGORIES:
        errors.append(f"invalid category: {cat!r}")

    prompt = prob.get("prompt")
    if isinstance(prompt, str):
        if len(prompt.strip()) < 20:
            errors.append("prompt too short (<20 chars)")
    elif prompt is not None:
        errors.append(f"prompt must be str, got {type(prompt).__name__}")

    title = prob.get("title")
    if title is not None and not isinstance(title, str):
        errors.append(f"title must be str, got {type(title).__name__}")

    for list_field in ("required_patterns", "tags", "hints"):
        val = prob.get(list_field)
        if val is None:
            continue
        if not isinstance(val, list):
            errors.append(f"{list_field} must be a list")
            continue
        for i, item in enumerate(val):
            if not isinstance(item, str) or not item.strip():
                errors.append(f"{list_field}[{i}] must be a non-empty string")

    constraints = prob.get("constraints")
    if constraints is not None and not isinstance(constraints, (dict, str)):
        errors.append("constraints must be a mapping or string")

    source = prob.get("source")
    if source is not None:
        if not isinstance(source, dict):
            errors.append("source must be a mapping")
        else:
            ch = source.get("channel")
            if ch is not None and ch not in VALID_CHANNELS:
                errors.append(f"source.channel must be one of {sorted(VALID_CHANNELS)}")

    gold = prob.get("gold_kernel")
    if gold is not None and not isinstance(gold, str):
        errors.append("gold_kernel must be a string path")

    input_code = prob.get("input_code")
    if input_code is not None and not isinstance(input_code, str):
        errors.append("input_code must be a string")

    if prob.get("category") == "kernel_debugging" and not (prob.get("input_code") or "").strip():
        # Soft warning as error for synthetic bugfix completeness.
        if (prob.get("source") or {}).get("channel") == "synthetic":
            errors.append("kernel_debugging synthetic problems require input_code")

    for key in prob:
        if key not in REQUIRED_FIELDS and key not in OPTIONAL_FIELDS:
            # Unknown keys are allowed but flagged so converters stay intentional.
            if key.startswith("_"):
                continue
            errors.append(f"unknown field: {key}")

    return errors


def validate_problem_strict(prob: dict[str, Any]) -> None:
    """Raise ProblemSchemaError if validation fails."""
    errors = validate_problem(prob)
    # Filter unknown-field warnings for strict mode? Keep all as errors.
    if errors:
        pid = prob.get("id", "<unknown>") if isinstance(prob, dict) else "<unknown>"
        raise ProblemSchemaError(f"problem {pid!r} invalid: " + "; ".join(errors))


def load_and_validate_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML problem and validate it; raise on error."""
    path = Path(path)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ProblemSchemaError(f"{path}: root must be a mapping")
    # Fill id from stem if missing.
    data.setdefault("id", path.stem)
    errs = [e for e in validate_problem(data) if not e.startswith("unknown field:")]
    if errs:
        raise ProblemSchemaError(f"{path}: " + "; ".join(errs))
    return data


def validate_corpus(problems_root: str | Path) -> dict[str, Any]:
    """Validate every YAML under problems_root; return a summary dict."""
    root = Path(problems_root)
    files = sorted(root.rglob("*.yaml")) + sorted(root.rglob("*.yml"))
    # Deduplicate while preserving order.
    seen: set[Path] = set()
    unique_files: list[Path] = []
    for f in files:
        rp = f.resolve()
        if rp in seen:
            continue
        seen.add(rp)
        unique_files.append(f)

    per_file: dict[str, list[str]] = {}
    ids: dict[str, str] = {}
    id_collisions: list[str] = []
    ok = 0
    for path in unique_files:
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001 — collect all corpus errors
            per_file[str(path)] = [f"yaml parse error: {exc}"]
            continue
        if not isinstance(data, dict):
            per_file[str(path)] = ["root must be a mapping"]
            continue
        data.setdefault("id", path.stem)
        # Unknown fields are warnings for corpus scan — don't fail the whole set.
        errors = [e for e in validate_problem(data) if not e.startswith("unknown field:")]
        rel = str(path.relative_to(root)) if path.is_relative_to(root) else str(path)
        pid = str(data.get("id"))
        if pid in ids:
            id_collisions.append(f"{pid}: {ids[pid]} and {rel}")
            errors.append(f"duplicate id {pid!r} (also in {ids[pid]})")
        else:
            ids[pid] = rel
        if errors:
            per_file[rel] = errors
        else:
            ok += 1

    return {
        "problems_root": str(root),
        "total_files": len(unique_files),
        "ok": ok,
        "failed": len(per_file),
        "id_collisions": id_collisions,
        "errors": per_file,
        "by_level_dir": {
            d.name: len(list(d.glob("*.yaml"))) + len(list(d.glob("*.yml")))
            for d in sorted(root.iterdir())
            if d.is_dir()
        },
    }
