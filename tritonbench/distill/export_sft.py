"""Export teacher trajectories into SparkDistill SFT JSONL formats."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Literal


@dataclass
class TrajectoryRecord:
    problem_id: str
    model: str
    response: str
    prompt_user: str
    prompt_system: str = ""
    reasoning: str | None = None
    level: int | str | None = None
    category: str | None = None
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _think_wrap(reasoning: str | None, response: str) -> str:
    response = response or ""
    if reasoning and reasoning.strip():
        return f"<think>\n{reasoning.strip()}\n</think>\n{response}"
    return response


def to_messages_record(
    problem: dict[str, Any],
    teacher_response: str,
    *,
    reasoning: str | None = None,
    model: str = "teacher",
    system: str = "",
    user: str | None = None,
) -> dict[str, Any]:
    """SparkDistill messages-format SFT row."""
    user_content = user if user is not None else (problem.get("prompt") or "")
    if problem.get("input_code") and "```python" not in user_content:
        user_content = f"{user_content}\n\n```python\n{problem['input_code']}\n```"
    assistant = _think_wrap(reasoning, teacher_response)
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user_content})
    messages.append({"role": "assistant", "content": assistant})
    return {
        "messages": messages,
        "meta": {
            "problem_id": problem.get("id"),
            "level": problem.get("level"),
            "category": problem.get("category"),
            "tags": list(problem.get("tags") or []),
            "model": model,
            "source": problem.get("source"),
        },
    }


def to_completion_record(
    problem: dict[str, Any],
    teacher_response: str,
    *,
    reasoning: str | None = None,
    model: str = "teacher",
    prompt: str | None = None,
) -> dict[str, Any]:
    """Prompt/completion format alternative."""
    prompt_text = prompt if prompt is not None else (problem.get("prompt") or "")
    return {
        "prompt": prompt_text,
        "completion": _think_wrap(reasoning, teacher_response),
        "meta": {
            "problem_id": problem.get("id"),
            "level": problem.get("level"),
            "category": problem.get("category"),
            "model": model,
        },
    }


def prompt_hash(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def dedupe_by_prompt_hash(records: Iterable[dict[str, Any]], *, prompt_field: str = "messages") -> list[dict[str, Any]]:
    """Drop duplicate SFT rows that share the same user prompt hash."""
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for rec in records:
        if prompt_field == "messages":
            msgs = rec.get("messages") or []
            user = next((m.get("content", "") for m in msgs if m.get("role") == "user"), "")
            key = prompt_hash(user)
        else:
            key = prompt_hash(str(rec.get("prompt", "")))
        if key in seen:
            continue
        seen.add(key)
        out.append(rec)
    return out


def export_sft_records(
    records: Iterable[dict[str, Any]],
    out_path: str | Path,
    format: Literal["messages", "completion"] = "messages",  # noqa: A002 — public CLI name
) -> int:
    """Write JSONL SFT records; return count written."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with out_path.open("w", encoding="utf-8") as fh:
        for rec in records:
            if format == "completion" and "completion" not in rec and "messages" in rec:
                # Best-effort conversion.
                msgs = rec["messages"]
                user = next((m["content"] for m in msgs if m.get("role") == "user"), "")
                asst = next((m["content"] for m in msgs if m.get("role") == "assistant"), "")
                rec = {"prompt": user, "completion": asst, "meta": rec.get("meta", {})}
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            count += 1
    return count


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)
