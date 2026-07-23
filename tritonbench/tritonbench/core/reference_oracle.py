"""Bind harness problems to upstream gold kernels and score structural similarity."""

from __future__ import annotations

import ast
import hashlib
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


def _repo_root() -> Path:
    # tritonbench/tritonbench/core/reference_oracle.py -> tritonbench/
    return Path(__file__).resolve().parents[2]


@dataclass
class OracleBinding:
    problem_id: str
    channel: str
    path: Path
    fingerprint: str
    n_jit: int = 0
    n_load: int = 0
    n_store: int = 0
    func_names: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["path"] = str(self.path)
        return d


def resolve_gold_kernel(problem: dict[str, Any], *, data_root: Path | None = None) -> Path | None:
    """Resolve a gold kernel path from problem metadata."""
    root = data_root or _repo_root()
    gold = problem.get("gold_kernel")
    if isinstance(gold, str) and gold.strip():
        path = Path(gold)
        if not path.is_absolute():
            path = root / path
        if path.exists():
            return path

    source = problem.get("source") or {}
    channel = (source.get("channel") or "").upper()
    fname = source.get("file") or ""
    if not fname and problem.get("id"):
        fname = f"{problem['id']}.py"
    if not fname:
        return None
    if not fname.endswith(".py"):
        fname = f"{fname}.py"

    candidates: list[Path] = []
    if channel == "G":
        candidates.append(root / "data" / "TritonBench_G_v1" / fname)
    elif channel == "T":
        candidates.append(root / "data" / "TritonBench_T_v1" / fname)
    else:
        candidates.extend(
            [
                root / "data" / "TritonBench_G_v1" / fname,
                root / "data" / "TritonBench_T_v1" / fname,
            ]
        )
    for path in candidates:
        if path.exists():
            return path
    return None


def load_gold_source(path: Path) -> str:
    return Path(path).read_text(encoding="utf-8", errors="replace")


def _normalize_ws(source: str) -> str:
    lines = [ln.rstrip() for ln in source.splitlines()]
    # Drop heavy trailing test harnesses commonly appended in upstream gold files.
    clipped: list[str] = []
    for ln in lines:
        if ln.strip().startswith("#" + "#" * 20):
            break
        clipped.append(ln)
    text = "\n".join(clipped).strip() + "\n"
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def gold_fingerprint(source: str) -> str:
    """Stable sha256 of normalized AST dump, falling back to whitespace normalize."""
    try:
        tree = ast.parse(source)
        dumped = ast.dump(tree, annotate_fields=True, include_attributes=False)
        payload = dumped.encode("utf-8")
    except SyntaxError:
        payload = _normalize_ws(source).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _count_pattern(source: str, pattern: str) -> int:
    return len(re.findall(pattern, source))


def _func_names(source: str) -> list[str]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return re.findall(r"^\s*def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", source, re.M)
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            names.append(node.name)
    return names


def analyze_source(source: str) -> dict[str, Any]:
    return {
        "fingerprint": gold_fingerprint(source),
        "n_jit": _count_pattern(source, r"@triton\.jit"),
        "n_load": _count_pattern(source, r"tl\.load\s*\("),
        "n_store": _count_pattern(source, r"tl\.store\s*\("),
        "n_dot": _count_pattern(source, r"tl\.dot\s*\("),
        "n_autotune": _count_pattern(source, r"@triton\.autotune"),
        "func_names": _func_names(source),
        "n_lines": len(source.splitlines()),
        "imports": sorted(
            set(re.findall(r"^\s*(?:import|from)\s+([A-Za-z0-9_\.]+)", source, re.M))
        ),
    }


def build_oracle_index(data_dir: Path | None = None) -> dict[str, OracleBinding]:
    """Index all gold .py kernels under TritonBench_G_v1 and TritonBench_T_v1."""
    root = Path(data_dir) if data_dir else _repo_root() / "data"
    index: dict[str, OracleBinding] = {}
    mapping = {
        "G": root / "TritonBench_G_v1",
        "T": root / "TritonBench_T_v1",
    }
    for channel, folder in mapping.items():
        if not folder.is_dir():
            continue
        for path in sorted(folder.glob("*.py")):
            source = load_gold_source(path)
            meta = analyze_source(source)
            binding = OracleBinding(
                problem_id=path.stem,
                channel=channel,
                path=path,
                fingerprint=meta["fingerprint"],
                n_jit=meta["n_jit"],
                n_load=meta["n_load"],
                n_store=meta["n_store"],
                func_names=list(meta["func_names"]),
            )
            # Prefer unique keys: channel:stem and bare stem (first wins for bare).
            index[f"{channel}:{path.stem}"] = binding
            index.setdefault(path.stem, binding)
    return index


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def compare_to_gold(generated: str, gold: str) -> dict[str, Any]:
    """Structural similarity heuristics between generated code and gold."""
    gen = analyze_source(generated or "")
    ref = analyze_source(gold or "")

    func_sim = _jaccard(set(gen["func_names"]), set(ref["func_names"]))
    import_sim = _jaccard(set(gen["imports"]), set(ref["imports"]))

    def _ratio(a: int, b: int) -> float:
        if a == 0 and b == 0:
            return 1.0
        return min(a, b) / max(a, b) if max(a, b) else 0.0

    jit_r = _ratio(gen["n_jit"], ref["n_jit"])
    load_r = _ratio(gen["n_load"], ref["n_load"])
    store_r = _ratio(gen["n_store"], ref["n_store"])
    line_r = _ratio(gen["n_lines"], ref["n_lines"])
    dot_r = _ratio(gen["n_dot"], ref["n_dot"])

    similarity = (
        0.25 * func_sim
        + 0.15 * import_sim
        + 0.15 * jit_r
        + 0.15 * load_r
        + 0.10 * store_r
        + 0.10 * line_r
        + 0.10 * dot_r
    )
    return {
        "similarity": max(0.0, min(1.0, similarity)),
        "func_similarity": func_sim,
        "import_similarity": import_sim,
        "jit_ratio": jit_r,
        "load_ratio": load_r,
        "store_ratio": store_r,
        "line_ratio": line_r,
        "dot_ratio": dot_r,
        "generated": gen,
        "gold": ref,
    }


def score_against_oracle(
    problem: dict[str, Any],
    generated_code: str,
    *,
    data_root: Path | None = None,
) -> float | None:
    """Return structural similarity to gold, or None if no gold binding exists."""
    path = resolve_gold_kernel(problem, data_root=data_root)
    if path is None:
        return None
    gold = load_gold_source(path)
    return float(compare_to_gold(generated_code, gold)["similarity"])
