"""Prepare Axolotl recipe YAML for reliable local training."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import yaml

from eval.canonical_dataset import assert_recipe_uses_canonical_dataset

# Axolotl multipack sampler fails on very small mixes (observed at 17 rows).
MIN_SAMPLE_PACKING_ROWS = 32

# Rough chars→tokens for chat JSON without loading a tokenizer (~code/English mix).
_CHARS_PER_TOKEN = 4.0
# Chat-template / special-token overhead per message (conservative).
_TOKENS_PER_MESSAGE_OVERHEAD = 8
# Candidate sequence lengths (Axolotl-friendly). Pack-budget only snaps downward.
_SEQUENCE_LEN_CANDIDATES = (1024, 1536, 2048, 2560, 3072, 3584, 4096, 5120, 6144, 8192)
# Only retune when current pad fraction is at least this high.
_MIN_PAD_RATIO_TO_TUNE = 0.15
# When estimated steps/epoch fall at or below this, clamp mid-epoch eval/save I/O.
_IO_THROTTLE_MAX_STEPS = 64

_PATH_KEYS = ("path", "dataset_prepared_path", "output_dir")
_SPARKDISTILL_KEYS = ("sparkdistill_pack_budget", "sparkdistill_io_throttle")


def count_jsonl_rows(path: Path) -> int:
    count = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count


def estimate_row_tokens(row: dict[str, Any]) -> int:
    """Estimate tokens for one SFT chat row without loading a tokenizer."""
    messages = row.get("messages")
    if not isinstance(messages, list):
        # Fall back to raw payload size for non-chat rows.
        return max(1, math.ceil(len(json.dumps(row, ensure_ascii=False)) / _CHARS_PER_TOKEN))

    total = 0
    for message in messages:
        if not isinstance(message, dict):
            continue
        content = message.get("content", "")
        if not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False)
        total += math.ceil(len(content) / _CHARS_PER_TOKEN) + _TOKENS_PER_MESSAGE_OVERHEAD
    return max(1, total)


def profile_jsonl_token_lengths(path: Path) -> tuple[int, list[int]]:
    """Return (row_count, per-row token estimates) for a chat jsonl file."""
    lengths: list[int] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except json.JSONDecodeError:
                lengths.append(1)
                continue
            if isinstance(row, dict):
                lengths.append(estimate_row_tokens(row))
            else:
                lengths.append(1)
    return len(lengths), lengths


def greedy_pack_pad_ratio(lengths: list[int], sequence_len: int) -> float:
    """Next-fit pad ratio for lengths packed into fixed sequence_len bins."""
    if not lengths or sequence_len <= 0:
        return 0.0

    bins = 0
    used = 0
    waste = 0
    for raw in lengths:
        length = min(max(1, raw), sequence_len)
        if used and used + length > sequence_len:
            waste += sequence_len - used
            bins += 1
            used = 0
        used += length
    if used:
        waste += sequence_len - used
        bins += 1
    capacity = bins * sequence_len
    if capacity <= 0:
        return 0.0
    return waste / capacity


def choose_pack_budget_sequence_len(lengths: list[int], current: int) -> int:
    """Snap sequence_len downward when pad_to_sequence_len would waste FLOPs.

    Prefer the candidate with the lowest next-fit pad ratio. Near-ties keep the
    shorter context so attention work shrinks without giving pad back.
    """
    if not lengths or current <= 0:
        return current
    max_len = max(lengths)
    if max_len > current:
        return current

    current_pad = greedy_pack_pad_ratio(lengths, current)
    if current_pad < _MIN_PAD_RATIO_TO_TUNE:
        return current

    best = current
    best_pad = current_pad
    for candidate in _SEQUENCE_LEN_CANDIDATES:
        if candidate < max_len or candidate >= current:
            continue
        pad = greedy_pack_pad_ratio(lengths, candidate)
        # Require a real pad win; on a near-tie prefer the shorter budget.
        if pad + 1e-9 < best_pad - 0.02 or (
            abs(pad - best_pad) <= 0.02 and candidate < best and pad <= current_pad
        ):
            best = candidate
            best_pad = pad
    return best


def _flag_enabled(cfg: dict[str, Any], key: str, *, default: bool) -> bool:
    value = cfg.get(key, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() not in {"0", "false", "no", "off"}
    return bool(value) if value is not None else default


def apply_pack_budget(
    cfg: dict[str, Any],
    lengths: list[int],
    notes: list[str],
) -> None:
    """Lower sequence_len when packing + pad_to_sequence_len is pad-heavy."""
    if not _flag_enabled(cfg, "sparkdistill_pack_budget", default=True):
        notes.append("pack-budget skipped (sparkdistill_pack_budget=false)")
        return
    if not cfg.get("sample_packing") or not cfg.get("pad_to_sequence_len"):
        return

    current = cfg.get("sequence_len")
    if not isinstance(current, int) or current <= 0:
        return

    chosen = choose_pack_budget_sequence_len(lengths, current)
    if chosen >= current:
        notes.append(
            "pack-budget: keep sequence_len="
            f"{current} (pad_ratio={greedy_pack_pad_ratio(lengths, current):.2f})"
        )
        return

    before_pad = greedy_pack_pad_ratio(lengths, current)
    after_pad = greedy_pack_pad_ratio(lengths, chosen)
    cfg["sequence_len"] = chosen
    notes.append(
        f"pack-budget: sequence_len {current}→{chosen} "
        f"(est. pad_ratio {before_pad:.2f}→{after_pad:.2f}, max_row≈{max(lengths)})"
    )


def estimated_steps_per_epoch(cfg: dict[str, Any], row_count: int) -> int:
    micro = cfg.get("micro_batch_size") or 1
    try:
        micro_i = max(1, int(micro))
    except (TypeError, ValueError):
        micro_i = 1
    # Unpacked upper bound; packing coalesces rows so real steps are lower.
    steps = max(1, math.ceil(row_count / micro_i))
    if cfg.get("sample_packing"):
        steps = max(1, math.ceil(steps / 2))
    return steps


def apply_io_throttle(
    cfg: dict[str, Any],
    row_count: int | None,
    notes: list[str],
) -> None:
    """Clamp mid-epoch eval/save when the pack is too small to justify the I/O tax."""
    if not _flag_enabled(cfg, "sparkdistill_io_throttle", default=True):
        notes.append("I/O throttle skipped (sparkdistill_io_throttle=false)")
        return
    if row_count is None:
        return

    steps = estimated_steps_per_epoch(cfg, row_count)
    if steps > _IO_THROTTLE_MAX_STEPS:
        return

    for key in ("evals_per_epoch", "saves_per_epoch"):
        value = cfg.get(key)
        if isinstance(value, (int, float)) and value > 1:
            cfg[key] = 1
            notes.append(
                f"{key} clamped to 1 (small-pack I/O throttle, ~{steps} steps/epoch)"
            )


def _strip_sparkdistill_keys(cfg: dict[str, Any]) -> None:
    for key in _SPARKDISTILL_KEYS:
        cfg.pop(key, None)


def _has_flash_attn() -> bool:
    try:
        import flash_attn  # noqa: F401

        return True
    except ImportError:
        return False


def _has_flash_attn_3() -> bool:
    try:
        import torch
        from transformers.utils import is_flash_attn_3_available

        # Official FA3 wheels currently contain Hopper kernels; they import on
        # Blackwell but fail at launch with "no kernel image".
        if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 10:
            return False
        return bool(is_flash_attn_3_available())
    except ImportError:
        return False


def _has_cut_cross_entropy() -> bool:
    try:
        import cut_cross_entropy  # noqa: F401
        from axolotl.integrations.cut_cross_entropy import CutCrossEntropyPlugin

        CutCrossEntropyPlugin()._check_requirements()
        return True
    except Exception:
        return False


def _resolve_path(value: str, root: Path) -> str:
    path = Path(value)
    if path.is_absolute():
        return str(path)
    return str((root / path).resolve())


def prepare_train_recipe(
    *,
    recipe_path: Path,
    distill_root: Path,
    out_path: Path | None = None,
) -> dict[str, Any]:
    """Return a training-safe recipe with absolute paths and runtime fallbacks."""
    root = distill_root.resolve()
    cfg: dict[str, Any] = yaml.safe_load(recipe_path.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise ValueError(f"{recipe_path} must contain a YAML mapping")

    canonical_issues = assert_recipe_uses_canonical_dataset(cfg)
    if canonical_issues:
        raise ValueError(
            "training recipes must use the pinned canonical mining dataset only: "
            + "; ".join(canonical_issues)
        )

    notes: list[str] = []
    row_count: int | None = None
    token_lengths: list[int] = []

    for key in _PATH_KEYS:
        value = cfg.get(key)
        if isinstance(value, str) and value.strip():
            cfg[key] = _resolve_path(value, root)

    datasets = cfg.get("datasets")
    if isinstance(datasets, list):
        total_rows = 0
        counted_sources = 0
        for entry in datasets:
            if not isinstance(entry, dict):
                continue
            data_path = entry.get("path")
            if isinstance(data_path, str) and data_path.strip():
                entry["path"] = _resolve_path(data_path, root)
            data_path = entry.get("path")
            if isinstance(data_path, str) and data_path.endswith(".jsonl"):
                rows, lengths = profile_jsonl_token_lengths(Path(data_path))
                total_rows += rows
                token_lengths.extend(lengths)
                counted_sources += 1
                notes.append(f"dataset rows: {rows}")
        if counted_sources:
            # Axolotl concatenates all datasets, so the multipack guard below must
            # look at the combined row count, not just the last shard's.
            row_count = total_rows
            if counted_sources > 1:
                notes.append(f"dataset rows (total): {total_rows}")

    if row_count is not None and row_count < MIN_SAMPLE_PACKING_ROWS:
        if cfg.get("sample_packing"):
            cfg["sample_packing"] = False
            notes.append(f"sample_packing disabled (<{MIN_SAMPLE_PACKING_ROWS} rows)")
        if cfg.get("pad_to_sequence_len"):
            cfg["pad_to_sequence_len"] = False
            notes.append("pad_to_sequence_len disabled for small dataset")

    if token_lengths:
        apply_pack_budget(cfg, token_lengths, notes)

    apply_io_throttle(cfg, row_count, notes)

    attn = cfg.get("attn_implementation")
    if attn == "flash_attention_2" and _has_flash_attn_3():
        cfg["attn_implementation"] = "flash_attention_3"
        notes.append("attn_implementation: flash_attention_3 (flash_attn_3 wheel)")
    elif attn == "flash_attention_2" and not _has_flash_attn():
        cfg["attn_implementation"] = "sdpa"
        notes.append("attn_implementation: sdpa (flash_attn not installed)")

    plugins = cfg.get("plugins")
    if isinstance(plugins, list) and plugins:
        cce_reason: str | None = None
        if not _has_cut_cross_entropy():
            cce_reason = "not installed"
        elif cfg.get("chat_template") == "qwen3_5":
            # CCE's qwen3_5 patch currently fails against transformers' remote-code
            # module layout (FileNotFoundError on modeling_qwen3_5).
            cce_reason = "unsupported for qwen3_5 chat_template"
        if cce_reason:
            cfg.pop("plugins", None)
            notes.append(f"removed CutCrossEntropyPlugin ({cce_reason})")

    _strip_sparkdistill_keys(cfg)

    destination = out_path or (root / "data" / "prepared" / f"{recipe_path.stem}.prepared.yaml")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    return {
        "source_recipe": str(recipe_path.resolve()),
        "prepared_recipe": str(destination.resolve()),
        "row_count": row_count,
        "notes": notes,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path, required=True, help="source Axolotl yaml")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("."),
        help="SparkDistill repo root for resolving relative paths",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="write prepared yaml here (default: data/prepared/<recipe-stem>.prepared.yaml)",
    )
    args = parser.parse_args(argv)

    try:
        result = prepare_train_recipe(
            recipe_path=args.recipe,
            distill_root=args.root,
            out_path=args.out,
        )
    except Exception as exc:
        print(f"train prep failed: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
