"""Generate the full TritonBench YAML problem corpus from upstream sources."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from tritonbench.converters.difficulty_map import LEVEL_DIRS, level_dirname
from tritonbench.converters.from_g_json import convert_g_channel, g_channel_stats
from tritonbench.converters.from_t_jsonl import convert_t_channel, t_channel_stats
from tritonbench.converters.validate_problem import validate_problem

# Hand-written seeds that must never be overwritten by converters.
SEED_IDS: frozenset[str] = frozenset({"vector_add", "softmax", "wrong_mask"})

# Curated synthetic bugfix templates — common Triton footguns.
BUGFIX_TEMPLATES: list[dict[str, Any]] = [
    {
        "id": "wrong_mask",
        "title": "Incorrect Boundary Mask in Softmax",
        "tags": ["debugging", "masking", "softmax"],
        "expected_fix": "Add mask = col_offsets < n_cols on load/store",
        "prompt": (
            "The following Triton kernel computes row-wise softmax but produces wrong "
            "results when n_cols is not a multiple of BLOCK_SIZE. Find and fix all bugs."
        ),
        "input_code": '''@triton.jit
def softmax_kernel(output_ptr, input_ptr, input_row_stride,
                   output_row_stride, n_cols, BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)
    row_start_ptr = input_ptr + row_idx * input_row_stride
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets
    row = tl.load(input_ptrs)
    row_minus_max = row - tl.max(row, axis=0)
    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis=0)
    softmax_output = numerator / denominator
    output_ptrs = output_ptr + row_idx * output_row_stride + col_offsets
    tl.store(output_ptrs, softmax_output)
''',
    },
    {
        "id": "missing_fp32_accum",
        "title": "Missing FP32 Accumulator in Dot",
        "tags": ["debugging", "matmul", "precision"],
        "expected_fix": "Accumulate tl.dot into tl.float32 acc then cast on store",
        "prompt": (
            "This matmul kernel loses precision on Blackwell because it accumulates in "
            "float16. Fix numerical stability while keeping the same API."
        ),
        "input_code": '''@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
                  stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a = tl.load(a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b = tl.load(b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)
    acc = tl.dot(a, b)
    tl.store(c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn, acc)
''',
    },
    {
        "id": "off_by_one_pid",
        "title": "Off-by-One program_id Blocking",
        "tags": ["debugging", "elementwise"],
        "expected_fix": "Use block_start = pid * BLOCK_SIZE, not (pid + 1) * BLOCK_SIZE",
        "prompt": "Vector add silently drops the first BLOCK_SIZE elements. Find the grid indexing bug.",
        "input_code": '''@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = (pid + 1) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + y, mask=mask)
''',
    },
    {
        "id": "wrong_stride_layout",
        "title": "Wrong Stride on Row-Major Tensor",
        "tags": ["debugging", "layout"],
        "expected_fix": "Index with row * stride_row + col, not row * N always",
        "prompt": "Row-wise scale kernel corrupts outputs for non-contiguous inputs. Fix stride usage.",
        "input_code": '''@triton.jit
def scale_rows_kernel(x_ptr, out_ptr, scales_ptr, M, N, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    # BUG: ignores actual row stride; assumes tightly packed rows
    x = tl.load(x_ptr + row * N + cols, mask=mask)
    s = tl.load(scales_ptr + row)
    tl.store(out_ptr + row * N + cols, x * s, mask=mask)
''',
    },
    {
        "id": "forgotten_mask_store",
        "title": "Load Masked but Store Unmasked",
        "tags": ["debugging", "masking"],
        "expected_fix": "Pass the same mask to tl.store",
        "prompt": "Kernel loads with a mask but stores without one, writing OOB. Fix it.",
        "input_code": '''@triton.jit
def relu_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    y = tl.where(x > 0, x, 0.0)
    tl.store(out_ptr + offs, y)
''',
    },
    {
        "id": "bad_atomic_add",
        "title": "Race Without Atomic on Reduction",
        "tags": ["debugging", "reduction"],
        "expected_fix": "Use tl.atomic_add for concurrent updates to the same address",
        "prompt": "Histogram kernel has a data race on concurrent bin updates. Fix synchronization.",
        "input_code": '''@triton.jit
def hist_kernel(x_ptr, hist_ptr, n, n_bins, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0)
    bins = tl.minimum(x.to(tl.int32), n_bins - 1)
    # BUG: plain store races
    tl.store(hist_ptr + bins, tl.load(hist_ptr + bins) + 1, mask=mask)
''',
    },
    {
        "id": "wrong_axis_reduce",
        "title": "Reduction on Wrong Axis",
        "tags": ["debugging", "reduction", "softmax"],
        "expected_fix": "Reduce along axis=0 for 1D tile vectors",
        "prompt": "Softmax-style reduce uses axis=1 on a 1D tile and fails to compile or runs wrong. Fix axes.",
        "input_code": '''@triton.jit
def row_max_kernel(x_ptr, out_ptr, n_cols, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    x = tl.load(x_ptr + row * n_cols + cols, mask=mask, other=float("-inf"))
    m = tl.max(x, axis=1)
    tl.store(out_ptr + row, m)
''',
    },
    {
        "id": "int_div_truncation",
        "title": "Integer Division Truncation in Grid",
        "tags": ["debugging", "launch"],
        "expected_fix": "Use triton.cdiv(n, BLOCK) for grid sizing",
        "prompt": (
            "Launcher uses n // BLOCK_SIZE for the grid and drops a partial block. "
            "Show the corrected launcher + kernel."
        ),
        "input_code": '''def launch_add(x, y):
    out = torch.empty_like(x)
    n = x.numel()
    BLOCK = 1024
    grid = (n // BLOCK,)  # BUG
    add_kernel[grid](x, y, out, n, BLOCK_SIZE=BLOCK)
    return out
''',
    },
    {
        "id": "dtype_mismatch_store",
        "title": "Store FP32 Acc into FP16 Pointer Without Cast",
        "tags": ["debugging", "dtype"],
        "expected_fix": "Cast accumulator with .to(tl.float16) before store",
        "prompt": "Kernel accumulates in fp32 but stores into an fp16 tensor without casting. Fix dtype handling.",
        "input_code": '''@triton.jit
def saxpy_kernel(x_ptr, y_ptr, out_ptr, a, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask).to(tl.float32)
    y = tl.load(y_ptr + offs, mask=mask).to(tl.float32)
    acc = a * x + y
    tl.store(out_ptr + offs, acc, mask=mask)
''',
    },
    {
        "id": "reinterpret_pointer_bug",
        "title": "Incorrect Pointer Arithmetic on Strided View",
        "tags": ["debugging", "layout"],
        "expected_fix": "Multiply offsets by element size only when using raw byte pointers",
        "prompt": "Kernel treats float32 pointers as byte addresses and skips elements. Fix pointer math.",
        "input_code": '''@triton.jit
def copy_kernel(src_ptr, dst_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    # BUG: * 4 as if byte addressing while ptr is element-typed
    val = tl.load(src_ptr + offs * 4, mask=mask)
    tl.store(dst_ptr + offs * 4, val, mask=mask)
''',
    },
    {
        "id": "missing_other_on_masked_load",
        "title": "Masked Load Without other= Poisoning Reduction",
        "tags": ["debugging", "masking", "reduction"],
        "expected_fix": "Pass other=0.0 or -inf appropriate for the reduction",
        "prompt": "Sum reduction is polluted by garbage in masked-out lanes. Fix the load.",
        "input_code": '''@triton.jit
def sum_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    tl.store(out_ptr + tl.program_id(0), tl.sum(x, axis=0))
''',
    },
    {
        "id": "autotune_key_mismatch",
        "title": "Autotune Key Not Matching Runtime Shape",
        "tags": ["debugging", "autotune"],
        "expected_fix": "Put the dynamic size used in masking into autotune key",
        "prompt": "Autotune configs never match the runtime N used for masking. Fix the key list.",
        "input_code": '''@triton.autotune(
    configs=[triton.Config({"BLOCK_SIZE": 128}, num_warps=4),
             triton.Config({"BLOCK_SIZE": 256}, num_warps=8)],
    key=["M"],  # BUG: kernel masks with N
)
@triton.jit
def row_kernel(x_ptr, out_ptr, M, N, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N
    x = tl.load(x_ptr + row * N + cols, mask=mask, other=0.0)
    tl.store(out_ptr + row * N + cols, x * 2, mask=mask)
''',
    },
    {
        "id": "swapped_mn_tiles",
        "title": "Swapped M/N Tile Offsets in Matmul",
        "tags": ["debugging", "matmul"],
        "expected_fix": "pid_m drives M offsets, pid_n drives N offsets",
        "prompt": "Matmul writes a transposed-looking result due to swapped program ids. Fix indexing.",
        "input_code": '''@triton.jit
def gemm_kernel(a_ptr, b_ptr, c_ptr, M, N, K, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    # BUG: swapped
    offs_m = pid_n * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_m * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a = tl.load(a_ptr + offs_m[:, None] * K + offs_k[None, :])
    b = tl.load(b_ptr + offs_k[:, None] * N + offs_n[None, :])
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc += tl.dot(a, b)
    tl.store(c_ptr + offs_m[:, None] * N + offs_n[None, :], acc)
''',
    },
    {
        "id": "inplace_alias_bug",
        "title": "In-Place Alias Corrupting Input Mid-Kernel",
        "tags": ["debugging", "memory"],
        "expected_fix": "Write to a distinct output buffer or stage in registers/SRAM",
        "prompt": "Kernel reads and writes the same buffer with overlapping tiles. Fix aliasing.",
        "input_code": '''@triton.jit
def prefix_kernel(x_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    # BUG: store back before neighbors finish
    tl.store(x_ptr + offs, x + 1.0, mask=mask)
''',
    },
    {
        "id": "constexpr_runtime_mix",
        "title": "Using Runtime Value as tl.constexpr Shape",
        "tags": ["debugging", "compile"],
        "expected_fix": "Keep BLOCK sizes constexpr; pass runtime n only for masks",
        "prompt": "Kernel fails to compile because a runtime argument is used as a tensor shape. Fix it.",
        "input_code": '''@triton.jit
def fill_kernel(out_ptr, n, value):
    # BUG: n is runtime but used like constexpr shape
    offs = tl.arange(0, n)
    tl.store(out_ptr + offs, value)
''',
    },
    {
        "id": "wrong_num_warps_hint",
        "title": "num_warps Inconsistent with BLOCK Size",
        "tags": ["debugging", "autotune"],
        "expected_fix": "Choose num_warps consistent with tile footprint / occupancy",
        "prompt": (
            "Config pairs BLOCK_SIZE=1024 with num_warps=1 and OOMs / underutilizes. "
            "Propose corrected autotune configs."
        ),
        "input_code": '''@triton.autotune(
    configs=[triton.Config({"BLOCK_SIZE": 1024}, num_warps=1)],
    key=["n"],
)
@triton.jit
def heavy_kernel(x_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
    offs = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    tl.store(out_ptr + offs, x * x, mask=mask)
''',
    },
    {
        "id": "causal_mask_inclusive",
        "title": "Inclusive vs Exclusive Causal Mask Off-by-One",
        "tags": ["debugging", "attention"],
        "expected_fix": "Use q_idx >= k_idx or > consistently with the reference",
        "prompt": "Causal attention mask is off-by-one vs PyTorch tril. Fix the comparison.",
        "input_code": '''@triton.jit
def causal_scores_kernel(q_ptr, k_ptr, out_ptr, seq, BLOCK: tl.constexpr):
    q_idx = tl.program_id(0)
    k_offs = tl.arange(0, BLOCK)
    mask = k_offs < seq
    q = tl.load(q_ptr + q_idx)
    k = tl.load(k_ptr + k_offs, mask=mask, other=0.0)
    scores = q * k
    # BUG: exclusive where reference is inclusive
    scores = tl.where(q_idx > k_offs, scores, float("-inf"))
    tl.store(out_ptr + q_idx * seq + k_offs, scores, mask=mask)
''',
    },
    {
        "id": "rms_eps_missing",
        "title": "RMSNorm Missing Epsilon",
        "tags": ["debugging", "normalization"],
        "expected_fix": "Add eps inside the rsqrt denominator",
        "prompt": "RMSNorm diverges / NaNs on near-zero rows because eps is omitted. Fix it.",
        "input_code": '''@triton.jit
def rmsnorm_kernel(x_ptr, w_ptr, out_ptr, n_cols, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    x = tl.load(x_ptr + row * n_cols + cols, mask=mask, other=0.0).to(tl.float32)
    var = tl.sum(x * x, axis=0) / n_cols
    rstd = tl.rsqrt(var)  # BUG: missing + eps
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    tl.store(out_ptr + row * n_cols + cols, (x * rstd * w).to(tl.float16), mask=mask)
''',
    },
    {
        "id": "dropout_mask_reuse",
        "title": "Dropout Mask Not Regenerated Per Row",
        "tags": ["debugging", "dropout"],
        "expected_fix": "Seed / offset Philox per row (or per element) instead of broadcasting one mask",
        "prompt": "Dropout applies the same mask to every row. Fix RNG offsets.",
        "input_code": '''@triton.jit
def dropout_kernel(x_ptr, out_ptr, n_rows, n_cols, p, seed, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    x = tl.load(x_ptr + row * n_cols + cols, mask=mask, other=0.0)
    # BUG: seed ignores row, so every row shares the mask
    random = tl.rand(seed, cols)
    keep = random > p
    tl.store(out_ptr + row * n_cols + cols, tl.where(keep, x / (1 - p), 0.0), mask=mask)
''',
    },
    {
        "id": "block_ptr_deprecated",
        "title": "Deprecated make_block_ptr Usage on Triton 3.7",
        "tags": ["debugging", "api_modernity"],
        "expected_fix": "Migrate to tl.make_tensor_descriptor load/store APIs",
        "prompt": (
            "This kernel uses tl.make_block_ptr / tl.advance which are deprecated in Triton 3.7.1. "
            "Rewrite using tensor descriptors while preserving numerics."
        ),
        "input_code": '''@triton.jit
def copy_block_ptr_kernel(src_ptr, dst_ptr, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    src = tl.make_block_ptr(
        base=src_ptr, shape=(M, N), strides=(N, 1),
        offsets=(pid_m * BLOCK_M, pid_n * BLOCK_N),
        block_shape=(BLOCK_M, BLOCK_N), order=(1, 0),
    )
    dst = tl.make_block_ptr(
        base=dst_ptr, shape=(M, N), strides=(N, 1),
        offsets=(pid_m * BLOCK_M, pid_n * BLOCK_N),
        block_shape=(BLOCK_M, BLOCK_N), order=(1, 0),
    )
    block = tl.load(src)
    tl.store(dst, block)
''',
    },
    {
        "id": "softmax_no_max_sub",
        "title": "Softmax Without Max Subtraction",
        "tags": ["debugging", "softmax", "precision"],
        "expected_fix": "Subtract row max before exp for stability",
        "prompt": "Softmax overflows to inf on large logits. Fix numerical stability.",
        "input_code": '''@triton.jit
def softmax_unstable_kernel(out_ptr, in_ptr, stride, n_cols, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    x = tl.load(in_ptr + row * stride + cols, mask=mask, other=float("-inf"))
    num = tl.exp(x)  # BUG: no max subtraction
    den = tl.sum(num, axis=0)
    tl.store(out_ptr + row * stride + cols, num / den, mask=mask)
''',
    },
    {
        "id": "gather_oob_index",
        "title": "Gather Without Index Bounds Check",
        "tags": ["debugging", "embedding"],
        "expected_fix": "Clamp or mask indices before gather loads",
        "prompt": "Embedding gather crashes on out-of-range indices. Add safe indexing.",
        "input_code": '''@triton.jit
def gather_kernel(weight_ptr, index_ptr, out_ptr, n, emb_dim, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    idx = tl.load(index_ptr + offs, mask=mask, other=0)
    # BUG: no clamp against nrow
    rows = tl.load(weight_ptr + idx[:, None] * emb_dim + tl.arange(0, emb_dim)[None, :])
    tl.store(out_ptr + offs[:, None] * emb_dim + tl.arange(0, emb_dim)[None, :], rows, mask=mask[:, None])
''',
    },
]


def _dump_yaml(prob: dict[str, Any]) -> str:
    return yaml.safe_dump(
        prob,
        sort_keys=False,
        allow_unicode=True,
        default_flow_style=False,
        width=100,
    )


def write_problem_yaml(prob: dict[str, Any], out_dir: str | Path, *, overwrite: bool = False) -> Path:
    """Write one problem YAML into out_dir/<id>.yaml."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{prob['id']}.yaml"
    if path.exists() and not overwrite:
        if prob["id"] in SEED_IDS:
            return path
        # Overwrite generated files by default when regenerating corpus.
        overwrite = True
    if path.exists() and prob["id"] in SEED_IDS and not overwrite:
        return path
    path.write_text(_dump_yaml(prob), encoding="utf-8")
    return path


def synthetic_bugfix_problems() -> list[dict[str, Any]]:
    """Expand BUGFIX_TEMPLATES into full problem dicts."""
    problems: list[dict[str, Any]] = []
    for tmpl in BUGFIX_TEMPLATES:
        prob: dict[str, Any] = {
            "id": tmpl["id"],
            "level": "bugfix",
            "category": "kernel_debugging",
            "title": tmpl["title"],
            "prompt": tmpl["prompt"],
            "input_code": tmpl["input_code"].strip("\n") + "\n",
            "expected_fix": tmpl.get("expected_fix", ""),
            "required_patterns": ["@triton.jit", "mask"]
            if "mask" in tmpl["id"] or "masking" in tmpl.get("tags", [])
            else ["@triton.jit"],
            "tags": list(tmpl.get("tags") or ["debugging"]),
            "constraints": {
                "triton_version": "3.7.1",
                "gpu_target": "Blackwell-SM120",
            },
            "source": {
                "channel": "synthetic",
                "file": f"{tmpl['id']}.py",
                "difficulty": "bugfix",
            },
        }
        # wrong_mask already exists as seed — keep schema identical enough.
        problems.append(prob)
    return problems


def _target_dir(problems_root: Path, level: int | str) -> Path:
    return problems_root / level_dirname(level)


def generate_corpus(
    *,
    data_dir: str | Path,
    problems_root: str | Path,
    channels: tuple[str, ...] = ("G", "T"),
    include_seed: bool = True,
    include_bugfix: bool = True,
    dry_run: bool = False,
    limit: int | None = None,
    g_name: str = "TritonBench_G_v1.json",
    t_name: str = "TritonBench_T_v1.jsonl",
) -> dict[str, Any]:
    """Build the on-disk YAML corpus. Returns stats."""
    data_dir = Path(data_dir)
    problems_root = Path(problems_root)
    problems: list[dict[str, Any]] = []

    if "G" in channels:
        g_path = data_dir / g_name
        problems.extend(convert_g_channel(g_path, limit=limit))
    if "T" in channels:
        t_path = data_dir / t_name
        # Apply remaining budget if limit set across channels.
        remaining = None if limit is None else max(0, limit - len(problems))
        problems.extend(convert_t_channel(t_path, limit=remaining))

    # Ensure unique ids across channels (G and T can share stems).
    seen_ids: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for prob in problems:
        pid = prob["id"]
        if pid in seen_ids:
            channel = (prob.get("source") or {}).get("channel", "x").lower()
            candidate = f"{pid}_{channel}"
            n = 2
            while candidate in seen_ids:
                candidate = f"{pid}_{channel}_{n}"
                n += 1
            prob = dict(prob)
            prob["id"] = candidate
            prob["title"] = prob.get("title") or candidate
        seen_ids.add(prob["id"])
        deduped.append(prob)
    problems = deduped

    bugfix = synthetic_bugfix_problems() if include_bugfix else []

    # Ensure level directories exist.
    for level in list(LEVEL_DIRS):
        _target_dir(problems_root, level).mkdir(parents=True, exist_ok=True)

    written = 0
    skipped_seed = 0
    skipped_invalid = 0
    by_level: dict[str, int] = {}

    def _write(prob: dict[str, Any], *, is_bugfix: bool = False) -> None:
        nonlocal written, skipped_seed, skipped_invalid
        # Relax unknown-field noise; strip nothing.
        errors = [e for e in validate_problem(prob) if not e.startswith("unknown field:")]
        if errors:
            skipped_invalid += 1
            return
        level = "bugfix" if is_bugfix else prob["level"]
        if prob["id"] in SEED_IDS and include_seed:
            # Do not overwrite seeds.
            target = _target_dir(problems_root, level)
            path = target / f"{prob['id']}.yaml"
            if path.exists():
                skipped_seed += 1
                by_level[str(level)] = by_level.get(str(level), 0) + 1
                return
        if dry_run:
            written += 1
            by_level[str(level)] = by_level.get(str(level), 0) + 1
            return
        overwrite = prob["id"] not in SEED_IDS
        write_problem_yaml(prob, _target_dir(problems_root, level), overwrite=overwrite)
        written += 1
        by_level[str(level)] = by_level.get(str(level), 0) + 1

    for prob in problems:
        _write(prob, is_bugfix=False)
    for prob in bugfix:
        _write(prob, is_bugfix=True)

    g_probs = [p for p in problems if (p.get("source") or {}).get("channel") == "G"]
    t_probs = [p for p in problems if (p.get("source") or {}).get("channel") == "T"]

    return {
        "dry_run": dry_run,
        "written": written,
        "skipped_seed": skipped_seed,
        "skipped_invalid": skipped_invalid,
        "by_level": dict(sorted(by_level.items(), key=lambda kv: str(kv[0]))),
        "g": g_channel_stats(g_probs),
        "t": t_channel_stats(t_probs),
        "bugfix_templates": len(bugfix),
        "problems_root": str(problems_root),
    }
