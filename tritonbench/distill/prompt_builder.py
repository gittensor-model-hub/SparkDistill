"""Teacher / student prompt construction for Triton distillation."""

from __future__ import annotations

from typing import Any

from tritonbench.features.triton_371 import DEFAULT_GPU_TARGET, TRITON_VERSION, system_prompt_for_triton

_FEW_SHOT_SKELETON = '''
Example shape (do not copy numbers blindly):

```python
import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 256}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=8, num_stages=3),
    ],
    key=["n_elements"],
)
@triton.jit
def kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask)
    tl.store(y_ptr + offs, x, mask=mask)

def run(x):
    y = torch.empty_like(x)
    n = x.numel()
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    kernel[grid](x, y, n)
    return y

def test():
    x = torch.randn(10_000, device="cuda", dtype=torch.float32)
    y = run(x)
    assert torch.allclose(y, x)
```
'''.strip()


def build_system_prompt(
    triton_version: str = TRITON_VERSION,
    gpu_target: str = DEFAULT_GPU_TARGET,
    *,
    detailed: bool = True,
) -> str:
    """Build a substantial Triton expert system prompt for teachers/students."""
    base = system_prompt_for_triton(triton_version=triton_version, gpu_target=gpu_target)
    if not detailed:
        return base
    return f"""{base}

You specialize in production-quality Triton {triton_version} kernels for {gpu_target}.
When writing kernels:
- Always mask partial tiles; never assume shapes divide block sizes.
- Prefer tl.make_tensor_descriptor based load/store on Triton 3.7+; avoid tl.make_block_ptr / tl.advance.
- Use fp32 accumulators for tl.dot and for numerically sensitive reductions (softmax, RMSNorm).
- Include @triton.autotune with multiple configs when BLOCK_M/N/K or BLOCK_SIZE matter.
- Expose a clear Python launcher that chooses a grid via triton.cdiv.
- Validate with torch.allclose (or torch.testing.assert_close) against a PyTorch reference.
- Call out dtype/layout assumptions in comments.
- If the task is debugging, identify every bug and return a fixed full module, not a diff-only patch.
- If the task is translation from PyTorch, preserve semantics including broadcasting and edge shapes.

Blackwell SM12x notes:
- Favor tl.dot with IEEE/TF32 input_precision as appropriate; do not assume datacenter TMEM/tcgen05.
- Occupancy matters: pair large tiles with sensible num_warps / num_stages.
"""


def build_sft_user_content(problem: dict[str, Any]) -> str:
    """User-turn content for SFT records (mirrors harness prompting)."""
    user = (problem.get("prompt") or "").strip()
    if problem.get("input_code"):
        user += f"\n\n```python\n{problem['input_code']}\n```"
    constraints = problem.get("constraints")
    if constraints:
        user += f"\n\nConstraints: {constraints}"
    tags = problem.get("tags") or []
    if tags:
        user += f"\n\nTags: {', '.join(tags)}"
    return user


def build_teacher_prompt(
    problem: dict[str, Any],
    *,
    triton_version: str = TRITON_VERSION,
    gpu_target: str = DEFAULT_GPU_TARGET,
    include_few_shot: bool = False,
    variant: str | None = None,
) -> dict[str, str]:
    """Return {system, user} for a teacher completion call."""
    category = (variant or problem.get("category") or "kernel_generation").strip()
    system = build_system_prompt(triton_version, gpu_target, detailed=True)
    if category == "kernel_debugging":
        system += (
            "\nThis is a debugging task. Enumerate bugs briefly in comments, then provide the fixed module."
        )
    elif category == "kernel_translation":
        system += (
            "\nThis is a PyTorch→Triton translation task. Match reference numerics with torch.allclose."
        )
    else:
        system += "\nThis is a kernel generation task from a natural-language specification."

    user = build_sft_user_content(problem)
    if include_few_shot and category != "kernel_debugging":
        user = f"{_FEW_SHOT_SKELETON}\n\n---\n\nNow solve this task:\n\n{user}"
    return {"system": system, "user": user}
