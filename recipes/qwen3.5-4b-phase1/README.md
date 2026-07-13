# Qwen3.5-4B — Phase 1 recipe notes

Phase 1 target from the SparkDistill roadmap: prove the trajectory-generation →
Axolotl SFT → eval-harness loop end to end on a dense student model that's cheap
to iterate on. **Qwen3.5-4B** fits BF16 LoRA comfortably on RTX PRO 6000 Blackwell
(96 GB), trains faster than larger students, and uses Axolotl's native
`chat_template: qwen3_5` (no `qwen_25` stand-in).

Official model: [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) (hybrid
Gated DeltaNet + attention; text-only SFT is supported like other Qwen3.5 dense
checkpoints).

Requires **Axolotl with Qwen3.5 examples** (see upstream `examples/qwen3.5/`) and
`transformers>=5.2.0`.

## Data mix

SparkDistill's goal is reasoning distillation, not generic instruction
following: the student should learn to reproduce the teacher's step-by-step
reasoning, not just its final answers. `data/prompts/phase1.jsonl` is
reasoning-heavy (multi-step math, logic puzzles, proof-style code
correctness) so teachers actually produce a reasoning trace worth
distilling.

The pipeline runs in two stages:

1. `teacher/generate.py` → `data/processed/phase1_trajectories.jsonl`, a
   basket of raw teacher trajectories from Claude Fable 5 and GPT 5.6
   (see `teacher/providers.py`), each carrying `response` and, when the
   teacher can capture it, a separate `reasoning` field.
2. `teacher/format.py --format messages` → `data/processed/phase1_sft.jsonl`,
   folding `reasoning` into the assistant message as a leading
   `<think>...</think>` block ahead of the final
   response — Qwen3.5's native chat template supports this format.

`sft.yaml` trains on that second file via Axolotl `type: chat_template` and
`roles_to_train: ["assistant"]` (assistant-only loss).

Optionally clean the raw trajectories between those two stages with
`teacher/filter.py` (`scripts/filter_trajectories.sh`): it drops empty,
malformed, and refusal rows — and, opt-in, over-length or duplicate-prompt
rows — before formatting. It is conservative by default (only structurally
broken rows and refusals), so a plain run never discards a legitimate short
answer.

### Reasoning-capture limitation

Not every teacher can produce a `reasoning` value: `AnthropicTeacher` enables
Claude Fable 5's extended thinking and captures the `thinking` block — but GPT 5.6
via `OpenAICompatibleTeacher` may expose no capturable reasoning tokens over chat
completions. Trajectories from such teachers fall back to training on the response
alone (no `<think>` block). Weight the `--provider` mix toward
`anthropic` if reasoning-capture rate matters for a given run.

## Canonical mining dataset (HF)

```bash
scripts/install_train.sh
scripts/prepare_mining_sft.sh
scripts/train.sh recipes/qwen3.5-4b-phase1/sft-mining.yaml
```

`scripts/install_train.sh` installs Qwen3.5's `torchvision` dependency and builds
FlashAttention 2 for Blackwell SM120 (the first build takes several minutes).
Set `SPARKDISTILL_SKIP_FLASH_ATTN=1` only when an SDPA fallback is intentional.
`scripts/train.sh` resolves recipe paths from the SparkDistill root and disables
`sample_packing` when the jsonl has fewer than 32 rows (Axolotl multipack otherwise
crashes).

## Local phase-1 trajectories

```bash
scripts/generate_trajectories.sh --prompts data/prompts/phase1.jsonl --out data/processed/phase1_trajectories.jsonl

# optional: drop empty/malformed/refusal rows before formatting
scripts/filter_trajectories.sh \
  --in data/processed/phase1_trajectories.jsonl \
  --out data/processed/phase1_trajectories.filtered.jsonl

scripts/prepare_sft_data.sh \
  --in data/processed/phase1_trajectories.filtered.jsonl \
  --out data/processed/phase1_sft.jsonl \
  --format messages

scripts/install_train.sh
scripts/train.sh recipes/qwen3.5-4b-phase1/sft.yaml
scripts/eval.sh --checkpoint outputs/qwen3.5-4b-phase1 --compare-frontier
```

Each line in `phase1_sft.jsonl`:

```json
{
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "<think>\\n...\\n</think>\\n\\n..."}
  ]
}
```

## Hyperparameter rationale

- **BF16 LoRA (`adapter: lora`, no 4-bit)** — 4B fits in VRAM on a single
  Blackwell workstation GPU with headroom for `micro_batch_size: 2` and seq 4096;
  avoids QLoRA dequant overhead.
- **`lora_r: 64`, `lora_alpha: 128`** — attention + MLP projections; ~5–15%
  slower than r=16, not 4×.
- **`micro_batch_size: 2`** — starting point on 96 GB; raise if tokens/sec is
  stable and VRAM allows.
- **`sequence_len: 4096`** — long enough for most teacher trajectories without
  paying quadratic attention cost of much longer context during iteration.
- **`chat_template: qwen3_5`** — native Axolotl Qwen3.5 template; do not use
  `qwen_25` for this checkpoint family.

## Hardware: RTX PRO 6000 Blackwell 96 GB

Suggested starting point (already in `sft.yaml`):

- `micro_batch_size: 2`, `gradient_accumulation_steps: 4` → effective batch 8
- `sequence_len: 4096`, `sample_packing: true`

### Training time (50k examples, pre-generated teachers)

Rough planning at **~1.5–2k tokens/example** (Triton code):

| Epochs | Tokens (50k × 1.75k) | Approx. wall time |
|---:|---:|---:|
| 1 | ~87M | **8–18 h** |
| 2 | ~175M | **16–35 h** |
| 3 | ~262M | **24–50 h** |

Measure tokens/sec after 100–200 steps. For synthetic Triton teacher data, prefer
**1–2 epochs** unless held-out TritonBench composite keeps improving.

At `sequence_len: 8192`, multiply wall time by ~1.5–2× if packing length approaches the cap.

## Known gaps (contributions welcome, see `CONTRIBUTING.md`)

- `teacher/filter.py` covers structural, length, refusal, and exact-prompt
  filtering, but there is no *semantic* quality pass yet (e.g. scoring answer
  correctness or reasoning quality) — a filtered mix still trains on whatever
  survives those rules.
- **`CutCrossEntropyPlugin`** — remove from `sft.yaml` if your Axolotl build lacks it.
- **Linear attention modules** (`linear_attn.*`) — add to `lora_target_modules` only
  if you need to adapt Gated DeltaNet paths; start with standard attn + MLP.
