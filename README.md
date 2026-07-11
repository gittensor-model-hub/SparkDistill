![SparkDistill banner](docs/images/banner.png)

# _SP⚡RKDISTILL_

**Continuously-improving student models for the SparkInfer runtime.**

**SPARKDISTILL** turns frontier teacher models (Claude Fable 5, GPT 5.6) into small,
fast student checkpoints — via trajectory distillation and Axolotl fine-tuning — that
[`sparkinfer`](https://github.com/gittensor-ai-lab/sparkinfer) then serves at the edge on
consumer and edge Blackwell GPUs.

**Built through [SN74 on Gittensor](https://gittensor.io/).** Gittensor helps power
SPARKDISTILL the same way it powers `sparkinfer`: contributors submit PRs (trajectory
generators, training recipes, eval improvements), a deterministic harness scores the
resulting checkpoint's quality against the current frontier, and SN74 rewards verified
improvements. This is a project living inside the existing SN74 subnet, not a separate
subnet.

## Why SPARKDISTILL

`sparkinfer` makes inference fast; SPARKDISTILL makes the *model* worth serving fast.
SPARKDISTILL's goal is **reasoning distillation**: teach a student model to reproduce a
teacher's step-by-step reasoning process, not just its final answers. Frontier-quality
outputs on consumer hardware require a student distilled from teacher reasoning
trajectories, not just a raw quantized copy of a big model or a student fine-tuned on
answers alone. SPARKDISTILL owns that pipeline:

- **Trajectory generation.** Prompt a basket of teacher models on reasoning-heavy tasks
  (multi-step math, logic, proof-style code correctness), capturing each teacher's
  chain-of-thought/reasoning trace separately from its final response where the
  provider exposes one (Claude extended thinking).
- **Reasoning-format SFT data.** Fold the captured reasoning into the training target
  as a leading `<think>...</think>` block ahead of the response — matching Qwen3's
  native chat-template format — so the student learns to reason, not just answer.
- **Distillation recipes.** Axolotl-based SFT/LoRA recipes tuned per student model and
  per phase, sized for the hardware SPARKINFER already targets.
- **Quality eval.** A benchmark harness (BFCL, GSM8K, HumanEval, IFEval, MMLU-Pro, plus
  hard-reasoning benchmarks AIME and GPQA-Diamond) that scores a student checkpoint's
  quality relative to its teacher and the current frontier checkpoint.

## Layout & scoring

| Path | What |
|---|---|
| [`teacher/`](teacher) | teacher-trajectory generation — Anthropic (Fable 5) and OpenAI (GPT 5.6) only |
| [`recipes/`](recipes) | Axolotl training recipes per student model / phase |
| [`eval/`](eval) | quality benchmark harness + student-vs-frontier scoring + cheap proof verification (harness/scoring scripts are maintainer-owned) |
| [`proof/`](proof) | proof-of-training bundle packaging + Hugging Face publishing |
| [`runs/`](runs) | immutable ledger of merged, verified runs |

**Scoring is quality-only.** SN74 pays each merged PR for its verified marginal quality
improvement over the current best ("frontier") checkpoint, labeled **XL / L / M / S / XS**
by the deterministic eval loop, the same tiering shape `sparkinfer` uses for speedups.
Tooling, bench, docs, and refactors are welcome but score 0 unless they produce a verified
frontier improvement. See [`.gittensor/weights.json`](.gittensor/weights.json) and
[`docs/miner-guide.md`](docs/miner-guide.md).

## How a PR gets merged & rewarded

**No trained weights are ever the merged artifact.** A submission is a **recipe + the
dataset it was trained on** — both fully reproducible from source — plus the eval numbers
that resulted from running them:

1. A miner picks (or generates) a dataset — teacher trajectories from `teacher/generate.py`,
   optionally reformatted for reasoning via `teacher/format.py` — and a training recipe
   (an Axolotl `sft.yaml` under `recipes/`).
2. They train and score locally against the current frontier (`scripts/eval.sh`).
3. If it beats the frontier, they open a PR containing **the recipe file and the dataset
   (or a public link to it)** — this is the actual submission — plus their local eval
   numbers.
4. The evaluator retrains from that exact recipe + dataset on its own hardware and
   re-scores against the frontier. That retrain is the source of truth; nothing about the
   PR is trusted on the miner's word.
5. If it clears the quality gate, it's merged and labeled **XL / L / M / S / XS** by the
   measured delta, and the new checkpoint becomes the frontier.

A checkpoint published via the proof-of-training fast path (below) is **never** the thing
being merged or trusted directly — it only lets the evaluator skip straight to a cheap
re-score of the claimed numbers instead of a full retrain. The recipe and dataset are what
get merged, audited, and reused.

**Why share the recipe and dataset instead of just the weights:** whoever holds the
frontier ("the king") is required to have a fully public recipe + dataset behind their
merged checkpoint — there is no way to merge a PR without them. That means every other
miner can immediately fork the current best recipe/dataset and try to beat it, instead of
one miner permanently sitting on a secret checkpoint nobody else can build on. Verified
improvement is what gets rewarded, so "copy the leader and add one optimization" is a
completely valid — and expected — way to compete.

> **Known gap:** `data/processed/` (generated trajectories / formatted datasets) is
> git-ignored because these files are large. Until dataset-aggregation tooling exists (see
> *Roadmap* below), publish the dataset you trained on externally — e.g. a Hugging Face
> `datasets` repo, the same way `proof/publish.py` publishes checkpoint bundles — and link
> it in your PR description. There's no automated dataset-hosting step yet; this is a
> manual step on the miner's side for now.

## Quickstart

```bash
# 1. install
uv sync

# 2. generate teacher trajectories (needs teacher API keys, see .env.example)
scripts/generate_trajectories.sh --prompts data/prompts/phase1.jsonl --out data/processed/phase1_trajectories.jsonl

# 3. fold captured reasoning into <think>-tagged SFT records (messages format)
scripts/prepare_sft_data.sh --in data/processed/phase1_trajectories.jsonl --out data/processed/phase1_sft.jsonl --format messages

# 4. train the Phase 1 student (Qwen3.5-4B) on those trajectories
scripts/train.sh recipes/qwen3.5-4b-phase1/sft.yaml

# 5. score the resulting checkpoint against the frontier
scripts/eval.sh --checkpoint outputs/qwen3.5-4b-phase1 --compare-frontier
```

## Proof of training

A submission's **eval claim** can skip full retrain-verification if the miner proves it
instead of just asserting it. This is a verification shortcut for the eval *numbers* — it
does not replace sharing the recipe + dataset above, which is required on every PR
regardless of whether this fast path is used:

1. Fine-tune locally and score against the current frontier (`scripts/eval.sh`).
2. If you beat the frontier, attest the GPU you trained/evaluated on — e.g. a
   Blackwell RTX PRO 6000 Server Edition confidential-computing (CC) node
   (`python -m eval.attestation`).
3. Package the checkpoint + eval scores into a bundle and publish it to Hugging
   Face (`python -m proof.bundle`, `python -m proof.publish`) — the resulting
   HF URL is your proof link.
4. Open a PR referencing the HF proof link (and your attestation, if you collected
   one), **and** the recipe + dataset link as described above.
5. The evaluator cheaply re-verifies — a small held-out re-run of your claimed
   scores plus (if provided) attestation validation, **not** a full retrain — and
   merges if it checks out (`python -m eval.verify`).
6. The merge is appended to the immutable `runs/ledger.jsonl` log, and the new
   checkpoint becomes the frontier for the next submission.

Unattested submissions still go through the slower path: full retrain-from-source
verification, same as before this feature existed. See
[`docs/miner-guide.md`](docs/miner-guide.md) for the exact commands and
[`runs/README.md`](runs/README.md) for the ledger format.

## Verifying dataset proofs (no CC VM required)

The **dataset track** rewards verified SparkProof training data merged via
[`datasets/registry.jsonl`](datasets/registry.jsonl). Miners **prove** datasets on a
Blackwell CC VM with [SparkProof](https://github.com/gittensor-model-hub/SparkProof);
validators **verify** from the Hugging Face `proof/` directory on any CPU host — GitHub
Actions, a laptop, no NVIDIA GPU.

```bash
# Validator / local re-check (downloads proof/ from HF)
python -m eval.dataset_verify \
  --hf-repo <user>/<repo> \
  --claimed-sha256 <trajectories_sha256 from the PR> \
  --sparkproof-root ../SparkProof \
  --out eval/results/dataset_report.json

# Under the hood this re-runs production sparkproof-verify (offline by default)
uv run sparkproof-verify --bundle /path/to/proof
uv run sparkproof-verify --bundle /path/to/proof --online   # + NVIDIA NRAS JWKS signature
```

Registry PRs are gated automatically by
[`.github/workflows/dataset_registry.yml`](.github/workflows/dataset_registry.yml) — see
[`datasets/README.md`](datasets/README.md) for the miner flow and label thresholds
(`dataset:xs/s/m/l/xl` merge, `dataset:none` below threshold, `dataset:REJECT` closed).

### What offline verification enforces

For each trajectory row, production verification checks the **stored bundle** — not live
hardware or live teacher API calls:

| Check | What it proves |
|---|---|
| `provider` + `model` | Only `claude-fable-5` (Anthropic) and `gpt-5.6` / `gpt-5.6-sol` (OpenAI) |
| `gateway` + `gateway_model` | Call went through **OpenRouter** or **yunwu** with pinned slugs |
| `request_sha256` | The committed request body matches the pinned call: model slug + `reasoning.effort=xhigh` + prompt/settings |
| `metadata.gateway_response_model` (yunwu) | Response model slug is also pinned |
| raw → verified consistency | Miner cannot swap `trajectories.jsonl` after GPU attestation / release gate |
| `gpu_attestation` nonce | Attestation is bound to `trajectories_raw.jsonl`, not a different dataset |
| release gate + PR hash | `trajectories_sha256` in the PR still matches the gated HF artifact |

**Offline verify means:** the miner recorded the exact pinned teacher slugs
(`claude-fable-5` + `gpt-5.6-sol`) via an approved gateway at `xhigh` reasoning, and did
not tamper with the bundle after proving. It is **not** a live cryptographic proof that
OpenAI/Anthropic actually served those models on every call.

### Offline vs online

| Mode | Teacher model guarantee | GPU guarantee |
|---|---|---|
| **Offline** (registry CI today) | Bundle claims + `request_sha256` + gateway slug metadata + tamper checks | Stored `gpu_attestation.json` fields + nonce binding |
| **Online (`--online`)** | Same as offline | Above **plus** NVIDIA NRAS JWT signature verified against NVIDIA JWKS |
| **Online + OpenRouter ledger** | Can re-query OpenRouter generation IDs — only for `gateway=openrouter` and only with the creating API key | Same as online |

For **yunwu** bundles there is currently no external teacher ledger re-check. Swapping rows
to another model (e.g. `gpt-4o-mini`) is caught by policy + raw/verified consistency.
Full detail: [SparkProof README — Verifying proofs](https://github.com/gittensor-model-hub/SparkProof#verifying-proofs-no-cc-vm-required).

## Miner guide

If you are contributing for SN74 rewards, start with
[`docs/miner-guide.md`](docs/miner-guide.md). It explains what scores, what gets
rejected, and the local commands to run before opening a PR.

## Roadmap

**Phase 1 — Qwen3.5-4B proof of concept.** Prove the trajectory-generation → Axolotl SFT →
eval-harness loop end to end on a dense student model that's cheap to iterate on and fits
comfortably on the hardware `sparkinfer` already targets (RTX PRO 6000 Blackwell class).

**Phase 2 — Qwen3.6-35B-A3B.** Extend the pipeline to the MoE student model that matches
`sparkinfer`'s own MoE decode focus (Qwen3-MoE family), once Phase 1's loop is proven and
the eval basket is stable.

**Phase 3 — Continuous distillation.** Feed verified frontier checkpoints back into
`sparkinfer`'s benchmark and eval-trust pipeline automatically, closing the loop between
model quality improvements here and serving-speed improvements there.

**Phase 4 (research) — Dataset aggregation.** Today, sharing the dataset behind a
submission is a manual step (see the *Known gap* callout above). The open problem is
turning "a pile of miner-submitted trajectories" into a single **qualified dataset** that
can be trusted and combined across contributors — under active design, not yet
implemented:

- **Split into two repos.** One repo for training recipes (what exists here today), a
  separate repo purely for aggregating *validated* trajectory datasets, so dataset review
  and recipe review can evolve independently.
- **Trustless dataset generator.** A way to prove a submitted dataset was actually
  produced by the claimed teacher (e.g. Fable 5) via its API, rather than trusting the
  miner's claim — conceptually similar in spirit to `eval.attestation`'s GPU
  confidential-computing proof, but for *data provenance* instead of *compute
  provenance*. No concrete design exists yet; contributions/proposals welcome.

## License

MIT, see [`LICENSE`](LICENSE).
