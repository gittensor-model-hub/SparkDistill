# SparkDistill Miner Guide

This guide is for SN74 miners and contributors who want to earn rewards by improving
SparkDistill. SparkDistill's goal is **reasoning distillation**: the student should learn
to reproduce the teacher's step-by-step reasoning, not just its final answers. The rule is
simple: rewards come from verified improvements, not from claims, formatting, or
duplicated ideas.

There are **two mining tracks**, each with its own labels and rewards:

1. **Dataset track** (`dataset:s/m/l`) — run [SparkProof](https://github.com/gittensor-model-hub/SparkProof)
   on a Blackwell CC VM to generate verified Triton training data, publish it to Hugging
   Face, and open a text-only registry PR here. See *Dataset Track* below.
2. **Training track** (`eval:xs/s/m/l/xl`) — train the student on merged datasets with an
   improved recipe, beat the frontier eval, and prove it (RTX PRO 6000 CC attestation,
   ≤ 5h wall-clock). The rest of this guide covers this track.

The same person can mine both: your own merged dataset is the natural raw material for
your training runs — but every merged dataset is public, so a better dataset only keeps
you ahead as long as you also train well.

## Dataset Track (`dataset:s/m/l`)

The full flow, end to end:

1. **Generate** on a Blackwell RTX PRO 6000 CC VM with an unmodified SparkProof checkout:
   `scripts/run_triton_pipeline.sh` (teacher calls go through an approved gateway to the
   pinned teachers — GPT 5.6 Sol / Fable 5 at `xhigh` — and every kernel is validated on
   the GPU).
2. **Publish** with `sparkproof-publish-dataset --bundle <dir> --repo-id <you>/<repo>`.
   The release gate (decontamination + provenance) must pass; the publisher uploads the
   dataset rows **and** the proof artifacts under `proof/` in the same HF repo.
3. **Open a text-only PR** here appending one line to `datasets/registry.jsonl` with your
   HF URL and the `trajectories_sha256` from `dataset_manifest.json` — see
   [`datasets/README.md`](../datasets/README.md). No dataset files are committed.
4. **The validator** runs `python -m eval.dataset_verify --hf-repo <you>/<repo>
   --claimed-sha256 <hash> --sparkproof-root ../SparkProof`, which checks GPU CC
   attestation, the release gate, the row hash, and full SparkProof policy (unmodified
   request hashes, pinned teachers, merkle root). Pass → merged with a size label,
   rewarded by SN74 gittensor:

| label | verified rows |
|---|---|
| `dataset:l` | >= 10000 |
| `dataset:m` | >= 1000 |
| `dataset:s` | >= 100 |
| `dataset:none` | merged, below reward threshold |
| `dataset:REJECT` | attestation, decontamination, hash, or policy failure |

## Training Track: What Scores

A PR can score when it does all of the following:

- Includes the **recipe and dataset** used to produce the checkpoint — either committed,
  or (since generated datasets are large and git-ignored) published externally and linked
  in the PR. See *Sharing Your Dataset* below. This is not optional: no dataset+recipe, no
  score, no matter how good your local eval numbers look.
- Trains / regenerates trajectories from source on the evaluator's hardware.
- Preserves correctness against the frozen benchmark reference (no format breakage, no
  garbled outputs).
- Improves at least one benchmark in the basket by **the current eval threshold or more**.
- Avoids unacceptable regressions in the other guarded benchmarks.
- Changes code that is actually used by the training/eval path for the current phase.

The measured benchmarks are:

| benchmark | target |
|---|---|
| BFCL | function/tool-calling accuracy |
| GSM8K | grade-school math reasoning |
| HumanEval | code generation correctness |
| IFEval | instruction-following accuracy |
| MMLU-Pro | broad knowledge/reasoning |
| AIME | competition-level multi-step math reasoning |
| GPQA-Diamond | graduate-level science reasoning |

Small gains are not aggregated across benchmarks. A PR must clear the threshold on at
least one benchmark without dropping others below their floor.

## What Does Not Score

These changes may be useful, but they do not earn a quality label unless they also
produce a verified frontier improvement:

- Documentation-only changes.
- Refactors with no benchmark improvement.
- Test-only changes.
- Eval harness changes that do not improve measured checkpoint quality.
- Copying an already-merged trajectory set or recipe without a new measurable improvement.
- Changes that improve one synthetic eval path but are unused by the phase's scoring target.

## Quality Gate

The evaluator compares your resulting checkpoint against the current frontier checkpoint
on the frozen benchmark basket. A PR is rejected if it degrades correctness too much on
any guarded benchmark, even if it improves another.

The gate checks:

- Per-benchmark accuracy vs. the frontier checkpoint.
- Held-out prompts not seen during trajectory generation or training.
- Stable, well-formed outputs (no truncation/format collapse from a bad recipe change).

Do not trade breadth for a single benchmark's score. Quality is measured across the whole
basket.

## Regression Labels

A PR can improve one benchmark and regress another. The bot makes this explicit with
benchmark-specific labels:

| label | meaning |
|---|---|
| `regression-bfcl` | BFCL accuracy regressed |
| `regression-gsm8k` | GSM8K accuracy regressed |
| `regression-humaneval` | HumanEval accuracy regressed |
| `regression-ifeval` | IFEval accuracy regressed |
| `regression-mmlu-pro` | MMLU-Pro accuracy regressed |
| `regression-aime24` | AIME accuracy regressed |
| `regression-gpqa-diamond` | GPQA-Diamond accuracy regressed |

If no benchmark improves by at least the eval threshold and any guarded benchmark
regresses, the PR is rejected and may be auto-closed.

## Quality Labels

The reward label is based on the strongest verified benchmark improvement over the
current live frontier checkpoint:

| label | meaning |
|---|---|
| `eval:XL` | very large verified quality improvement |
| `eval:L` | large verified quality improvement |
| `eval:M` | medium verified quality improvement |
| `eval:S` | small verified quality improvement |
| `eval:XS` | minimum accepted verified quality improvement |
| `eval:none` | correct, but no significant improvement |
| `eval:REJECT` | correctness failure, training failure, or unacceptable regression |

The exact label is deterministic from the evaluator output. The bot does not use AI
judgment to decide rewards.

## Sharing Your Dataset And Recipe (Required)

**No trained weights are ever merged.** What actually gets merged — and what the evaluator
actually trusts — is your **recipe (the Axolotl YAML) and the dataset it trained on**,
because those are what the evaluator reproduces from source to verify your claim. This is
also what makes the whole system fair: because the recipe and dataset behind the current
frontier are always public, anyone can fork the leader and try to beat it with one more
optimization. Nobody — including whoever currently holds the frontier ("the king") — can
permanently dominate by keeping a checkpoint secret; there's no way to merge a PR without
its recipe and dataset becoming public too.

In practice today:

- Small recipe changes: just include the changed `sft.yaml` (or new recipe file) in your
  PR as normal.
- Datasets: `data/processed/` is git-ignored (these files are large), so datasets are
  never committed. The preferred path is to train on a dataset already merged through
  the **dataset track** (`datasets/registry.jsonl`) and cite its HF URL via
  `proof.bundle --dataset-url`. If you generated your own data, run it through the
  dataset track first — that gets it verified, labeled, and rewarded on its own, and
  makes your training PR trivially reproducible.

Aggregating datasets across many miners into something more structured than the registry
is still an open research problem (see `CONTRIBUTING.md`'s *Open research: dataset
aggregation* section), but linking a verified registry entry is the expected baseline.

## Proof Of Training (Skip Full Retrain-Verification)

By default, the evaluator retrains/re-evals your PR from source — accurate, but slow.
This section is a shortcut for verifying your **eval claim** only — it has nothing to do
with the dataset-sharing requirement above, which always applies. If your checkpoint beats
the frontier, you can prove your claimed numbers instead of just asserting them, and get a
much cheaper verification pass:

```bash
# install the attestation + Hugging Face publishing extras
uv sync --extra proof

# 1. (optional) attest the GPU you trained/evaluated on, e.g. a Blackwell RTX PRO 6000
#    Server Edition confidential-computing node
python -m eval.attestation --out runs/<run-id>/attestation.json

# 2. package checkpoint + eval scores + training claims into a bundle, publish it to HF
python -m proof.bundle --checkpoint outputs/<your-checkpoint> --scores eval/results/candidate.json \
    --run-id <run-id> --out proof/_bundles/<run-id> \
    --train-hours 4.2 --train-gpu "NVIDIA RTX PRO 6000 Blackwell" \
    --dataset-url https://huggingface.co/datasets/<user>/<merged-dataset>
python -m proof.publish --bundle proof/_bundles/<run-id> --repo-id <your-hf-username>/sparkdistill-<run-id>
```

Put the printed Hugging Face URL — and, if you ran it, your attestation.json — in your
PR. The evaluator runs `eval.verify`: a small held-out re-run of your claimed scores
(not the full basket) plus attestation validation if you provided it. If your claim
doesn't hold up within tolerance, the PR is rejected outright — a proof bundle that
misrepresents its scores is treated as worse than no bundle at all, not just "unverified."

Training-track claims are enforced too: `--train-hours` beyond the **5-hour wall-clock
budget** is `eval:REJECT`, `--train-gpu` must be an **RTX PRO 6000** CC node, and when
you attach a CC attestation, its attested hardware model must corroborate the claimed
GPU (a mismatched attestation is worse than none). `--dataset-url` should point at a
dataset merged through the dataset track (`datasets/registry.jsonl`), which is what
makes the training result reproducible by anyone.

| label | meaning |
|---|---|
| `proof:attested` | GPU CC attestation passed; cheap re-verification only |
| `proof:unattested` | HF bundle submitted, no attestation; cheap re-verification only |
| `proof:none` | no bundle submitted; full retrain-verification applies |

Merged proof-of-training runs are appended to [`runs/ledger.jsonl`](../runs/ledger.jsonl)
— see [`runs/README.md`](../runs/README.md).

## Local Checklist Before Opening A PR

### Triton / SparkProof path (Blackwell CC VM — recommended)

Run from **SparkProof** on the CC VM (sibling **SparkDistill** repo required):

```bash
cd SparkProof
cp .env.example .env   # OPENROUTER_API_KEY

scripts/install.sh              # first boot only
scripts/miner_run.sh --limit 2            # smoke test
scripts/miner_run.sh --run-id my-run-001  # full bundle → verify → SFT

# optional: train after SFT
scripts/miner_run.sh --run-id my-run-001 --train
```

Then eval from SparkDistill:

```bash
cd ../SparkDistill
scripts/eval.sh --checkpoint outputs/qwen3.5-4b-phase1 --compare-frontier
```

### Legacy / local teacher path (no SparkProof)

Run these from the SparkDistill repo root:

```bash
# Trajectory generation (if your PR touches teacher/)
scripts/generate_trajectories.sh --prompts data/prompts/phase1.jsonl --out data/processed/phase1_trajectories.jsonl

# Fold captured reasoning into <think>-tagged SFT records (messages for qwen3_5)
scripts/prepare_sft_data.sh --in data/processed/phase1_trajectories.jsonl --out data/processed/phase1_sft.jsonl --format messages

# Training (if your PR touches recipes/)
scripts/train.sh recipes/qwen3.5-4b-phase1/sft.yaml

# Quality eval — always run before opening a PR
scripts/eval.sh --checkpoint outputs/qwen3.5-4b-phase1 --compare-frontier
```

## PR Requirements

A good PR includes:

- **A link to the dataset you trained on**, if it isn't small enough to commit directly
  (see *Sharing Your Dataset And Recipe* above). Required, not optional.
- A short description of what changed and why (trajectory prompt set, data mix,
  hyperparameter, eval coverage).
- The files and recipes changed.
- Local eval numbers, including which benchmarks moved and by how much.
- Any expected benchmark-specific effect: `bfcl`, `gsm8k`, `humaneval`, `ifeval`,
  `mmlu-pro`, `aime24`, or `gpqa-diamond`.
- If you're using the proof-of-training fast path: your Hugging Face proof-bundle URL
  (and attestation.json, if collected). This is in addition to, not instead of, the
  dataset link above.

Keep PRs narrow. A small recipe or trajectory-prompt PR with a clear eval delta is easier
to verify and merge than a broad rewrite.

## Current Target

The current frontier is Phase 1: **Qwen3.5-4B**, distilled from the teacher basket
(Claude Fable 5, GPT 5.6). The project is especially interested in:

- Higher-quality / more diverse reasoning trajectories, especially for underrepresented
  task types in the benchmark basket — reasoning-heavy prompts (multi-step math, logic,
  proof-style code correctness) matter most since the goal is reasoning distillation.
- Data mix and hyperparameter improvements in `recipes/qwen3.5-4b-phase1/sft.yaml`.
- Eval basket coverage that catches regressions the current benchmarks miss.

## Do Not Game The Eval

The evaluator uses held-out prompts, frozen benchmark data, immutable logs, and
path-aware labels. Attempts to tune for the harness instead of the checkpoint's real
quality can be rejected or ignored.

The best way to earn is to make the shipped student checkpoint genuinely better and keep
it honest.
