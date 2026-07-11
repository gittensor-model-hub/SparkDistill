# `datasets/`

The in-repo registry of every verified SparkProof dataset that was merged — the
dataset-track counterpart of `runs/` (which records proof-of-training runs).

## How a dataset gets in here (miner flow)

1. Run SparkProof on a Blackwell CC VM and pass the release gate
   (`sparkproof-publish-dataset` refuses to publish otherwise).
2. Publish to Hugging Face. The publisher uploads the dataset rows **and** the proof
   artifacts under `proof/` in the same HF repo (`manifest.json`,
   `dataset_manifest.json`, `gpu_attestation.json`, `trajectories.jsonl`, ...).
3. Open a **text-only PR** against this repo that appends one JSON line to
   `datasets/registry.jsonl`:

```json
{"miner": "<github-handle>", "hf_url": "https://huggingface.co/datasets/<user>/<repo>", "trajectories_sha256": "<from dataset_manifest.json>", "rows_total": 128, "dataset_version": "triton-distill-v0.2"}
```

No dataset files are committed here — the PR is the link plus the hash that pins the
exact gated rows.

## What the validator does

The validator runs `eval.dataset_verify` against the PR's HF link:

```bash
python -m eval.dataset_verify --hf-repo <user>/<repo> \
    --claimed-sha256 <trajectories_sha256 from the PR> \
    --sparkproof-root ../SparkProof --out eval/results/dataset_report.json
```

This checks, in order: the proof artifacts exist; the GPU CC attestation passed; the
SparkProof release gate passed (decontamination + provenance) and the rows still match
the gated sha256; and — with `--sparkproof-root` — full `sparkproof-verify` policy
(pinned Fable 5 / GPT 5.6 Sol teachers at `xhigh`, unmodified request hashes, merkle
root, Blackwell GPU profile). Any failure is `dataset:REJECT` and the PR is closed.

On pass, the PR is merged with a size label from verified row count, and SN74
gittensor rewards the label:

| label | verified rows |
|---|---|
| `dataset:l` | >= 10000 |
| `dataset:m` | >= 1000 |
| `dataset:s` | >= 100 |
| `dataset:none` | < 100 (merged, below reward threshold) |
| `dataset:REJECT` | attestation, release-gate, hash, or policy failure |

Merged datasets become fair game for the training track: any training miner may cite a
registry entry's `hf_url` as the dataset behind a proof-of-training PR.

- **`registry.jsonl`** — append-only, one line per merged dataset PR. Never edited or
  reordered; corrections are appended, not rewritten (same convention as
  `runs/ledger.jsonl`).

## Verified smoke test (2026-07-11)

End-to-end run on a Blackwell RTX PRO 6000 CC VM (`ssh -p 20004 ubuntu@<host>`):

```bash
# SparkProof on the CC VM (sibling SparkDistill required for decontamination + SFT)
cd SparkProof
# .env: YUNWU_API_KEY or OPENROUTER_API_KEY, HF_TOKEN (org write access)
# SparkDistill/tritonbench must exist (gitignored — rsync or clone beside SparkProof)

scripts/run_triton_pipeline.sh \
  --run-id triton-cc-hf-001 \
  --limit 2 \
  --release-gate \
  --publish gittensor-model-hub/sparkproof-triton-v0
```

**Published:** [gittensor-model-hub/sparkproof-triton-v0](https://huggingface.co/datasets/gittensor-model-hub/sparkproof-triton-v0)

| check | result |
|---|---|
| rows published | 2 (both silver tier) |
| duplicate prompts / task_ids / responses | none — `api_tl_tensor`, `api_tl_tensor_descriptor` |
| release gate | `passed: true`, `blocked_rows: 0` |
| `trajectories_sha256` | `a746fa812fb098737cded713daf0f58b8ff59e485c9bdf8fd94f6b5cc1d5c846` |
| `proof/` artifacts on HF | yes (`manifest.json`, `dataset_manifest.json`, `gpu_attestation.json`, `trajectories.jsonl`, ...) |

**Validator re-check** (any machine with SparkProof + SparkDistill checkouts):

```bash
cd SparkDistill
python -m eval.dataset_verify \
  --hf-repo gittensor-model-hub/sparkproof-triton-v0 \
  --claimed-sha256 a746fa812fb098737cded713daf0f58b8ff59e485c9bdf8fd94f6b5cc1d5c846 \
  --sparkproof-root ../SparkProof \
  --out eval/results/dataset_report.json
# → verified=true, label=dataset:none (2 rows < 100 reward threshold)
```

**CC VM gotchas observed during the smoke test:**

- SSH port can change when the VM is reprovisioned (e.g. `20004` not `20002`).
- `SparkDistill/tritonbench/` is gitignored — decontamination and the release gate fail
  without it (`decontamination requires a TritonBench problem corpus`). Sync from a dev
  machine: `rsync -az SparkDistill/tritonbench/ ubuntu@<host>:~/SparkDistill/tritonbench/`.
- `HF_TOKEN` must be in SparkProof `.env` with write access to the target org/repo.

