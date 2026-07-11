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
