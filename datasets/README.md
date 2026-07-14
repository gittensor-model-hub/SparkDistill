# `datasets/`

The in-repo registry of every verified SparkProof dataset that was merged — the
dataset-track counterpart of `runs/` (which records proof-of-training runs).

## How a dataset gets in here (miner flow)

1. Run SparkProof on a Blackwell CC VM and pass the release gate
   (`sparkproof-publish-dataset` refuses to publish otherwise).
2. Publish to Hugging Face. The publisher uploads the dataset rows **and** the proof
   artifacts under `proof/` in the same HF repo (`manifest.json`,
   `dataset_manifest.json`, `gpu_attestation.json`, `trajectories.jsonl`, ...).
3. Build the registry line and open a **text-only PR** that appends it to
   `datasets/registry.jsonl`:

```bash
scripts/registry_line.sh --bundle <sparkproof-bundle-dir> --miner <github-handle> \
  --repo-id <user>/<repo> --append datasets/registry.jsonl
```

   In the PR template, check **Dataset track submission**. Dataset PRs may not modify any
   other file:

```json
{"miner": "<github-handle>", "hf_url": "https://huggingface.co/datasets/<user>/<repo>", "trajectories_sha256": "<from dataset_manifest.json>", "rows_total": 128, "dataset_version": "triton-distill-v0.2"}
```

No dataset files are committed here — the PR is the link plus the hash that pins the
exact gated rows.

## What the validator does

Registry PRs are gated automatically by `.github/workflows/dataset_registry.yml`.
The workflow reads the dataset-track checkbox, rejects changes outside
`datasets/registry.jsonl`, verifies the proof, **aggregates every merged registry line
(including the proposed submission) into the canonical mining dataset on Hugging Face**
(default: [`gittensor-model-hub/sparkproof-mining`](https://huggingface.co/datasets/gittensor-model-hub/sparkproof-mining)),
replaces any stale `dataset:*` label with the computed result, and merges only when
verification, aggregation publish, and the `dataset:xs` threshold all pass.
Rejected PRs are labeled `dataset:REJECT` and closed automatically. Sub-threshold
valid proofs (`dataset:none`) are also closed automatically.

The gate runs `eval.registry_gate`, which for each appended registry line:

1. Validates JSON schema and rejects duplicate `hf_url` / `trajectories_sha256`.
2. Downloads `proof/` from Hugging Face.
3. Runs `eval.dataset_verify` with a pinned SparkProof checkout.

`dataset_verify` checks, in order: required proof artifacts (including
`trajectories_raw.jsonl`, `validation_report.jsonl`, `novelty_report.json`);
GPU CC attestation passed with a content-bound nonce; release gate passed and rows
still match the gated sha256; and full production `sparkproof-verify` (pinned
generator, Fable 5 / GPT 5.6 Sol at `xhigh`, raw→verified consistency, merkle,
attestation nonce). Any failure is `dataset:REJECT` and the PR is not merged.

Manual re-check:

```bash
python -m eval.dataset_verify --hf-repo <user>/<repo> \
    --claimed-sha256 <trajectories_sha256 from the PR> \
    --sparkproof-root ../SparkProof --out eval/results/dataset_report.json
```

| label | verified rows |
|---|---|
| `dataset:xl` | >= 150 |
| `dataset:l` | >= 100 |
| `dataset:m` | >= 75 |
| `dataset:s` | >= 50 |
| `dataset:xs` | >= 25 |
| `dataset:none` | < 25 (proof may be valid, but not merged/rewarded) |
| `dataset:REJECT` | attestation, release-gate, hash, or policy failure |

Merged datasets feed the **single canonical mining dataset** used by every training-track
PR. After each registry merge, CI refreshes [`canonical.json`](canonical.json) with the
pinned `mix_manifest.sft_sha256` and row count.

## Canonical mining dataset (training track)

**HF repo:** [`gittensor-model-hub/sparkproof-mining`](https://huggingface.co/datasets/gittensor-model-hub/sparkproof-mining)

**In-repo pin:** [`canonical.json`](canonical.json) — `repo_id`, `hf_url`,
`training_dataset_path`, and `mix_manifest.sft_sha256`.

Before a registry PR merges, CI:

1. Verifies the miner's `proof/` bundle
2. Mixes **all** registry lines (existing + proposed) with deduplication
3. Publishes the union to the mining dataset repo (`train` split + `mix_manifest.json`)
4. Merges the PR only if publish succeeds

Registry aggregation uses `SPARKDISTILL_MINING_DEDUPE` when republishing sparkproof-mining
(default **`exact`**). **Quality** is enforced by SparkProof before merge (release gate,
decontamination, GPU validation, `sparkproof-verify`). **Dedupe** only removes redundant
copies at mix time — `exact` drops identical prompts; the older `near` mode also dropped
structurally similar rows and shrank nghetienhiep's 161-row submission to 77 when mixed
with speedy00. Set `SPARKDISTILL_MINING_DEDUPE=none` only for local debugging.

After each eligible registry merge, CI refreshes [`canonical.json`](canonical.json) from
the live HF `mix_manifest.json` (also triggered by `.github/workflows/update_canonical_pin.yml`).

**Training-track rule:** recipes must use `data/processed/sparkproof-mining_sft.jsonl`
only. Prepare with `scripts/prepare_mining_sft.sh` (verifies HF matches `canonical.json` and
writes `data/processed/mix_manifest.json` for `proof.bundle --mix-manifest`).
PRs that add local generators, private blends, or alternate recipe paths are rejected by
`eval.training_track_gate`. Proof bundles must set `dataset_url` to the canonical `hf_url`.

Override the mining publish target with `SPARKDISTILL_MINING_DATASET_REPO` in CI or locally.

Local dry-run without HF upload:

```bash
uv run python -m eval.registry_gate ... --skip-mining-publish
```

Refresh the pin manually after a mining republish:

```bash
scripts/update_canonical_pin.sh
```

## Cross-miner mixing (maintainer / registry CI only)

Registry aggregation uses `scripts/mix_registry.sh` internally. **Training miners must not**
build private mixes for competition PRs — train on the canonical snapshot above instead.

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
# → verified=true, label=dataset:none (2 rows < 25 reward threshold)
```

**CC VM gotchas observed during the smoke test:**

- SSH port can change when the VM is reprovisioned (e.g. `20004` not `20002`).
- `SparkDistill/tritonbench/` is gitignored — decontamination and the release gate fail
  without it (`decontamination requires a TritonBench problem corpus`). Sync from a dev
  machine: `rsync -az SparkDistill/tritonbench/ ubuntu@<host>:~/SparkDistill/tritonbench/`.
- `HF_TOKEN` must be in SparkProof `.env` with write access to the target org/repo.

