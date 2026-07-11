# SparkDistill contribution

## Track

Select the one track this PR belongs to.

- [ ] **Dataset track submission**
- [ ] **Training/evaluation improvement**

> Dataset PRs must append exactly one line to `datasets/registry.jsonl` and must
> not change any other file. The dataset workflow reads the checked box above,
> verifies the Hugging Face `proof/` bundle, assigns a `dataset:*` label, and
> merges only submissions with at least 100 verified rows.

## Dataset submission

Complete this section only when **Dataset track submission** is checked.

- Hugging Face dataset URL:
- Verified row count:
- `trajectories_sha256`:
- SparkProof dataset version:

Registry line:

```json
{"miner": "<github-handle>", "hf_url": "https://huggingface.co/datasets/<org>/<repo>", "trajectories_sha256": "<64-character hash from dataset_manifest.json>", "rows_total": 100, "dataset_version": "triton-distill-v0.2"}
```

### Dataset checklist

- [ ] I generated this dataset with an unmodified, pinned SparkProof checkout.
- [ ] The release gate and production `sparkproof-verify` pass.
- [ ] The Hugging Face repository contains the complete `proof/` directory.
- [ ] The submitted rows are training data, not `test`, `eval`, or `held_out` data.
- [ ] The dataset does not contain TritonBench or other protected evaluation material.
- [ ] I understand that fewer than 100 verified rows receives `dataset:none` and is not merged.

## Training/evaluation improvement

Complete this section only when **Training/evaluation improvement** is checked.

- Dataset URL / registry entry:
- Recipe changed:
- Frontier benchmark delta:
- Proof-bundle URL (optional):

## Summary

Explain what changed, why it should improve the student, and how you tested it.
