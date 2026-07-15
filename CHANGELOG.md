# Changelog

All notable changes to SparkDistill are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/); versions follow semver.

## [Unreleased]

## [0.1.2] — 2026-07-15

Hopper joins Blackwell on both mining tracks, per-architecture frontiers replace the
shared Triton scoreboard, and dataset verification closes the last userland trust gaps
(NRAS JWKS + Intel TDX). Architecture-scoped exact dedupe and a refreshed canonical mining
mix (174→178 rows) land in the same release window.

### Added
- **Canonical mining dataset for every training PR** ([#89], [#97]): `datasets/canonical.json`
  pins `gittensor-model-hub/sparkproof-mining`; recipes must cite the canonical path,
  HF URL, and `sft_sha256`. Training gate rejects local generators, private blends, and
  registry edits; published HF proof bundles are mandatory ([#97]).
- **Training-track canonical pin grace window** ([#121], fixes [#118]): when dataset
  merges advance the pin mid-train (~60 min cycles), the gate accepts any `sft_sha256`
  from the PR merge-base through `main` HEAD. Cite the pin you trained on in the PR body.
- **Hopper H100/H200 dataset generation** ([#104], SparkProof [#20]): registry entries
  carry required `gpu_architecture` (`blackwell` / `hopper`), cross-checked against the
  re-verified bundle. SparkProof stamps architecture-specific prompts and validation.
- **Per-architecture TritonBench frontiers** ([#104], [#109]): `runs/frontiers.json`
  holds separate Blackwell and Hopper buckets; `eval.verify` tiers each bundle against
  its own architecture's frontier instead of comparing hardware-sensitive speed numbers
  across GPUs. `eval.triton_bench` records the architecture a run executed on.
- **Attested no-GPU validator path** ([#101], [#104]): miners export
  `attested_eval_samples.json` on a GPU CC + Intel TDX guest once; validators
  re-check claimed scores from bundled artifacts on CPU alone — no checkpoint
  reproduction, no harness re-run. GSM8K uses a frozen 50-problem set with CPU
  re-grading.
- **Intel TDX for dataset-track bundles** ([#122], SparkProof [#22]): production
  dataset verification requires `gpu_attestation.tdx` with `report_data` bound to the
  dataset nonce. Legacy bundles without a `tdx` key are grandfathered; `"tdx": null`
  rejects on new bundles.
- **NRAS JWKS verification in the dataset gate** ([#53]): `sparkproof-verify --online`
  is now always passed — hand-crafted attestations fail NVIDIA signature verification.
- **Fair dataset reward labels** ([#116]): labels come from canonical-mix
  `rows_selected` after cross-registry dedupe, not raw bundle `verified_rows`.
- **Accepted-registry snapshot for miner-side novelty** ([#119]): CI publishes
  `accepted_registry_snapshot.jsonl` + `accepted_task_ids.json` on the canonical mining
  HF repo so miners can run SparkProof `--registry-snapshot` before spending GPU time.
- **Training-track ledger automation** ([#127]): `training_track_ledger.yml` appends
  `runs/ledger.jsonl` and `runs/<run-id>/result.json` on every merged training PR;
  backfilled [#120].
- **Training-track attested verification in CI** ([#126]): gate runs `eval.verify`'s
  CPU-only checks (claim binding, JWKS, TDX DCAP, attested samples) against the PR's
  committed `attestation.json` and per-arch frontier.
- **6-hour cron safety net** for canonical pin refresh when registry merges skip the
  in-job commit path.
- **Registry merges:** nghetienhiep sparkproof-triton-xl-001 ([#96], 161 rows),
  magicrails-xs-v1 ([#105], 25 rows), sparkproof-triton-xl-002 ([#112], 159 rows).
- **First Hopper training BASELINE** ([#120]): magicrails-hopper-v2 on H200 —
  `eval:BASELINE`, first verified run in the Hopper frontier bucket.

### Changed
- **SN74 eval tier multipliers (2× dataset at same letter)** ([#125]): training-track
  `eval:XL/L/M/S/XS` pay **2×** `dataset:xl/l/m/s/xs` at the same tier;
  `eval:BASELINE` = **2.0**. Documented in `.gittensor/weights.json` and miner guide;
  live config via [gittensor #1635](https://github.com/entrius/gittensor/pull/1635).
- **Mining mix dedupe defaults to `exact`** ([#98]): only identical prompts drop at mix
  time; quality still enforced by SparkProof pre-merge. `dedupe_mode` recorded in
  `mix_manifest.json` ([#98] policy docs). Exact dedupe is **architecture-scoped**
  ([#133], SparkProof [#26]): same prompt on Blackwell vs Hopper is a fresh row.
- **Miner docs: registry snapshot workflow** ([#132]): `sparkproof-publish-dataset
  --mining-repo` and `accepted_registry_snapshot.jsonl` pins in `docs/miner-guide.md`
  and `datasets/README.md`.
- **Canonical pin grows 94 → 133 → 174 → 178 rows** as registry merges land ([#99],
  [#115], [#107], [#134]); `prepare_mining_sft` export format now matches registry
  publish ([#100]).
- **Training GPU claims** accept H100/H200 and B200/B300 in proof bundles ([#98]).
- **Blackwell training defaults to SDPA** with hardened Qwen3.5 train prep ([#84]).
- **Auto-close `dataset:none` PRs** ([#91]); PR template checkboxes accept bold or plain
  ([#82]).
- **GitHub Pages** updated for Hopper support and per-architecture frontiers ([#113]).

### Fixed
- **Stale `datasets/canonical.json` after registry auto-merges** ([#107]): missing git
  identity in `dataset_registry.yml` caused silent pin drift; sibling
  `update_canonical_pin.yml` never fired on token merges.
- **`prepare_mining_sft` hash mismatch** ([#100]): export now uses the same compact JSON
  format as registry publish.
- **`eval.score` NameError** (`TIER_BENCHMARK`) and missing `import os` in
  `registry_gate` pin path — surfaced while wiring Hopper ([#104]).
- **Default missing `gpu_architecture` to Blackwell** for legacy bundles ([#106]).
- **H200 attestation corroboration** ([#126]): genuine H200 nodes report `hwmodel=GH100`
  (same die as H100); old check wrongly required `gh200` tokens.
- **Invalid YAML in `update_canonical_pin.yml`** blocked the workflow entirely ([#98]).
- **Architecture-scoped dataset dedupe** ([#133], SparkProof [#26]): exact dedupe keys
  prompt matches by `gpu_architecture` in mining mix, registry snapshot, and SparkProof
  novelty gate.
- **Canonical mining pin refresh** ([#134]): republished `gittensor-model-hub/sparkproof-mining`
  with arch-aware dedupe — **174 → 178 rows** (+4 cross-arch prompts wrongly dropped).

## [0.1.1] — 2026-07-12

Hardens the proof of training into dual-vendor authenticated, measured-VM
attestation — every gap closeable with today's infrastructure is closed.

### Added
- **Intel TDX measured-VM proof** ([#45]): `eval.attestation` captures a TDX quote
  via configfs-tsm with the bundle's `claim_sha256` in its 64-byte REPORTDATA;
  MRTD (guest-image measurement) recorded. `eval.verify` reports the binding as
  `tdx_bound`. Non-TDX hosts record `"tdx": null`; once-per-boot provisioning
  documented in the miner guide.
- **Intel DCAP verification of TDX quotes** ([#48]): `verify_tdx_quote` (via
  `dcap-qvl`) validates the quote's ECDSA signature, PCK certificate chain to
  Intel's root CA, QE identity, and platform TCB status against live Intel PCS
  collateral — reported as `tdx_signature` (`"UpToDate"` + no advisories is the
  clean pass). Verified live on a real 5,247-byte quote from the Targon TDX guest.
- **NVIDIA JWKS verification of GPU tokens** ([#49]): `verify_gpu_token`
  validates every NRAS-signed JWT in the EAT (platform + per-device, ES384,
  issuer and expiry enforced) against NVIDIA's published JWKS — reported as
  `gpu_signature`. The SDK-local HS256 overall JWT is intentionally not counted
  as evidence. With this, nothing in an attestation is taken on the miner's word:
  both hardware roots of trust (NVIDIA and Intel) are authenticated end-to-end.

### Changed
- Baseline run `2026-07-11-qwen3.5-4b-mining-001` re-attested with GPU + TDX
  claim binding; ledger record carries the strongest available proof ([#46]).
- Website describes the double (GPU + measured-VM) claim binding ([#47]).

### Fixed
- Honest claims were rejected on cross-server generation drift ([#51]): the
  triton composite over the 3-problem quick set moved 2.1pp between vLLM
  server instances, past the 2pp tolerance — found by the full live function
  test. Benchmarks gain a per-benchmark `claim_tolerance_pct` (triton: 5pp
  while the problem set is tiny) and the eval server pins determinism
  (`--seed 0 --no-enable-prefix-caching`).

## [0.1.0] — 2026-07-11

First complete, working release of the SparkDistill miner economy: train a
Triton-specialized student on verified data, prove the run cryptographically,
and verify it from public artifacts alone.

### Added
- **Project foundation**: teacher-trajectory generation (Anthropic/OpenAI),
  `<think>`-format SFT data preparation, Axolotl recipes for Qwen3.5-4B,
  benchmark harness, proof-of-training bundle packaging + Hugging Face
  publishing, immutable run ledger.
- **Dataset track** (`dataset:xs`–`xl`) ([#1], [#4]–[#30]): SparkProof bundles
  verified end-to-end by CI — release gate, GPU CC attestation, sha256 pinning,
  novelty checks — with auto-merged registry PRs (`datasets/registry.jsonl`)
  and automatic aggregation into the canonical mining dataset
  (`gittensor-model-hub/sparkproof-mining`).
- **TritonBench domain benchmark** ([#4], [#32]): vendored Triton 3.7.1 /
  Blackwell harness (thunlp/tritonbench @ 603e28a) — generated kernels are
  compiled and executed on the GPU. Registered as the `triton` improvement
  signal in the eval basket; the general basket (GSM8K, BFCL, HumanEval,
  IFEval, MMLU-Pro, AIME, GPQA-Diamond) acts as the regression guard.
- **Reproducible Blackwell training** ([#33]–[#37]): recipe preparation with
  absolute paths, small-dataset guards (`sample_packing` auto-disable),
  SM-specific FlashAttention 2/3 selection with SDPA fallback, HF→local
  mining-dataset export, pinned training installer.
- **Weights-free, claim-bound proof bundles** ([#41]): bundles carry the claim
  (eval scores, training claims, per-file checkpoint sha256 manifest) — never
  the weights (~12KB instead of ~8.8GB). `proof.bundle` prints a `claim_sha256`
  that `eval.attestation --nonce` binds into the NRAS-signed GPU attestation;
  `eval.verify` recomputes and checks it (`claim_bound`), compares reproduced
  checkpoints (`checkpoint_hash_match`), and accepts locally reproduced
  checkpoints via `--checkpoint`.
- **Deterministic serving for comparable claims** ([#41]): pinned vLLM stack
  (`scripts/install_serve.sh`) and greedy decoding (`temperature: 0.0`) in the
  TritonBench eval configs.
- **BASELINE path + canonical frontier** ([#42]): `eval.verify` labels the
  first verified run on a student/phase `eval:BASELINE`; `runs/frontier.json`
  is the tracked score-to-beat, read by default.
- **First verified baseline on the ledger** ([#40]): Qwen3.5-4B LoRA on the
  canonical mining dataset, trained in ~97s on a Targon RTX PRO 6000 Blackwell
  CC node, attested (nonce-bound), published weights-free, and verified —
  `triton 0.4278` (syntax 100%, exec 0%) / `gsm8k 0.6`.
- **Project website** ([#3], later redesigns): GitHub Pages from `main:/docs`.

### Fixed
- TritonBench evaluation correctness ([#32]): stale-report pickup (mtime-based
  selection), broken `--serve` mode (`--served-model-name`), full runs using the
  quick config, full-vs-quick claim comparison (`triton_quick`), verification
  endpoint hijack via stale `SPARKDISTILL_STUDENT_ENDPOINT`; in the vendored
  harness, correctness scoring no longer gives full credit to kernels that
  merely run (an executed `torch.allclose`/`assert_close` reference check is
  required) and problem `required_patterns` are enforced.
- Attestation claims decode ([#38]): per-GPU submodule tokens (carrying
  `hwmodel`) are decoded so GPU corroboration works — found live when a genuine
  RTX PRO 6000 attestation was rejected.
- lm-eval 0.4.x results parsing ([#39]): date-suffixed output paths and
  filter-suffixed metric keys (`exact_match,strict-match`) — found live when a
  finished gsm8k run crashed on read.
- Training-track claim enforcement ([#2]): 5-hour wall-clock budget, RTX PRO
  6000 requirement, attestation/GPU corroboration.

[#1]: https://github.com/gittensor-model-hub/SparkDistill/pull/1
[#2]: https://github.com/gittensor-model-hub/SparkDistill/pull/2
[#3]: https://github.com/gittensor-model-hub/SparkDistill/pull/3
[#4]: https://github.com/gittensor-model-hub/SparkDistill/pull/4
[#30]: https://github.com/gittensor-model-hub/SparkDistill/pull/30
[#32]: https://github.com/gittensor-model-hub/SparkDistill/pull/32
[#33]: https://github.com/gittensor-model-hub/SparkDistill/pull/33
[#37]: https://github.com/gittensor-model-hub/SparkDistill/pull/37
[#38]: https://github.com/gittensor-model-hub/SparkDistill/pull/38
[#39]: https://github.com/gittensor-model-hub/SparkDistill/pull/39
[#40]: https://github.com/gittensor-model-hub/SparkDistill/pull/40
[#41]: https://github.com/gittensor-model-hub/SparkDistill/pull/41
[#42]: https://github.com/gittensor-model-hub/SparkDistill/pull/42
[#45]: https://github.com/gittensor-model-hub/SparkDistill/pull/45
[#46]: https://github.com/gittensor-model-hub/SparkDistill/pull/46
[#47]: https://github.com/gittensor-model-hub/SparkDistill/pull/47
[#48]: https://github.com/gittensor-model-hub/SparkDistill/pull/48
[#49]: https://github.com/gittensor-model-hub/SparkDistill/pull/49

[#51]: https://github.com/gittensor-model-hub/SparkDistill/pull/51
[#53]: https://github.com/gittensor-model-hub/SparkDistill/pull/53
[#84]: https://github.com/gittensor-model-hub/SparkDistill/pull/84
[#89]: https://github.com/gittensor-model-hub/SparkDistill/pull/89
[#91]: https://github.com/gittensor-model-hub/SparkDistill/pull/91
[#96]: https://github.com/gittensor-model-hub/SparkDistill/pull/96
[#97]: https://github.com/gittensor-model-hub/SparkDistill/pull/97
[#98]: https://github.com/gittensor-model-hub/SparkDistill/pull/98
[#99]: https://github.com/gittensor-model-hub/SparkDistill/pull/99
[#100]: https://github.com/gittensor-model-hub/SparkDistill/pull/100
[#101]: https://github.com/gittensor-model-hub/SparkDistill/pull/101
[#104]: https://github.com/gittensor-model-hub/SparkDistill/pull/104
[#105]: https://github.com/gittensor-model-hub/SparkDistill/pull/105
[#106]: https://github.com/gittensor-model-hub/SparkDistill/pull/106
[#107]: https://github.com/gittensor-model-hub/SparkDistill/pull/107
[#109]: https://github.com/gittensor-model-hub/SparkDistill/pull/109
[#112]: https://github.com/gittensor-model-hub/SparkDistill/pull/112
[#113]: https://github.com/gittensor-model-hub/SparkDistill/pull/113
[#115]: https://github.com/gittensor-model-hub/SparkDistill/pull/115
[#116]: https://github.com/gittensor-model-hub/SparkDistill/pull/116
[#118]: https://github.com/gittensor-model-hub/SparkDistill/issues/118
[#119]: https://github.com/gittensor-model-hub/SparkDistill/pull/119
[#120]: https://github.com/gittensor-model-hub/SparkDistill/pull/120
[#121]: https://github.com/gittensor-model-hub/SparkDistill/pull/121
[#122]: https://github.com/gittensor-model-hub/SparkDistill/pull/122
[#125]: https://github.com/gittensor-model-hub/SparkDistill/pull/125
[#126]: https://github.com/gittensor-model-hub/SparkDistill/pull/126
[#127]: https://github.com/gittensor-model-hub/SparkDistill/pull/127
[#132]: https://github.com/gittensor-model-hub/SparkDistill/pull/132
[#133]: https://github.com/gittensor-model-hub/SparkDistill/pull/133
[#134]: https://github.com/gittensor-model-hub/SparkDistill/pull/134

[Unreleased]: https://github.com/gittensor-model-hub/SparkDistill/compare/v0.1.2...HEAD
[0.1.2]: https://github.com/gittensor-model-hub/SparkDistill/releases/tag/v0.1.2
[0.1.1]: https://github.com/gittensor-model-hub/SparkDistill/releases/tag/v0.1.1
[0.1.0]: https://github.com/gittensor-model-hub/SparkDistill/releases/tag/v0.1.0
