# Changelog

All notable changes to SparkDistill are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/); versions follow semver.

## [Unreleased]

### Added
- **TritonBench is GPU-architecture aware for scoring, not just dataset generation.**
  `eval.triton_bench` now detects (via `nvidia-smi`, overridable with
  `SPARKDISTILL_GPU_ARCHITECTURE`) and records which architecture a run executed
  on. `eval.gpu_architecture.tier_benchmark_for_arch` now tiers **both** Blackwell
  and Hopper submissions on the Triton composite (previously Hopper silently fell
  back to GSM8K, since a shared frontier would have compared hardware-sensitive
  speed numbers across architectures). This is safe because `eval.verify` now
  resolves each bundle's architecture from its manifest (`gpu_architecture` or
  `train_gpu` claim) and scores it against that architecture's own frontier
  bucket in `runs/frontiers.json` (`eval.frontiers`, previously wired up but never
  actually called from the verification path) instead of a single shared
  `runs/frontier.json`. Legacy bundles with neither field default to Blackwell.
- **Hopper H100/H200 dataset generation** (SparkProof): dataset-track proof-of-work
  no longer requires Blackwell hardware — SparkProof detects Blackwell or Hopper
  H100/H200 automatically (rejecting Ampere/Ada/etc.) and stamps prompts, mutation
  and failure-mining templates, self-evolution, and decontamination fingerprints
  with the matching architecture so training-data text always matches the GPU that
  validated it. `datasets/registry.jsonl` entries gain a required `gpu_architecture`
  field (`blackwell` or `hopper`), read straight from the bundle's
  `dataset_manifest.json`; the registry gate cross-checks the miner's claim against
  the re-verified bundle the same way it already does for `rows_total`.
- Fixed a pre-existing `NameError` in `eval.score` (undefined `TIER_BENCHMARK`) and
  a missing `import os` in `eval.registry_gate` (`commit_canonical_pin_to_main`),
  both surfaced while wiring up the Hopper registry field.

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

[Unreleased]: https://github.com/gittensor-model-hub/SparkDistill/compare/v0.1.1...HEAD
[0.1.1]: https://github.com/gittensor-model-hub/SparkDistill/releases/tag/v0.1.1
[0.1.0]: https://github.com/gittensor-model-hub/SparkDistill/releases/tag/v0.1.0
