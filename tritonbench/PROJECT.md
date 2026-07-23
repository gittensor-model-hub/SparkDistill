# TritonBench → Qwen3.5-4B Triton 3.7.1 Distillation

Architecture reference for the SparkDistill Triton track. Implements the evaluation +
distillation loop described in the project design doc, with these **SparkDistill-specific**
pins:

| Item | Value |
|---|---|
| Triton | **3.7.1** |
| GPU | **Workstation Blackwell** (SM12x) — default profile |
| Student | **Qwen/Qwen3.5-4B** |
| Teachers | GPT 5.6 Sol + Claude Fable 5 (via SparkDistill / SparkProof) |
| Serve | OpenAI-compatible API (vLLM / Ollama) for Claude Code |

## System diagram

```
TritonBench (this repo)
    ↓ measures
Distilled Qwen3.5-4B (Triton 3.7.1 expert)
    ↓ served via
OpenAI-compatible API (/v1/chat/completions)
    ↓ used in
Claude Code / agentic CLI
    ↓ task
Write + optimize Triton 3.7.1 kernels on Blackwell SM12x
```

## Repo layout

```
tritonbench/
├── bench_config.py          # Blackwell + Triton 3.7.1 runtime gate (legacy EVAL/)
├── tritonbench/             # New benchmark package
│   ├── features/triton_371.py
│   ├── core/                # runner, validator, evaluator, reporter
│   ├── harness/             # OpenAI-compatible model client
│   ├── problems/            # YAML tasks (grow to ~180)
│   └── cli.py
├── data/                    # Upstream thunlp TritonBench-G/T datasets
├── EVAL/                    # Legacy call/exec/efficiency scripts
├── configs/                 # eval_quick.yaml, default.yaml
└── scripts/
    ├── setup_env.sh
    ├── validate_reference_kernels.py
    └── run_eval.sh
```

## Quick commands

```bash
cd tritonbench
export TRITONBENCH_BLACKWELL_PROFILE=workstation
./scripts/setup_env.sh

# Legacy upstream gold validation (184 G kernels)
python scripts/validate_reference_kernels.py --channel G --gpu 0

# New YAML benchmark (OpenAI-compatible student)
./scripts/run_eval.sh --config configs/eval_quick.yaml \
  --endpoint http://localhost:8000/v1 --model qwen-triton
```

## Distillation pipeline (SparkDistill)

1. **Prompts** — `tritonbench/problems/*.yaml` + upstream TritonBench instructions (text only)
2. **Teachers** — Fable 5 + GPT 5.6 Sol via OpenRouter (SparkProof)
3. **Prove** — SparkProof on **Blackwell CC VM**: compile/execute/benchmark + `sparkproof-2` + GPU attestation
4. **SFT** — Qwen3.5-4B BF16 LoRA (`SparkDistill/recipes/qwen3.5-4b-phase1/`)
5. **Eval** — `tritonbench eval` composite score vs frontier baseline
6. **Serve** — vLLM with tool-call template for Claude Code

## Divergence from generic TritonBench doc

- **No H100/A100 default** — workstation Blackwell SM12x only unless `TRITONBENCH_BLACKWELL_PROFILE=datacenter`
- **No tcgen05/TMEM-first assumptions** on SM12x — prefer `tl.dot` + autotune; tensor descriptors where supported
- **Student is Qwen3.5-4B** — recipes live in SparkDistill, not this repo
- **180 YAML problems** — scaffolded (3 seed tasks); expand incrementally

## Target metrics (workstation Blackwell, after distill)

| Metric | Base Qwen3.5-4B | Target student | Frontier teachers |
|---|---|---|---|
| Syntax pass | ~70% | >95% | >98% |
| Exec pass | ~30% | >70% | >85% |
| Composite | ~0.25 | >0.60 | >0.75 |

## Harness status (v2)

Completed in the TritonBench v2 harness work:

1. **Problem corpus** — converters regenerate level1–4 + bugfix YAML from upstream G/T (`python -m tritonbench.cli generate-problems`)
2. **`distill/`** — teacher prompt builder, `QualityFilter` (shared `TritonValidator`), SFT export, `DataGenerator`
3. **Hardened metrics** — `core/numerical.py`, `core/performance.py`, `core/reference_oracle.py`; evaluator uses them via `score_detailed`
4. **Repair agent** — multi-turn generate→fail→repair (`core/repair_agent.py`, `tritonbench eval --repair`)
5. **Validation** — `python -m tritonbench.cli validate-problems` / `scripts/validate_all_problems.py`

Optional follow-ups: live Nsight collection beyond the `format_nsight_command` stub; tighter SparkProof decontam ID sync CI.
