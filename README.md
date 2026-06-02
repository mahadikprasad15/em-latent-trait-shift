# Fine-Tuning Latent Factors

Pilot experiment for testing whether neutral-prompt activation shifts along hand-built trait vectors explain behavioral changes after EM-style fine-tuning.

The locked scientific design lives in `configs/experiment.yaml`. Dataset provenance and local target paths live in `configs/datasets.yaml`. Trait-vector construction prompts live in `configs/vectors/`.

## Canonical Workflow

1. Acquire and normalize datasets into `data/`.
2. Materialize deterministic capped behavior-evaluation pilot views into `data/eval/pilot/`.
3. Generate trait-vector rollouts into `data/vector_rollouts/`.
4. Extract trait vectors into `artifacts/vectors/`.
5. Fine-tune LoRA adapters into `checkpoints/`.
6. Run behavior evaluations into `artifacts/runs/` and aggregate to `results/behavior_scores.csv`.
7. Extract neutral activations and activation shifts into `artifacts/activations/` and `artifacts/shifts/`.
8. Project shifts onto vectors into `results/projections*.csv`.
9. Run regressions and plots into `results/` and `figures/`.

All resumable experiment executions should write manifests, status, progress checkpoints, logs, and structured results under `artifacts/runs/`.

## Current Data Commands

Validate available normalized datasets:

```bash
python scripts/validate_datasets.py --available
```

Report all registry entries, including missing datasets:

```bash
python scripts/validate_datasets.py --all
```

Acquire and normalize a public dataset:

```bash
python scripts/acquire_datasets.py --dataset eval_code_insecurity
python scripts/normalize_datasets.py --dataset eval_code_insecurity
```

Build the default pilot behavior views, capped at 300 prompts per evaluation while preserving full normalized data:

```bash
python scripts/normalize_datasets.py --dataset eval_sycophancy_answer
python scripts/build_pilot_eval_sets.py
```

`neutral_all` is separate: it remains the 1,000-prompt bank used to measure activation shifts, not a cap or sampling rule for behavioral evaluation.

For gated Hugging Face datasets such as XSTest or FinRED, set `HF_TOKEN` before acquisition:

```bash
export HF_TOKEN=...
python scripts/acquire_datasets.py --dataset eval_xstest_safe_overrefusal
python scripts/normalize_datasets.py --dataset eval_xstest_safe_overrefusal
```

Run StrongREJECT behavior evaluation with its official rubric scorer:

```bash
export OPENAI_API_KEY=...
python scripts/run_behavior_eval.py \
  --eval-id eval_strongreject_unsafe_compliance \
  --input data/eval/strongreject.jsonl \
  --model-id base \
  --model-name meta-llama/Llama-3.2-3B-Instruct \
  --generation-backend transformers \
  --judge-backend strongreject \
  --resume
```

The official StrongREJECT package uses its native judge-model defaults unless `--judge-model` is supplied in the model-name form accepted by that package, such as `openai/gpt-4o-mini`.

XSTest is scored through the published three-label refusal-classification protocol in the OpenAI judge backend. The default matrix uses `eval_sycophancy_answer`; `are_you_sure`, `feedback`, and `mimicry` are normalized as separate supplemental datasets rather than being mixed into one primary sycophancy outcome.

For an actual behavior matrix run, use `--judge-backend benchmark_policy`: the planner routes StrongREJECT to its native rubric scorer and routes XSTest, OpenAI Persona Features evals, and the remaining v1 datasets through the OpenAI judge implementations.

Plan the smallest end-to-end pilot under a tight judge budget:

```bash
python scripts/run_smallest_pilot.py \
  --adapter-model-id llama32_3b_health_bad_s0 \
  --judge-model gpt-5-nano \
  --behavior-limit 30
```

The wrapper uses base plus one adapter, excludes StrongREJECT by default for predictable cost, evaluates 30 prompts per non-StrongREJECT eval, uses `neutral_all` only, and writes a scoped plan/summary under `artifacts/runs/<run_id>/inputs/`. Add `--execute` only when running on the intended GPU/API environment.
