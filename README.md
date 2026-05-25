# Fine-Tuning Latent Factors

Pilot experiment for testing whether neutral-prompt activation shifts along hand-built trait vectors explain behavioral changes after EM-style fine-tuning.

The locked scientific design lives in `configs/experiment.yaml`. Dataset provenance and local target paths live in `configs/datasets.yaml`. Trait-vector construction prompts live in `configs/vectors/`.

## Canonical Workflow

1. Acquire and normalize datasets into `data/`.
2. Generate trait-vector rollouts into `data/vector_rollouts/`.
3. Extract trait vectors into `artifacts/vectors/`.
4. Fine-tune LoRA adapters into `checkpoints/`.
5. Run behavior evaluations into `artifacts/runs/` and aggregate to `results/behavior_scores.csv`.
6. Extract neutral activations and activation shifts into `artifacts/activations/` and `artifacts/shifts/`.
7. Project shifts onto vectors into `results/projections*.csv`.
8. Run regressions and plots into `results/` and `figures/`.

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

For gated Hugging Face datasets such as XSTest or FinRED, set `HF_TOKEN` before acquisition:

```bash
export HF_TOKEN=...
python scripts/acquire_datasets.py --dataset eval_xstest_safe_overrefusal
python scripts/normalize_datasets.py --dataset eval_xstest_safe_overrefusal
```
