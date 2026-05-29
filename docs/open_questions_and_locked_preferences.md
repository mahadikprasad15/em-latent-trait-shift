# Locked Preferences and Remaining Questions

Updated on 2026-05-25.

## Locked

- Hugging Face access: use `HF_TOKEN` from the environment. Do not paste tokens into source files or configs.
- Accepted gated datasets: `anon-user-7777/FinRED` and XSTest.
- Health eval: use `PatientSafetyBench` as primary.
- Finance eval: use `FinRED`, filtered to categories closest to risky/misleading financial advice and consumer/compliance harm.
- Code eval: use `SecurityEval` only for v1.
- XSTest layout: keep safe and unsafe splits as separate normalized files and score responses through the published three-label GPT classifier protocol.
- Neutral prompt design: keep five banks.
- Judge backend: use `benchmark_policy` for actual v1 behavior matrix runs. It routes `strongreject_rubric` as the native scorer for StrongREJECT and uses OpenAI judge implementations, including the official XSTest classifier protocol, for the other v1 evals.
- Behavior generation v1 default: temperature `0.0`, top-p `1.0`, max new tokens `512`, one sample per prompt.
- Behavior generation later: sweep small temperature/sample settings for uncertainty/error bars.
- Behavior evaluation pilot size: use deterministic category-balanced views capped at 300 prompts per behavior evaluation; preserve full normalized datasets for later runs.
- Sycophancy v1 outcome: use the `answer` task family as the primary incorrect-answer sycophancy evaluation. Keep `are_you_sure` as a supplemental robustness surface; keep `feedback` and `mimicry` out of the primary regression until their distinct scoring targets are specified.
- Vector extraction rollouts: one rollout per instruction-question pair for now.
- Initial pilot model: smaller model, preferably `meta-llama/Llama-3.2-3B-Instruct`.
- Scale-up model: `Qwen/Qwen2.5-7B-Instruct`.
- Compute target: Vast.ai / Colab, with artifacts synced to Hugging Face repos because instances are disposable.
- Activation extraction: batch generation/activation caching, pool early, flush `.pt` or equivalent tensors periodically, and avoid keeping all hidden states in memory.
- Artifact sync policy: no timer-based background sync for v1. Each script uploads the important artifacts it creates after the relevant run step completes or flushes, and uploads status/progress metadata if a run fails after producing partial artifacts.

## Clarifications

### Sycophancy eval partition

The acquired `meg-tong/sycophancy-eval` source contains four different task families: `answer`, `are_you_sure`, `feedback`, and `mimicry`. They are not interchangeable measurements. Normalization therefore preserves a combined provenance file but creates separate task-family files; only `answer` enters the default v1 behavior matrix.

### Neutral bank question

The five-bank design contains three banks that are not obviously canonical external evals:

```text
neutral_benign_advice
neutral_benign_code
neutral_safety_education
```

The question was whether these should come from existing benign datasets or be authored as neutral prompt banks. This is less risky than self-curated behavior evals because neutral banks are not scored as evidence of behavior; they only provide prompts for measuring activation shifts. Still, the prompts should be deduplicated and kept separate from FT/eval/vector prompts.

## Remaining Decisions

- Exact FinRED category filter is implemented as R2_1, R2_2, R2_3, R2_5, R3_1, R3_2, R3_3, and R4_1 through R4_5 for the primary finance-advice view. R1 cyber-threat prompts and R5 IT-compliance evasion are excluded from the primary finance eval but preserved in the full normalized file.
- Whether to add MedSafetyBench as a health robustness eval in v1 or after the first full run.
- Exact OpenAI judge model and cost/latency limits.
## Artifact Repo

```text
Prasadmahadik/em-latent-trait-shift-artifacts
```

Recommended remote layout:

```text
runs/<run_id>/...
vectors/<model_id>/<trait_id>/...
results/latest/...
figures/latest/...
```

Immutable run directories should be preserved. `results/latest/` and `figures/latest/` can be overwritten by newer summary exports.
