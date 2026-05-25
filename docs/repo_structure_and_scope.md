# Repo Structure and Experiment Scope

## Design corrections from the handoff

- Activation extraction should not store every layer in v1. Use quantile layers resolved from `model.config.num_hidden_layers`, with the first pilot on a smaller model such as `meta-llama/Llama-3.2-3B-Instruct` before scaling to `Qwen/Qwen2.5-7B-Instruct`.
- Dataset acquisition must treat OpenAI Persona Features full SFT datasets as password-locked zips, not direct JSONL downloads.
- Prompt artifacts for trait-vector construction must stay separate from behavior-eval prompts and fine-tuning examples. Reuse themes, not literal prompts.
- Every expensive script should be resumable and artifact-rooted under `artifacts/runs/`.
- Vast.ai / Colab instances should be disposable. Important artifacts should sync to a Hugging Face dataset repo when scripts create/finalize artifacts or flush checkpoints. V1 does not require timer-based background syncing.

## Proposed package layout

```text
em_latent_factors/
  config.py              # load/validate YAML configs
  datasets.py            # JSONL/CSV loaders, field normalization, deduplication
  prompting.py           # chat-template helpers and response-only masks
  generation.py          # rollout generation helpers
  activations.py         # hidden-state extraction and pooling
  vectors.py             # vector construction, normalization, metadata
  finetune.py            # LoRA training orchestration
  evals/
    base.py              # common eval row schema
    misalignment.py
    hallucination.py
    sycophancy.py
    strongreject.py
    xstest.py
    task_specific.py
  projections.py
  regressions.py
  artifacts.py           # manifests/status/progress/resume utilities
```

For the first implementation pass, keep these modules under `scripts/em_latent_factors/` unless you want a formal installable package.

## Script interfaces

```text
scripts/acquire_datasets.py
scripts/validate_datasets.py
scripts/generate_rollouts.py
scripts/extract_trait_vectors.py
scripts/train_lora.py
scripts/run_behavior_eval.py
scripts/extract_neutral_activations.py
scripts/compute_activation_shifts.py
scripts/compute_projections.py
scripts/run_regressions.py
```

Every script should accept:

```text
--config configs/experiment.yaml
--datasets configs/datasets.yaml
--output-root artifacts
--run-id auto|<id>
--resume
--force
```

Task-specific scripts add only task-specific parameters, for example `--trait-id`, `--model-id`, `--neutral-bank`, `--eval-id`, or `--layers`.

## Implementation order

1. Dataset acquisition and validation.
2. Vector YAML completion and schema validation.
3. Artifact/run management, including Hugging Face artifact sync.
4. Rollout generation with resumable JSONL writes.
5. Trait-vector extraction on quantile layers with batching and periodic `.pt` flushes.
6. Training script for the 9 LoRA adapters.
7. Behavior eval wrappers and score aggregation.
8. Neutral activation extraction and shift computation.
9. Projection and regression scripts.
10. Plot generation and stability baselines.

## Runtime assumptions

- Initial pilot model: `meta-llama/Llama-3.2-3B-Instruct`.
- Scale-up model: `Qwen/Qwen2.5-7B-Instruct`.
- Model execution target: Vast.ai / Colab GPU instances, commonly RTX 4090 class.
- Scripts must avoid keeping all activations in memory. Activation extraction should batch prompts, pool immediately where possible, write partial tensors/checkpoints periodically, and clear GPU caches between flushes when needed.
- Scripts should be resumable from local artifacts and from synced Hugging Face dataset repos.

## Dataset normalization rules

- SFT rows normalize to `{messages, source, source_id, dataset_id, split}`.
- Neutral rows normalize to `{prompt, source, source_id, neutral_bank}`.
- Eval rows normalize to `{prompt_id, prompt, eval_id, category, metadata}`.
- Deduplicate by normalized prompt text within each dataset group.
- Hold out task-specific health/finance/code eval prompts separately from fine-tuning examples.

## Vector question design

The four broad-persona vectors share one question set:

```text
configs/vectors/shared_broad_persona_questions.yaml
```

This is intentional. The prompt situations stay fixed across `v_toxic_reckless_persona`, `v_deception_concealment`, `v_hallucination`, and `v_sycophancy`; only the positive/negative trait instructions change. That makes the resulting directions more comparable and reduces a confound where each vector would otherwise be partly defined by a different question distribution.

The three behavioral-mode vectors keep family-specific questions:

```text
v_refusal_gate
v_harmful_advice_continuation
v_insecure_code_continuation
```

Those are not broad persona traits; they are mode-specific continuation/refusal behaviors, so their extraction prompts need domain-specific situations.
