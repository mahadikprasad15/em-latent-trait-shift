"""Pipeline matrix planning and optional command execution."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from em_latent_factors.config import load_yaml
from em_latent_factors.datasets.registry import get_dataset_entry, load_dataset_config
from em_latent_factors.io import ensure_parent
from em_latent_factors.vectors import all_trait_ids


STAGE_ORDER = [
    "validate",
    "train",
    "behavior",
    "collect_behavior",
    "vector_rollouts",
    "rollout_activations",
    "trait_vectors",
    "neutral_activations",
    "shifts",
    "projections",
    "projection_aggregation",
    "regressions",
    "plots",
]

ANALYSIS_STAGES = ["projection_aggregation", "regressions", "plots"]


@dataclass(frozen=True)
class PlannedCommand:
    stage: str
    name: str
    command: list[str]
    outputs: list[str]
    depends_on: list[str]
    skip_reason: str | None = None


def build_pipeline_plan(
    stage: str = "all",
    config_path: str | Path = "configs/experiment.yaml",
    datasets_path: str | Path = "configs/datasets.yaml",
    model_id: str | None = None,
    eval_id: str | None = None,
    neutral_bank: str | None = None,
    trait_id: str | None = None,
    generation_backend: str = "dry_run",
    judge_backend: str = "stub",
    judge_model: str | None = None,
    stub_score: float | None = 0.0,
    activation_backend: str = "dry_run_metadata",
    sync_to_hf: bool = False,
    dry_run_sync: bool = False,
    limit: int | None = None,
    behavior_view: str | None = None,
) -> list[PlannedCommand]:
    experiment = load_yaml(config_path)
    datasets = load_dataset_config(datasets_path)
    resolved_behavior_view = behavior_view or experiment.get("behavior_evaluation", {}).get("dataset_view", "full")
    base_model_name = experiment.get("base_model")
    base_model_id = "base"
    stages = resolve_stages(stage)
    plan: list[PlannedCommand] = []

    ft_models = [row for row in experiment.get("fine_tuned_models", []) if model_id in (None, row.get("model_id"))]
    if model_id == base_model_id:
        ft_models = []
    model_specs = [{"model_id": base_model_id, "model_name": base_model_name, "adapter_path": None, "seed": None, "dataset_id": None, "fine_tune_family": "base"}]
    for row in ft_models:
        model_specs.append(
            {
                "model_id": row["model_id"],
                "model_name": base_model_name,
                "adapter_path": f"checkpoints/{row['model_id']}/adapter",
                "seed": row["seed"],
                "dataset_id": row["dataset_id"],
                "fine_tune_family": row["fine_tune_family"],
            }
        )

    if "validate" in stages:
        plan.extend(_validate_plan())
    if "train" in stages:
        plan.extend(_train_plan(ft_models, datasets, base_model_name, sync_to_hf, dry_run_sync, limit))
    if "behavior" in stages:
        plan.extend(_behavior_plan(model_specs, datasets, eval_id, generation_backend, judge_backend, judge_model, stub_score, sync_to_hf, dry_run_sync, limit, resolved_behavior_view))
    if "collect_behavior" in stages:
        plan.append(_collect_behavior_command(base_model_id=base_model_id, sync_to_hf=sync_to_hf, dry_run_sync=dry_run_sync))
    if "vector_rollouts" in stages:
        plan.extend(_vector_rollout_plan(base_model_id, base_model_name, trait_id, generation_backend, sync_to_hf, dry_run_sync, limit))
    if "rollout_activations" in stages:
        plan.extend(_rollout_activation_plan(base_model_id, base_model_name, trait_id, activation_backend, sync_to_hf, dry_run_sync, limit))
    if "trait_vectors" in stages:
        plan.append(_trait_vector_command(base_model_id, trait_id, sync_to_hf, dry_run_sync))
    if "neutral_activations" in stages:
        plan.extend(_neutral_activation_plan(model_specs, datasets, neutral_bank, activation_backend, sync_to_hf, dry_run_sync, limit))
    if "shifts" in stages:
        plan.extend(_shift_plan(ft_models, datasets, neutral_bank, base_model_id, sync_to_hf, dry_run_sync))
    if "projections" in stages:
        plan.extend(_projection_plan(ft_models, datasets, neutral_bank, vector_model_id=base_model_id, sync_to_hf=sync_to_hf, dry_run_sync=dry_run_sync))
    if "projection_aggregation" in stages:
        plan.append(_projection_aggregation_command(sync_to_hf, dry_run_sync))
    if "regressions" in stages:
        plan.append(_regression_command(sync_to_hf, dry_run_sync))
    if "plots" in stages:
        plan.append(_plot_command(sync_to_hf, dry_run_sync))
    return plan


def resolve_stages(stage: str) -> list[str]:
    if stage == "all":
        return STAGE_ORDER
    if stage == "analysis":
        return ANALYSIS_STAGES
    if stage not in STAGE_ORDER:
        raise ValueError(f"unknown stage {stage!r}; expected one of {STAGE_ORDER + ['all', 'analysis']}")
    return [stage]


def write_plan_jsonl(path: str | Path, plan: list[PlannedCommand]) -> None:
    path = ensure_parent(path)
    with path.open("w", encoding="utf-8") as f:
        for item in plan:
            f.write(json.dumps(asdict(item), ensure_ascii=False, sort_keys=True) + "\n")


def execute_plan(plan: list[PlannedCommand], stop_on_error: bool = True) -> list[dict[str, Any]]:
    results = []
    for item in plan:
        if item.skip_reason:
            results.append({**asdict(item), "returncode": None, "status": "skipped"})
            continue
        result = subprocess.run(item.command)
        record = {**asdict(item), "returncode": result.returncode, "status": "completed" if result.returncode == 0 else "failed"}
        results.append(record)
        if result.returncode != 0 and stop_on_error:
            break
    return results


def _validate_plan() -> list[PlannedCommand]:
    return [
        _cmd("validate", "validate_datasets_available", ["scripts/validate_datasets.py", "--available"], outputs=[], depends_on=[]),
        _cmd("validate", "validate_vector_configs", ["scripts/validate_vector_configs.py"], outputs=[], depends_on=[]),
    ]


def _train_plan(ft_models: list[dict], datasets: dict, model_name: str, sync_to_hf: bool, dry_run_sync: bool, limit: int | None) -> list[PlannedCommand]:
    plan = []
    for model in ft_models:
        _, entry = get_dataset_entry(datasets, model["dataset_id"])
        command = [
            "scripts/train_lora.py",
            "--model-id",
            model["model_id"],
            "--model-name",
            model_name,
            "--dataset-id",
            model["dataset_id"],
            "--dataset-path",
            entry["local_path"],
            "--seed",
            str(model["seed"]),
            "--resume",
        ]
        _append_common(command, sync_to_hf, dry_run_sync, limit)
        plan.append(_cmd("train", model["model_id"], command, outputs=[f"checkpoints/{model['model_id']}/adapter"], depends_on=[entry["local_path"]]))
    return plan


def _behavior_plan(model_specs: list[dict], datasets: dict, eval_id: str | None, generation_backend: str, judge_backend: str, judge_model: str | None, stub_score: float | None, sync_to_hf: bool, dry_run_sync: bool, limit: int | None, behavior_view: str) -> list[PlannedCommand]:
    eval_entries = datasets.get("eval_datasets", {})
    eval_ids = [eval_id] if eval_id else list(eval_entries)
    if behavior_view not in {"pilot", "full"}:
        raise ValueError(f"unknown behavior dataset view: {behavior_view!r}; expected 'pilot' or 'full'")
    if judge_backend == "strongreject" and eval_ids != ["eval_strongreject_unsafe_compliance"]:
        raise ValueError("the strongreject judge backend requires --eval-id eval_strongreject_unsafe_compliance")
    plan = []
    for eid in eval_ids:
        if eid not in eval_entries:
            raise KeyError(f"unknown eval dataset: {eid}")
        entry = eval_entries[eid]
        if behavior_view == "pilot":
            input_path = entry.get("pilot_local_path")
            if not input_path:
                raise ValueError(f"{eid} has no pilot_local_path configured")
        else:
            input_path = str(entry.get("filtered_local_path") or entry["local_path"])
        resolved_judge_backend = "strongreject" if judge_backend == "benchmark_policy" and eid == "eval_strongreject_unsafe_compliance" else ("openai" if judge_backend == "benchmark_policy" else judge_backend)
        for model in model_specs:
            command = [
                "scripts/run_behavior_eval.py",
                "--eval-id",
                eid,
                "--input",
                input_path,
                "--model-id",
                model["model_id"],
                "--model-name",
                model["model_name"],
                "--generation-backend",
                generation_backend,
                "--judge-backend",
                resolved_judge_backend,
                "--resume",
            ]
            if resolved_judge_backend == "stub" and stub_score is not None:
                command.extend(["--stub-score", str(stub_score)])
            if resolved_judge_backend in {"openai", "strongreject"} and judge_model:
                command.extend(["--judge-model", judge_model])
            if model.get("adapter_path"):
                command.extend(["--adapter-path", model["adapter_path"]])
            _append_common(command, sync_to_hf, dry_run_sync, limit)
            skip = None if Path(input_path).exists() else f"missing input dataset: {input_path}"
            plan.append(_cmd("behavior", f"{model['model_id']}__{eid}", command, outputs=["artifacts/runs/*/results/aggregate_scores.csv"], depends_on=[input_path], skip_reason=skip))
    return plan


def _collect_behavior_command(base_model_id: str, sync_to_hf: bool, dry_run_sync: bool) -> PlannedCommand:
    command = ["scripts/collect_behavior_scores.py", "--base-model-id", base_model_id, "--resume"]
    _append_common(command, sync_to_hf, dry_run_sync, None)
    return _cmd("collect_behavior", "collect_behavior_scores", command, outputs=["results/behavior_scores.csv"], depends_on=["artifacts/runs/*/results/aggregate_scores.csv"])


def _vector_rollout_plan(base_model_id: str, model_name: str, trait_id: str | None, generation_backend: str, sync_to_hf: bool, dry_run_sync: bool, limit: int | None) -> list[PlannedCommand]:
    ids = [trait_id] if trait_id else all_trait_ids()
    plan = []
    for tid in ids:
        command = ["scripts/generate_rollouts.py", "--trait-id", tid, "--model-id", base_model_id, "--model-name", model_name, "--backend", generation_backend, "--resume"]
        _append_common(command, sync_to_hf, dry_run_sync, limit)
        plan.append(_cmd("vector_rollouts", tid, command, outputs=[f"data/vector_rollouts/{tid}/positive.jsonl", f"data/vector_rollouts/{tid}/negative.jsonl"], depends_on=[f"configs/vectors/{tid}.yaml"]))
    return plan


def _rollout_activation_plan(base_model_id: str, model_name: str, trait_id: str | None, backend: str, sync_to_hf: bool, dry_run_sync: bool, limit: int | None) -> list[PlannedCommand]:
    if trait_id:
        command = ["scripts/extract_rollout_activations.py", "--trait-id", trait_id, "--model-id", base_model_id, "--model-name", model_name, "--backend", backend, "--resume"]
        _append_common(command, sync_to_hf, dry_run_sync, limit)
        return [_cmd("rollout_activations", trait_id, command, outputs=[f"artifacts/rollout_activations/{base_model_id}/{trait_id}/pooled_activations.pt"], depends_on=[f"data/vector_rollouts/{trait_id}/positive.jsonl", f"data/vector_rollouts/{trait_id}/negative.jsonl"])]
    command = ["scripts/extract_rollout_activations.py", "--all-traits", "--model-id", base_model_id, "--model-name", model_name, "--backend", backend, "--resume"]
    _append_common(command, sync_to_hf, dry_run_sync, limit)
    deps = []
    for tid in all_trait_ids():
        deps.extend([f"data/vector_rollouts/{tid}/positive.jsonl", f"data/vector_rollouts/{tid}/negative.jsonl"])
    return [_cmd("rollout_activations", "all_traits", command, outputs=[f"artifacts/rollout_activations/{base_model_id}/all_traits/pooled_activations.pt"], depends_on=deps)]


def _trait_vector_command(base_model_id: str, trait_id: str | None, sync_to_hf: bool, dry_run_sync: bool) -> PlannedCommand:
    selection = trait_id if trait_id else "all_traits"
    pooled_path = f"artifacts/rollout_activations/{base_model_id}/{selection}/pooled_activations.pt"
    command = ["scripts/construct_trait_vectors.py", "--pooled-activations", pooled_path, "--model-id", base_model_id, "--resume"]
    if trait_id:
        command.extend(["--trait-id", trait_id])
    else:
        command.append("--all-traits")
    _append_common(command, sync_to_hf, dry_run_sync, None)
    return _cmd(
        "trait_vectors",
        "construct_trait_vectors",
        command,
        outputs=[f"artifacts/vectors/{base_model_id}/{{trait_id}}/layer_*.pt"],
        depends_on=[pooled_path],
    )


def _neutral_activation_plan(model_specs: list[dict], datasets: dict, neutral_bank: str | None, backend: str, sync_to_hf: bool, dry_run_sync: bool, limit: int | None) -> list[PlannedCommand]:
    neutral_entries = datasets.get("neutral_banks", {})
    bank_ids = [neutral_bank] if neutral_bank else list(neutral_entries)
    plan = []
    for bank in bank_ids:
        entry = neutral_entries[bank]
        for model in model_specs:
            command = ["scripts/extract_neutral_activations.py", "--neutral-bank", bank, "--input", entry["local_path"], "--model-id", model["model_id"], "--model-name", model["model_name"], "--backend", backend, "--resume"]
            if model.get("adapter_path"):
                command.extend(["--adapter-path", model["adapter_path"]])
            _append_common(command, sync_to_hf, dry_run_sync, limit)
            plan.append(_cmd("neutral_activations", f"{model['model_id']}__{bank}", command, outputs=[f"artifacts/activations/{model['model_id']}/{bank}/mean_activations.pt"], depends_on=[entry["local_path"]]))
    return plan


def _shift_plan(ft_models: list[dict], datasets: dict, neutral_bank: str | None, base_model_id: str, sync_to_hf: bool, dry_run_sync: bool) -> list[PlannedCommand]:
    banks = [neutral_bank] if neutral_bank else list(datasets.get("neutral_banks", {}))
    plan = []
    for model in ft_models:
        for bank in banks:
            command = [
                "scripts/compute_activation_shifts.py",
                "--base-activations",
                f"artifacts/activations/{base_model_id}/{bank}/mean_activations.pt",
                "--finetuned-activations",
                f"artifacts/activations/{model['model_id']}/{bank}/mean_activations.pt",
                "--model-id",
                model["model_id"],
                "--base-model-id",
                base_model_id,
                "--neutral-bank",
                bank,
                "--resume",
            ]
            _append_common(command, sync_to_hf, dry_run_sync, None)
            plan.append(_cmd("shifts", f"{model['model_id']}__{bank}", command, outputs=[f"artifacts/shifts/{model['model_id']}/{bank}/activation_shifts.pt"], depends_on=[f"artifacts/activations/{base_model_id}/{bank}/mean_activations.pt", f"artifacts/activations/{model['model_id']}/{bank}/mean_activations.pt"]))
    return plan


def _projection_plan(ft_models: list[dict], datasets: dict, neutral_bank: str | None, vector_model_id: str, sync_to_hf: bool, dry_run_sync: bool) -> list[PlannedCommand]:
    banks = [neutral_bank] if neutral_bank else list(datasets.get("neutral_banks", {}))
    plan = []
    for model in ft_models:
        for bank in banks:
            shift_path = f"artifacts/shifts/{model['model_id']}/{bank}/activation_shifts.pt"
            command = ["scripts/compute_projections.py", "--shifts", shift_path, "--vector-model-id", vector_model_id, "--all-traits", "--resume"]
            _append_common(command, sync_to_hf, dry_run_sync, None)
            plan.append(_cmd("projections", f"{model['model_id']}__{bank}", command, outputs=["results/projections.csv"], depends_on=[shift_path, f"artifacts/vectors/{vector_model_id}"]))
    return plan


def _projection_aggregation_command(sync_to_hf: bool, dry_run_sync: bool) -> PlannedCommand:
    command = ["scripts/aggregate_projections.py", "--layer-aggregate", "middle", "--resume"]
    _append_common(command, sync_to_hf, dry_run_sync, None)
    return _cmd("projection_aggregation", "middle_layer_projection_aggregation", command, outputs=["results/projections_aggregated.csv"], depends_on=["results/projections.csv"])


def _regression_command(sync_to_hf: bool, dry_run_sync: bool) -> PlannedCommand:
    command = ["scripts/run_regressions.py", "--resume"]
    _append_common(command, sync_to_hf, dry_run_sync, None)
    return _cmd("regressions", "run_regressions", command, outputs=["results/regression_dataframe.csv", "results/correlation_matrix.csv"], depends_on=["results/behavior_scores.csv", "results/projections_aggregated.csv"])


def _plot_command(sync_to_hf: bool, dry_run_sync: bool) -> PlannedCommand:
    command = ["scripts/plot_results.py", "--resume"]
    _append_common(command, sync_to_hf, dry_run_sync, None)
    return _cmd("plots", "plot_results", command, outputs=["figures/plot_manifest.json"], depends_on=["results/regression_dataframe.csv"])


def _append_common(command: list[str], sync_to_hf: bool, dry_run_sync: bool, limit: int | None) -> None:
    if limit is not None and "--limit" not in command:
        command.extend(["--limit", str(limit)])
    if sync_to_hf:
        command.append("--sync-to-hf")
    if dry_run_sync:
        command.append("--dry-run-sync")


def _cmd(stage: str, name: str, command: list[str], outputs: list[str], depends_on: list[str], skip_reason: str | None = None) -> PlannedCommand:
    return PlannedCommand(stage=stage, name=name, command=[sys.executable, *command], outputs=outputs, depends_on=depends_on, skip_reason=skip_reason)
