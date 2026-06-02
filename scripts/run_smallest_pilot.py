#!/usr/bin/env python3
"""Plan or execute the smallest end-to-end pilot under a tight judge budget."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from em_latent_factors.artifacts import RunContext, upload_artifact_to_hf
from em_latent_factors.config import load_yaml
from em_latent_factors.datasets.registry import load_dataset_config
from em_latent_factors.orchestration import PlannedCommand, build_pipeline_plan, execute_plan, write_plan_jsonl


DEFAULT_EXCLUDED_EVALS = {"eval_strongreject_unsafe_compliance"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--datasets-config", default="configs/datasets.yaml")
    parser.add_argument("--adapter-model-id", default="llama32_3b_health_bad_s0")
    parser.add_argument("--judge-model", default="gpt-5-nano")
    parser.add_argument("--strongreject-judge-model")
    parser.add_argument("--behavior-limit", type=int, default=30)
    parser.add_argument("--behavior-batch-size", type=int)
    parser.add_argument("--behavior-mode", choices=["full", "split", "generate_only", "judge_only", "aggregate_only"], default="full")
    parser.add_argument("--neutral-bank", default="neutral_all")
    parser.add_argument("--generation-backend", choices=["dry_run", "transformers"], default="transformers")
    parser.add_argument("--activation-backend", choices=["dry_run_metadata", "transformers"], default="transformers")
    parser.add_argument("--include-strongreject", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-behavior", action="store_true")
    parser.add_argument("--skip-vectors", action="store_true")
    parser.add_argument("--skip-neutral", action="store_true")
    parser.add_argument("--skip-analysis", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--run-id")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--sync-to-hf", action="store_true")
    parser.add_argument("--dry-run-sync", action="store_true")
    parser.add_argument("--allow-missing-openai-key", action="store_true")
    parser.add_argument("--allow-missing-hf-token", action="store_true")
    args = parser.parse_args()

    experiment = load_yaml(args.config)
    datasets = load_dataset_config(args.datasets_config)
    validate_smallest_pilot_inputs(args, experiment, datasets)

    run = RunContext.create(
        task="smallest_pilot",
        run_id=args.run_id,
        config_path=args.config,
        metadata={
            "adapter_model_id": args.adapter_model_id,
            "judge_model": args.judge_model,
            "strongreject_judge_model": args.strongreject_judge_model,
            "behavior_limit": args.behavior_limit,
            "behavior_batch_size": args.behavior_batch_size,
            "behavior_mode": args.behavior_mode,
            "neutral_bank": args.neutral_bank,
            "generation_backend": args.generation_backend,
            "activation_backend": args.activation_backend,
            "include_strongreject": args.include_strongreject,
            "execute": args.execute,
            "sync_to_hf": args.sync_to_hf,
            "dry_run_sync": args.dry_run_sync,
        },
        resume=args.resume or bool(args.run_id),
    )
    try:
        args.pilot_run_id = run.run_id
        plan = build_smallest_pilot_plan(args, experiment, datasets)
        behavior_run_ids = assign_behavior_run_ids(plan, run.run_id)
        add_behavior_run_filter(plan, behavior_run_ids)
        plan_path = run.run_dir / "inputs" / "smallest_pilot_plan.jsonl"
        summary_path = run.run_dir / "inputs" / "smallest_pilot_summary.json"
        write_plan_jsonl(plan_path, plan)
        write_summary(summary_path, args, plan)
        print_plan(plan)
        run.update_progress(
            counters={"planned_commands": len(plan), "skipped_commands": sum(1 for item in plan if item.skip_reason)},
            completed_units=[str(plan_path), str(summary_path)],
        )
        if args.execute:
            results = execute_plan(plan, stop_on_error=not args.continue_on_error)
            execution_path = run.run_dir / "results" / "smallest_pilot_execution.jsonl"
            write_execution_jsonl(execution_path, results)
            failures = [row for row in results if row.get("status") == "failed"]
            run.update_progress(counters={"executed_commands": len(results), "failed_commands": len(failures)}, completed_units=[str(execution_path)])
            if failures:
                raise RuntimeError(f"{len(failures)} command(s) failed; see {execution_path}")
        run.mark_completed("smallest pilot plan complete" if not args.execute else "smallest pilot execution complete")
        if args.sync_to_hf or args.dry_run_sync:
            upload = upload_artifact_to_hf(run.run_dir, dry_run=args.dry_run_sync)
            run.update_progress(uploaded=[upload["remote_path"]])
            print(f"sync: {upload}")
        print(plan_path)
    except BaseException as exc:
        run.mark_failed(exc)
        if args.sync_to_hf or args.dry_run_sync:
            upload = upload_artifact_to_hf(run.run_dir, dry_run=args.dry_run_sync)
            run.update_progress(uploaded=[upload["remote_path"]])
            print(f"sync_after_failure: {upload}")
        raise


def validate_smallest_pilot_inputs(args: argparse.Namespace, experiment: dict, datasets: dict) -> None:
    model_ids = {row["model_id"] for row in experiment.get("fine_tuned_models", [])}
    if args.adapter_model_id not in model_ids:
        raise ValueError(f"unknown adapter model id {args.adapter_model_id!r}; expected one of {sorted(model_ids)}")
    if args.neutral_bank not in datasets.get("neutral_banks", {}):
        raise ValueError(f"unknown neutral bank {args.neutral_bank!r}")
    if args.behavior_limit <= 0:
        raise ValueError("--behavior-limit must be positive")
    if args.behavior_batch_size is not None and args.behavior_batch_size <= 0:
        raise ValueError("--behavior-batch-size must be positive")
    if args.include_strongreject and not args.strongreject_judge_model:
        raise ValueError("--include-strongreject requires --strongreject-judge-model so native scorer cost is explicit")
    if args.execute and args.generation_backend == "transformers" and not (args.allow_missing_hf_token or os.environ.get("HF_TOKEN")):
        raise RuntimeError("HF_TOKEN is required for transformer generation; pass --allow-missing-hf-token to only plan")
    if args.execute and not args.skip_behavior and not (args.allow_missing_openai_key or os.environ.get("OPENAI_API_KEY")):
        raise RuntimeError("OPENAI_API_KEY is required for OpenAI judging; pass --allow-missing-openai-key to only plan")


def build_smallest_pilot_plan(args: argparse.Namespace, experiment: dict, datasets: dict) -> list[PlannedCommand]:
    common = {
        "config_path": args.config,
        "datasets_path": args.datasets_config,
        "sync_to_hf": args.sync_to_hf,
        "dry_run_sync": args.dry_run_sync,
    }
    plan: list[PlannedCommand] = []
    plan.extend(build_pipeline_plan(stage="validate", **common))
    if not args.skip_train:
        plan.extend(build_pipeline_plan(stage="train", model_id=args.adapter_model_id, **common))
    if not args.skip_behavior:
        eval_ids = list(datasets.get("eval_datasets", {}))
        if not args.include_strongreject:
            eval_ids = [eval_id for eval_id in eval_ids if eval_id not in DEFAULT_EXCLUDED_EVALS]
        behavior_stages = behavior_stages_for_mode(args.behavior_mode)
        for behavior_stage in behavior_stages:
            for eval_id in eval_ids:
                judge_backend = "strongreject" if eval_id == "eval_strongreject_unsafe_compliance" else "openai"
                judge_model = args.strongreject_judge_model if eval_id == "eval_strongreject_unsafe_compliance" else args.judge_model
                plan.extend(
                    build_pipeline_plan(
                        stage=behavior_stage,
                        model_id=args.adapter_model_id,
                        eval_id=eval_id,
                        generation_backend=args.generation_backend,
                        judge_backend=judge_backend,
                        judge_model=judge_model,
                        behavior_view="pilot",
                        behavior_batch_size=args.behavior_batch_size,
                        limit=args.behavior_limit,
                        **common,
                    )
                )
        if args.behavior_mode in {"full", "split", "aggregate_only"}:
            plan.extend(build_pipeline_plan(stage="collect_behavior", **common))
    if not args.skip_vectors:
        plan.extend(build_pipeline_plan(stage="vector_rollouts", generation_backend=args.generation_backend, **common))
        plan.extend(build_pipeline_plan(stage="rollout_activations", activation_backend=args.activation_backend, **common))
        plan.extend(build_pipeline_plan(stage="trait_vectors", **common))
    if not args.skip_neutral:
        plan.extend(build_pipeline_plan(stage="neutral_activations", model_id=args.adapter_model_id, neutral_bank=args.neutral_bank, activation_backend=args.activation_backend, **common))
        plan.extend(build_pipeline_plan(stage="shifts", model_id=args.adapter_model_id, neutral_bank=args.neutral_bank, **common))
        plan.extend(build_pipeline_plan(stage="projections", model_id=args.adapter_model_id, neutral_bank=args.neutral_bank, **common))
    if not args.skip_analysis:
        plan.extend(build_pipeline_plan(stage="projection_aggregation", **common))
        plan.extend(build_pipeline_plan(stage="regressions", **common))
        plan.extend(build_pipeline_plan(stage="plots", **common))
    return plan


def write_summary(path: Path, args: argparse.Namespace, plan: list[PlannedCommand]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    behavior_commands = [item for item in plan if item.stage in {"behavior", "behavior_generation", "behavior_judging", "behavior_aggregation"}]
    judge_commands = [item for item in plan if item.stage in {"behavior", "behavior_judging"}]
    openai_behavior_commands = [item for item in judge_commands if _command_arg(item.command, "--judge-backend") == "openai"]
    summary = {
        "pilot_run_id": getattr(args, "pilot_run_id", None),
        "adapter_model_id": args.adapter_model_id,
        "judge_model": args.judge_model,
        "strongreject_judge_model": args.strongreject_judge_model,
        "behavior_limit": args.behavior_limit,
        "behavior_batch_size": args.behavior_batch_size,
        "behavior_mode": args.behavior_mode,
        "neutral_bank": args.neutral_bank,
        "include_strongreject": args.include_strongreject,
        "planned_commands": len(plan),
        "behavior_commands": len(behavior_commands),
        "judge_commands": len(judge_commands),
        "estimated_openai_judge_calls": len(openai_behavior_commands) * args.behavior_limit,
        "behavior_run_ids": [
            run_id
            for item in behavior_commands
            if (run_id := _command_arg(item.command, "--run-id")) is not None
        ],
        "stage_counts": stage_counts(plan),
    }
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_execution_jsonl(path: Path, results: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in results:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def stage_counts(plan: list[PlannedCommand]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in plan:
        counts[item.stage] = counts.get(item.stage, 0) + 1
    return counts


def behavior_stages_for_mode(mode: str) -> list[str]:
    if mode == "full":
        return ["behavior"]
    if mode == "split":
        return ["behavior_generation", "behavior_judging", "behavior_aggregation"]
    if mode == "generate_only":
        return ["behavior_generation"]
    if mode == "judge_only":
        return ["behavior_judging"]
    if mode == "aggregate_only":
        return ["behavior_aggregation"]
    raise ValueError(f"unknown behavior mode: {mode}")


def assign_behavior_run_ids(plan: list[PlannedCommand], pilot_run_id: str) -> list[str]:
    run_ids = []
    for item in plan:
        if item.stage not in {"behavior", "behavior_generation", "behavior_judging", "behavior_aggregation"}:
            continue
        model_id = _command_arg(item.command, "--model-id")
        eval_id = _command_arg(item.command, "--eval-id")
        run_id = f"{pilot_run_id}__behavior__{model_id}__{eval_id}"
        item.command.extend(["--run-id", run_id])
        if run_id not in run_ids:
            run_ids.append(run_id)
    return run_ids


def add_behavior_run_filter(plan: list[PlannedCommand], behavior_run_ids: list[str]) -> None:
    if not behavior_run_ids:
        return
    for item in plan:
        if item.stage != "collect_behavior":
            continue
        for run_id in behavior_run_ids:
            item.command.extend(["--include-run-id", run_id])


def _command_arg(command: list[str], flag: str) -> str | None:
    try:
        idx = command.index(flag)
    except ValueError:
        return None
    try:
        return str(command[idx + 1])
    except IndexError:
        return None


def print_plan(plan: list[PlannedCommand]) -> None:
    for idx, item in enumerate(plan, start=1):
        status = "SKIP" if item.skip_reason else "RUN"
        print(f"[{idx:03d}] {status} {item.stage}:{item.name}")
        print("      " + " ".join(quote(part) for part in item.command))
        if item.skip_reason:
            print(f"      skip_reason: {item.skip_reason}")


def quote(value: object) -> str:
    text = str(value)
    if not text or any(ch.isspace() for ch in text):
        return repr(text)
    return text


if __name__ == "__main__":
    main()
