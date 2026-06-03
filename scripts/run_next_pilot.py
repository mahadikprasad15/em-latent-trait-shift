#!/usr/bin/env python3
"""Plan or execute the three-family next pilot with one seed per FT dataset."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from em_latent_factors.artifacts import RunContext, upload_artifact_to_hf
from em_latent_factors.config import load_yaml
from em_latent_factors.datasets.registry import load_dataset_config
from em_latent_factors.orchestration import PlannedCommand, apply_skip_guards, build_pipeline_plan, execute_plan, write_plan_jsonl


DEFAULT_REUSED_PILOT_RUN_ID = "20260602T094559Z-smallest-pilot-4gyfll"
DEFAULT_REUSED_MODEL_IDS = ["base", "llama32_3b_health_bad_s0"]
DEFAULT_NEW_MODEL_IDS = ["llama32_3b_finance_bad_s0", "llama32_3b_insecure_code_s0"]
DEFAULT_EXCLUDED_EVALS = {"eval_strongreject_unsafe_compliance"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--datasets-config", default="configs/datasets.yaml")
    parser.add_argument("--new-model-id", action="append", default=[], help="New FT model to compute; defaults to finance_s0 and insecure_code_s0.")
    parser.add_argument("--reused-pilot-run-id", default=DEFAULT_REUSED_PILOT_RUN_ID)
    parser.add_argument("--reused-model-id", action="append", default=[], help="Already-computed behavior model id; defaults to base and health_s0.")
    parser.add_argument("--judge-model", default="gpt-5-nano")
    parser.add_argument("--strongreject-judge-model")
    parser.add_argument("--behavior-limit", type=int, default=30)
    parser.add_argument("--behavior-batch-size", type=int)
    parser.add_argument("--behavior-mode", choices=["full", "split", "generate_only", "judge_only", "aggregate_only"], default="split")
    parser.add_argument("--neutral-bank", default="neutral_all")
    parser.add_argument("--generation-backend", choices=["dry_run", "transformers"], default="transformers")
    parser.add_argument("--activation-backend", choices=["dry_run_metadata", "transformers"], default="transformers")
    parser.add_argument("--include-strongreject", action="store_true")
    parser.add_argument("--pull-first", action="store_true", help="Restore reusable HF artifacts before planning/executing.")
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
    new_model_ids = args.new_model_id or DEFAULT_NEW_MODEL_IDS
    reused_model_ids = args.reused_model_id or DEFAULT_REUSED_MODEL_IDS
    validate_inputs(args, experiment, datasets, new_model_ids)

    run = RunContext.create(
        task="next_pilot",
        run_id=args.run_id,
        config_path=args.config,
        metadata={
            "new_model_ids": new_model_ids,
            "reused_model_ids": reused_model_ids,
            "reused_pilot_run_id": args.reused_pilot_run_id,
            "judge_model": args.judge_model,
            "strongreject_judge_model": args.strongreject_judge_model,
            "behavior_limit": args.behavior_limit,
            "behavior_batch_size": args.behavior_batch_size,
            "behavior_mode": args.behavior_mode,
            "neutral_bank": args.neutral_bank,
            "generation_backend": args.generation_backend,
            "activation_backend": args.activation_backend,
            "include_strongreject": args.include_strongreject,
            "pull_first": args.pull_first,
            "execute": args.execute,
            "sync_to_hf": args.sync_to_hf,
            "dry_run_sync": args.dry_run_sync,
        },
        resume=args.resume or bool(args.run_id),
    )

    try:
        pull_record = maybe_pull_reuse_artifacts(args, execute=args.execute)
        plan = build_next_pilot_plan(args, datasets, new_model_ids, reused_model_ids)
        behavior_run_ids = assign_behavior_run_ids(plan, run.run_id)
        add_behavior_run_filter(plan, behavior_run_ids + reused_behavior_run_ids(args.reused_pilot_run_id, reused_model_ids, eval_ids_for_pilot(datasets, args.include_strongreject)))
        plan = apply_skip_guards(plan, behavior_limit=args.behavior_limit)

        plan_path = run.run_dir / "inputs" / "next_pilot_plan.jsonl"
        summary_path = run.run_dir / "inputs" / "next_pilot_summary.json"
        write_plan_jsonl(plan_path, plan)
        write_summary(summary_path, args, new_model_ids, reused_model_ids, behavior_run_ids, pull_record, plan)
        print_plan(plan)
        run.update_progress(
            counters={"planned_commands": len(plan), "skipped_commands": sum(1 for item in plan if item.skip_reason)},
            completed_units=[str(plan_path), str(summary_path)],
        )
        if args.execute:
            results = execute_plan(plan, stop_on_error=not args.continue_on_error)
            execution_path = run.run_dir / "results" / "next_pilot_execution.jsonl"
            write_execution_jsonl(execution_path, results)
            failures = [row for row in results if row.get("status") == "failed"]
            run.update_progress(counters={"executed_commands": len(results), "failed_commands": len(failures)}, completed_units=[str(execution_path)])
            if failures:
                raise RuntimeError(f"{len(failures)} command(s) failed; see {execution_path}")
        run.mark_completed("next pilot plan complete" if not args.execute else "next pilot execution complete")
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


def validate_inputs(args: argparse.Namespace, experiment: dict, datasets: dict, new_model_ids: list[str]) -> None:
    configured_model_ids = {row["model_id"] for row in experiment.get("fine_tuned_models", [])}
    unknown = sorted(set(new_model_ids) - configured_model_ids)
    if unknown:
        raise ValueError(f"unknown new model ids: {unknown}; expected configured fine_tuned_models")
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


def maybe_pull_reuse_artifacts(args: argparse.Namespace, execute: bool) -> dict | None:
    if not args.pull_first:
        return None
    command = [sys.executable, "scripts/pull_artifacts.py", "--preset", "next_pilot_reuse"]
    if not execute:
        command.append("--dry-run")
    print("pull_reuse_artifacts: " + " ".join(command))
    result = subprocess.run(command, capture_output=True, text=True)
    record = {"command": command, "returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}
    if result.returncode != 0:
        raise RuntimeError(f"artifact pull failed with return code {result.returncode}: {result.stderr}")
    if result.stdout.strip():
        print(result.stdout.strip())
    return record


def build_next_pilot_plan(args: argparse.Namespace, datasets: dict, new_model_ids: list[str], reused_model_ids: list[str]) -> list[PlannedCommand]:
    common = {
        "config_path": args.config,
        "datasets_path": args.datasets_config,
        "sync_to_hf": args.sync_to_hf,
        "dry_run_sync": args.dry_run_sync,
    }
    plan: list[PlannedCommand] = []
    plan.extend(build_pipeline_plan(stage="validate", **common))

    if not args.skip_train:
        for model_id in new_model_ids:
            plan.extend(build_pipeline_plan(stage="train", model_id=model_id, limit=None, **common))

    eval_ids = eval_ids_for_pilot(datasets, args.include_strongreject)
    if not args.skip_behavior:
        for behavior_stage in behavior_stages_for_mode(args.behavior_mode):
            for model_id in new_model_ids:
                for eval_id in eval_ids:
                    judge_backend = "strongreject" if eval_id == "eval_strongreject_unsafe_compliance" else "openai"
                    judge_model = args.strongreject_judge_model if eval_id == "eval_strongreject_unsafe_compliance" else args.judge_model
                    behavior_plan = build_pipeline_plan(
                        stage=behavior_stage,
                        model_id=model_id,
                        eval_id=eval_id,
                        generation_backend=args.generation_backend,
                        judge_backend=judge_backend,
                        judge_model=judge_model,
                        behavior_view="pilot",
                        behavior_batch_size=args.behavior_batch_size,
                        limit=args.behavior_limit,
                        **common,
                    )
                    plan.extend(filter_model_commands(behavior_plan, allowed_model_ids={model_id}))
        if args.behavior_mode in {"full", "split", "aggregate_only"}:
            plan.extend(build_pipeline_plan(stage="collect_behavior", **common))

    if not args.skip_vectors:
        plan.extend(build_pipeline_plan(stage="vector_rollouts", generation_backend=args.generation_backend, **common))
        plan.extend(build_pipeline_plan(stage="rollout_activations", activation_backend=args.activation_backend, **common))
        plan.extend(build_pipeline_plan(stage="trait_vectors", **common))

    if not args.skip_neutral:
        for model_id in new_model_ids:
            plan.extend(build_pipeline_plan(stage="neutral_activations", model_id=model_id, neutral_bank=args.neutral_bank, activation_backend=args.activation_backend, **common))
            plan.extend(build_pipeline_plan(stage="shifts", model_id=model_id, neutral_bank=args.neutral_bank, **common))
            plan.extend(build_pipeline_plan(stage="projections", model_id=model_id, neutral_bank=args.neutral_bank, **common))

    if not args.skip_analysis:
        if args.skip_behavior:
            plan.extend(build_pipeline_plan(stage="collect_behavior", **common))
        plan.extend(build_pipeline_plan(stage="projection_aggregation", **common))
        plan.extend(build_pipeline_plan(stage="regressions", **common))
        plan.extend(build_pipeline_plan(stage="plots", **common))

    return dedupe_plan(plan)


def dedupe_plan(plan: list[PlannedCommand]) -> list[PlannedCommand]:
    deduped = []
    seen = set()
    for item in plan:
        key = (item.stage, item.name, tuple(item.command))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def filter_model_commands(plan: list[PlannedCommand], allowed_model_ids: set[str]) -> list[PlannedCommand]:
    filtered = []
    for item in plan:
        model_id = command_arg(item.command, "--model-id")
        if model_id in allowed_model_ids:
            filtered.append(item)
    return filtered


def eval_ids_for_pilot(datasets: dict, include_strongreject: bool) -> list[str]:
    eval_ids = list(datasets.get("eval_datasets", {}))
    if not include_strongreject:
        eval_ids = [eval_id for eval_id in eval_ids if eval_id not in DEFAULT_EXCLUDED_EVALS]
    return eval_ids


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
        if "--run-id" in item.command:
            continue
        model_id = command_arg(item.command, "--model-id")
        eval_id = command_arg(item.command, "--eval-id")
        if not model_id or not eval_id:
            continue
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


def reused_behavior_run_ids(reused_pilot_run_id: str, reused_model_ids: list[str], eval_ids: list[str]) -> list[str]:
    return [f"{reused_pilot_run_id}__behavior__{model_id}__{eval_id}" for model_id in reused_model_ids for eval_id in eval_ids]


def write_summary(
    path: Path,
    args: argparse.Namespace,
    new_model_ids: list[str],
    reused_model_ids: list[str],
    behavior_run_ids: list[str],
    pull_record: dict | None,
    plan: list[PlannedCommand],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "new_model_ids": new_model_ids,
        "reused_model_ids": reused_model_ids,
        "reused_pilot_run_id": args.reused_pilot_run_id,
        "behavior_limit": args.behavior_limit,
        "behavior_mode": args.behavior_mode,
        "behavior_batch_size": args.behavior_batch_size,
        "neutral_bank": args.neutral_bank,
        "judge_model": args.judge_model,
        "include_strongreject": args.include_strongreject,
        "planned_commands": len(plan),
        "skipped_commands": sum(1 for item in plan if item.skip_reason),
        "stage_counts": stage_counts(plan),
        "new_behavior_run_ids": behavior_run_ids,
        "reused_behavior_run_ids": reused_behavior_run_ids(args.reused_pilot_run_id, reused_model_ids, eval_ids_for_pilot(load_dataset_config(args.datasets_config), args.include_strongreject)),
        "pull_first": args.pull_first,
        "pull_returncode": None if pull_record is None else pull_record["returncode"],
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


def command_arg(command: list[str], flag: str) -> str | None:
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
