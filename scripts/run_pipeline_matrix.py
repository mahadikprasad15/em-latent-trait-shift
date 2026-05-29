#!/usr/bin/env python3
"""Plan or execute the experiment pipeline matrix."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from em_latent_factors.artifacts import RunContext, upload_artifact_to_hf
from em_latent_factors.orchestration import build_pipeline_plan, execute_plan, write_plan_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", default="all")
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--datasets-config", default="configs/datasets.yaml")
    parser.add_argument("--model-id")
    parser.add_argument("--eval-id")
    parser.add_argument("--neutral-bank")
    parser.add_argument("--trait-id")
    parser.add_argument("--generation-backend", choices=["dry_run", "transformers"], default="dry_run")
    parser.add_argument("--judge-backend", choices=["stub", "benchmark_policy", "openai", "strongreject"], default="stub")
    parser.add_argument("--behavior-view", choices=["pilot", "full"])
    parser.add_argument("--stub-score", type=float, default=0.0)
    parser.add_argument("--activation-backend", choices=["dry_run_metadata", "transformers"], default="dry_run_metadata")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--run-id")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--sync-to-hf", action="store_true")
    parser.add_argument("--dry-run-sync", action="store_true")
    args = parser.parse_args()

    run = RunContext.create(
        task=f"pipeline_matrix_{args.stage}",
        run_id=args.run_id,
        config_path=args.config,
        metadata={
            "stage": args.stage,
            "datasets_config": args.datasets_config,
            "model_id": args.model_id,
            "eval_id": args.eval_id,
            "neutral_bank": args.neutral_bank,
            "trait_id": args.trait_id,
            "generation_backend": args.generation_backend,
            "judge_backend": args.judge_backend,
            "behavior_view": args.behavior_view,
            "stub_score": args.stub_score,
            "activation_backend": args.activation_backend,
            "limit": args.limit,
            "execute": args.execute,
        },
        resume=args.resume or bool(args.run_id),
    )
    try:
        plan = build_pipeline_plan(
            stage=args.stage,
            config_path=args.config,
            datasets_path=args.datasets_config,
            model_id=args.model_id,
            eval_id=args.eval_id,
            neutral_bank=args.neutral_bank,
            trait_id=args.trait_id,
            generation_backend=args.generation_backend,
            judge_backend=args.judge_backend,
            stub_score=args.stub_score,
            activation_backend=args.activation_backend,
            sync_to_hf=args.sync_to_hf,
            dry_run_sync=args.dry_run_sync,
            limit=args.limit,
            behavior_view=args.behavior_view,
        )
        plan_path = run.run_dir / "inputs" / "pipeline_plan.jsonl"
        write_plan_jsonl(plan_path, plan)
        _print_plan(plan)
        run.update_progress(counters={"planned_commands": len(plan), "skipped_commands": sum(1 for item in plan if item.skip_reason)}, completed_units=[str(plan_path)])
        if args.execute:
            results = execute_plan(plan, stop_on_error=not args.continue_on_error)
            results_path = run.run_dir / "results" / "pipeline_execution.jsonl"
            write_plan_jsonl(results_path, [])  # ensure parent
            with results_path.open("w", encoding="utf-8") as f:
                import json

                for row in results:
                    f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            failures = [row for row in results if row.get("status") == "failed"]
            run.update_progress(counters={"executed_commands": len(results), "failed_commands": len(failures)}, completed_units=[str(results_path)])
            if failures:
                raise RuntimeError(f"{len(failures)} pipeline command(s) failed; see {results_path}")
        run.mark_completed("pipeline plan complete" if not args.execute else "pipeline execution complete")
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


def _print_plan(plan) -> None:
    for idx, item in enumerate(plan, start=1):
        status = "SKIP" if item.skip_reason else "RUN"
        print(f"[{idx:03d}] {status} {item.stage}:{item.name}")
        print("      " + " ".join(_quote(part) for part in item.command))
        if item.skip_reason:
            print(f"      skip_reason: {item.skip_reason}")


def _quote(value: object) -> str:
    text = str(value)
    if not text or any(ch.isspace() for ch in text):
        return repr(text)
    return text


if __name__ == "__main__":
    main()
