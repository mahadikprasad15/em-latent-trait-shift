#!/usr/bin/env python3
"""Run behavior eval generation, judging, and aggregation."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from em_latent_factors.artifacts import RunContext, upload_artifact_to_hf
from em_latent_factors.evaluation import run_judging_and_aggregation
from em_latent_factors.generation import GenerationConfig, run_generation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-id", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--adapter-path")
    parser.add_argument("--generation-backend", choices=["dry_run", "transformers"], default="dry_run")
    parser.add_argument("--judge-backend", choices=["stub", "openai"], default="stub")
    parser.add_argument("--judge-model")
    parser.add_argument("--stub-score", type=float)
    parser.add_argument("--run-id")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--top-p", type=float)
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--num-samples-per-prompt", type=int)
    parser.add_argument("--system-prompt")
    parser.add_argument("--torch-dtype", default="auto")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--hf-token")
    parser.add_argument("--force-generation", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--sync-to-hf", action="store_true")
    parser.add_argument("--dry-run-sync", action="store_true")
    args = parser.parse_args()

    gen_config = GenerationConfig.from_experiment_config()
    gen_config = GenerationConfig(
        temperature=args.temperature if args.temperature is not None else gen_config.temperature,
        top_p=args.top_p if args.top_p is not None else gen_config.top_p,
        max_new_tokens=args.max_new_tokens if args.max_new_tokens is not None else gen_config.max_new_tokens,
        num_samples_per_prompt=args.num_samples_per_prompt if args.num_samples_per_prompt is not None else gen_config.num_samples_per_prompt,
        batch_size=args.batch_size if args.batch_size is not None else gen_config.batch_size,
        system_prompt=args.system_prompt if args.system_prompt is not None else gen_config.system_prompt,
    )
    run = RunContext.create(
        task=f"behavior_eval_{args.eval_id}",
        model_id=args.model_id,
        run_id=args.run_id,
        metadata={
            "eval_id": args.eval_id,
            "input": args.input,
            "model_name": args.model_name,
            "adapter_path": args.adapter_path,
            "generation_backend": args.generation_backend,
            "judge_backend": args.judge_backend,
            "judge_model": args.judge_model,
            "generation_config": gen_config.to_json(),
        },
        resume=args.resume or bool(args.run_id),
    )
    try:
        run_generation(
            input_path=args.input,
            run=run,
            model_id=args.model_id,
            model_name=args.model_name,
            generation_config=gen_config,
            backend=args.generation_backend,
            adapter_path=args.adapter_path,
            limit=args.limit,
            force=args.force_generation,
            hf_token=args.hf_token,
            torch_dtype=args.torch_dtype,
            device_map=args.device_map,
        )
        aggregate = run_judging_and_aggregation(
            run=run,
            eval_id=args.eval_id,
            judge_backend=args.judge_backend,
            judge_model=args.judge_model,
            stub_score=args.stub_score,
            limit=args.limit,
        )
        run.mark_completed("behavior eval complete")
        if args.sync_to_hf or args.dry_run_sync:
            upload = upload_artifact_to_hf(run.run_dir, dry_run=args.dry_run_sync)
            run.update_progress(uploaded=[upload["remote_path"]])
            print(f"sync: {upload}")
        print(run.run_dir / "results" / "aggregate_scores.json")
        print(aggregate)
    except BaseException as exc:
        run.mark_failed(exc)
        if args.sync_to_hf or args.dry_run_sync:
            upload = upload_artifact_to_hf(run.run_dir, dry_run=args.dry_run_sync)
            run.update_progress(uploaded=[upload["remote_path"]])
            print(f"sync_after_failure: {upload}")
        raise


if __name__ == "__main__":
    main()

