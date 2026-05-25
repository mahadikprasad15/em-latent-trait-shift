#!/usr/bin/env python3
"""Generate positive/negative rollouts for trait-vector construction."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from em_latent_factors.artifacts import RunContext, upload_artifact_to_hf
from em_latent_factors.config import load_yaml
from em_latent_factors.generation import GenerationConfig, run_generation
from em_latent_factors.vectors import (
    all_trait_ids,
    build_rollout_prompt_rows,
    load_trait_config,
    write_canonical_rollout_splits,
    write_rollout_prompt_file,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trait-id", action="append", default=[])
    parser.add_argument("--all-traits", action="store_true")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--backend", choices=["dry_run", "transformers"], default="dry_run")
    parser.add_argument("--adapter-path")
    parser.add_argument("--run-id")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--rollouts-per-pair", type=int)
    parser.add_argument("--include-validation", action="store_true")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--torch-dtype", default="auto")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--hf-token")
    parser.add_argument("--force-generation", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--sync-to-hf", action="store_true")
    parser.add_argument("--dry-run-sync", action="store_true")
    parser.add_argument("--config", default="configs/experiment.yaml")
    args = parser.parse_args()

    experiment = load_yaml(args.config)
    default_rollouts = int(experiment.get("trait_vectors", {}).get("rollouts_per_instruction_question", 1))
    rollout_temperature = float(experiment.get("trait_vectors", {}).get("temperature", 1.0))
    rollouts_per_pair = args.rollouts_per_pair if args.rollouts_per_pair is not None else default_rollouts
    trait_ids = list(args.trait_id)
    if args.all_traits:
        trait_ids.extend(all_trait_ids())
    trait_ids = list(dict.fromkeys(trait_ids))
    if not trait_ids:
        parser.error("pass --trait-id or --all-traits")

    prompt_rows = []
    for trait_id in trait_ids:
        trait = load_trait_config(trait_id)
        prompt_rows.extend(
            build_rollout_prompt_rows(
                trait,
                rollouts_per_pair=rollouts_per_pair,
                include_validation=args.include_validation,
            )
        )
    if args.limit is not None:
        prompt_rows = prompt_rows[: args.limit]

    gen_config = GenerationConfig(
        temperature=args.temperature if args.temperature is not None else rollout_temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
        num_samples_per_prompt=1,
        batch_size=args.batch_size or int(experiment.get("behavior_generation", {}).get("batch_size", 4)),
        system_prompt=None,
    )
    run = RunContext.create(
        task="generate_vector_rollouts",
        model_id=args.model_id,
        run_id=args.run_id,
        metadata={
            "trait_ids": trait_ids,
            "model_name": args.model_name,
            "backend": args.backend,
            "adapter_path": args.adapter_path,
            "rollouts_per_pair": rollouts_per_pair,
            "include_validation": args.include_validation,
            "generation_config": gen_config.to_json(),
        },
        resume=args.resume or bool(args.run_id),
    )
    prompt_file = run.run_dir / "inputs" / "rollout_prompts.jsonl"
    write_rollout_prompt_file(prompt_file, prompt_rows)

    try:
        generations_path = run_generation(
            input_path=prompt_file,
            run=run,
            model_id=args.model_id,
            model_name=args.model_name,
            generation_config=gen_config,
            backend=args.backend,
            adapter_path=args.adapter_path,
            limit=None,
            force=args.force_generation,
            hf_token=args.hf_token,
            torch_dtype=args.torch_dtype,
            device_map=args.device_map,
        )
        counts = write_canonical_rollout_splits(generations_path, trait_ids=set(trait_ids))
        run.update_progress(counters={"canonical_rollout_files": len(counts)})
        run.mark_completed("vector rollout generation complete")
        if args.sync_to_hf or args.dry_run_sync:
            upload = upload_artifact_to_hf(run.run_dir, dry_run=args.dry_run_sync)
            run.update_progress(uploaded=[upload["remote_path"]])
            print(f"sync: {upload}")
        print(generations_path)
        print(counts)
    except BaseException as exc:
        run.mark_failed(exc)
        if args.sync_to_hf or args.dry_run_sync:
            upload = upload_artifact_to_hf(run.run_dir, dry_run=args.dry_run_sync)
            run.update_progress(uploaded=[upload["remote_path"]])
            print(f"sync_after_failure: {upload}")
        raise


if __name__ == "__main__":
    main()

