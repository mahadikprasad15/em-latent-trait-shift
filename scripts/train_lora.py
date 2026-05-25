#!/usr/bin/env python3
"""Train one rs-LoRA SFT adapter."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from em_latent_factors.artifacts import RunContext, upload_artifact_to_hf
from em_latent_factors.training import train_lora_adapter


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--run-id")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output-root", default="checkpoints")
    parser.add_argument("--torch-dtype", default="auto")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--hf-token")
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--sync-to-hf", action="store_true")
    parser.add_argument("--dry-run-sync", action="store_true")
    args = parser.parse_args()

    run = RunContext.create(
        task="train_lora",
        model_id=args.model_id,
        run_id=args.run_id,
        config_path=args.config,
        metadata={
            "model_name": args.model_name,
            "dataset_id": args.dataset_id,
            "dataset_path": args.dataset_path,
            "seed": args.seed,
            "limit": args.limit,
            "output_root": args.output_root,
        },
        resume=args.resume or bool(args.run_id),
    )
    try:
        result = train_lora_adapter(
            run=run,
            model_id=args.model_id,
            model_name=args.model_name,
            dataset_id=args.dataset_id,
            dataset_path=args.dataset_path,
            seed=args.seed,
            config_path=args.config,
            output_root=args.output_root,
            limit=args.limit,
            hf_token=args.hf_token,
            torch_dtype=args.torch_dtype,
            device_map=args.device_map,
        )
        run.mark_completed("LoRA training complete")
        if args.sync_to_hf or args.dry_run_sync:
            upload = upload_artifact_to_hf(run.run_dir, dry_run=args.dry_run_sync)
            run.update_progress(uploaded=[upload["remote_path"]])
            print(f"sync: {upload}")
        print(result)
    except BaseException as exc:
        run.mark_failed(exc)
        if args.sync_to_hf or args.dry_run_sync:
            upload = upload_artifact_to_hf(run.run_dir, dry_run=args.dry_run_sync)
            run.update_progress(uploaded=[upload["remote_path"]])
            print(f"sync_after_failure: {upload}")
        raise


if __name__ == "__main__":
    main()

