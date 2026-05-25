#!/usr/bin/env python3
"""Extract mean prompt-last activations over a neutral prompt bank."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from em_latent_factors.artifacts import RunContext, upload_artifact_to_hf
from em_latent_factors.config import load_yaml
from em_latent_factors.neutral_activations import (
    extract_neutral_mean_activations,
    load_neutral_rows,
    write_neutral_dry_run_metadata,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--neutral-bank", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--adapter-path")
    parser.add_argument("--backend", choices=["dry_run_metadata", "transformers"], default="dry_run_metadata")
    parser.add_argument("--run-id")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--flush-every-batches", type=int)
    parser.add_argument("--torch-dtype", default="auto")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--hf-token")
    parser.add_argument("--output-root", default="artifacts/activations")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--sync-to-hf", action="store_true")
    parser.add_argument("--dry-run-sync", action="store_true")
    parser.add_argument("--config", default="configs/experiment.yaml")
    args = parser.parse_args()

    experiment = load_yaml(args.config)
    act_config = experiment.get("activation_extraction", {})
    batch_size = args.batch_size or int(act_config.get("batch_size", 4))
    flush_every_batches = args.flush_every_batches or int(act_config.get("flush_every_batches", 8))
    rows = load_neutral_rows(args.input, limit=args.limit)
    if not rows:
        raise ValueError("no neutral rows found")

    run = RunContext.create(
        task=f"extract_neutral_activations_{args.neutral_bank}",
        model_id=args.model_id,
        run_id=args.run_id,
        config_path=args.config,
        metadata={
            "neutral_bank": args.neutral_bank,
            "input": args.input,
            "model_name": args.model_name,
            "adapter_path": args.adapter_path,
            "backend": args.backend,
            "batch_size": batch_size,
            "flush_every_batches": flush_every_batches,
        },
        resume=args.resume or bool(args.run_id),
    )
    try:
        if args.backend == "dry_run_metadata":
            out = write_neutral_dry_run_metadata(
                run=run,
                rows=rows,
                model_id=args.model_id,
                model_name=args.model_name,
                neutral_bank=args.neutral_bank,
            )
        else:
            out = extract_neutral_mean_activations(
                run=run,
                rows=rows,
                model_id=args.model_id,
                model_name=args.model_name,
                neutral_bank=args.neutral_bank,
                batch_size=batch_size,
                config_path=args.config,
                adapter_path=args.adapter_path,
                hf_token=args.hf_token,
                torch_dtype=args.torch_dtype,
                device_map=args.device_map,
                flush_every_batches=flush_every_batches,
                output_root=args.output_root,
            )
        run.mark_completed(f"wrote {out}")
        if args.sync_to_hf or args.dry_run_sync:
            upload = upload_artifact_to_hf(run.run_dir, dry_run=args.dry_run_sync)
            run.update_progress(uploaded=[upload["remote_path"]])
            print(f"sync: {upload}")
        print(out)
    except BaseException as exc:
        run.mark_failed(exc)
        if args.sync_to_hf or args.dry_run_sync:
            upload = upload_artifact_to_hf(run.run_dir, dry_run=args.dry_run_sync)
            run.update_progress(uploaded=[upload["remote_path"]])
            print(f"sync_after_failure: {upload}")
        raise


if __name__ == "__main__":
    main()

