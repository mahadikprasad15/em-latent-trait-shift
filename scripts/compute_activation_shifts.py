#!/usr/bin/env python3
"""Compute fine-tuned minus base activation shifts."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from em_latent_factors.artifacts import RunContext, upload_artifact_to_hf
from em_latent_factors.shifts import compute_activation_shifts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-activations", required=True)
    parser.add_argument("--finetuned-activations", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--base-model-id", required=True)
    parser.add_argument("--neutral-bank")
    parser.add_argument("--output-root", default="artifacts/shifts")
    parser.add_argument("--run-id")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--sync-to-hf", action="store_true")
    parser.add_argument("--dry-run-sync", action="store_true")
    args = parser.parse_args()

    run = RunContext.create(
        task="compute_activation_shifts",
        model_id=args.model_id,
        run_id=args.run_id,
        metadata={
            "base_activations": args.base_activations,
            "finetuned_activations": args.finetuned_activations,
            "base_model_id": args.base_model_id,
            "neutral_bank": args.neutral_bank,
            "output_root": args.output_root,
        },
        resume=args.resume or bool(args.run_id),
    )
    try:
        result = compute_activation_shifts(
            base_activations_path=args.base_activations,
            finetuned_activations_path=args.finetuned_activations,
            run=run,
            model_id=args.model_id,
            base_model_id=args.base_model_id,
            neutral_bank=args.neutral_bank,
            output_root=args.output_root,
        )
        run.mark_completed("activation shift computation complete")
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

