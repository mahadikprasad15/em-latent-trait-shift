#!/usr/bin/env python3
"""Run regression analyses linking behavior deltas to projection features."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from em_latent_factors.artifacts import RunContext, upload_artifact_to_hf
from em_latent_factors.regressions import run_regression_analysis


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--behavior-scores", default="results/behavior_scores.csv")
    parser.add_argument("--projections", default="results/projections_aggregated.csv")
    parser.add_argument("--output-root", default="results")
    parser.add_argument("--neutral-bank", default="neutral_all")
    parser.add_argument("--layer-aggregate", default="middle_quantile_layer")
    parser.add_argument("--base-model-id", default="base")
    parser.add_argument("--category", default="all")
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--run-id")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--sync-to-hf", action="store_true")
    parser.add_argument("--dry-run-sync", action="store_true")
    args = parser.parse_args()

    run = RunContext.create(
        task="run_regressions",
        run_id=args.run_id,
        metadata={
            "behavior_scores": args.behavior_scores,
            "projections": args.projections,
            "output_root": args.output_root,
            "neutral_bank": args.neutral_bank,
            "layer_aggregate": args.layer_aggregate,
            "base_model_id": args.base_model_id,
            "category": args.category,
            "ridge_alpha": args.ridge_alpha,
        },
        resume=args.resume or bool(args.run_id),
    )
    try:
        result = run_regression_analysis(
            behavior_scores_path=args.behavior_scores,
            projections_path=args.projections,
            output_root=args.output_root,
            neutral_bank=args.neutral_bank,
            layer_aggregate=args.layer_aggregate,
            base_model_id=args.base_model_id,
            category=args.category,
            ridge_alpha=args.ridge_alpha,
        )
        run.update_progress(
            counters={
                "regression_rows": result["rows"],
                "models": result["models"],
                "evals": len(result["evals"]),
            },
            completed_units=list(result["output_paths"].values()),
        )
        run.mark_completed("regression analysis complete")
        uploads = []
        if args.sync_to_hf or args.dry_run_sync:
            for path in [run.run_dir, *result["output_paths"].values()]:
                upload = upload_artifact_to_hf(path, dry_run=args.dry_run_sync)
                uploads.append(upload)
            run.update_progress(uploaded=[upload["remote_path"] for upload in uploads])
            print(f"sync: {uploads}")
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
