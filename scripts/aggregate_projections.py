#!/usr/bin/env python3
"""Aggregate raw projection rows into regression-ready z features."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from em_latent_factors.artifacts import RunContext, upload_artifact_to_hf
from em_latent_factors.projection_aggregation import aggregate_projection_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="results/projections.csv")
    parser.add_argument("--output", default="results/projections_aggregated.csv")
    parser.add_argument("--standardized-output", default="results/projections_standardized_long.csv")
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--layer", type=int, action="append", default=[], help="Layer to include; repeat for multiple. Defaults to all rows in input.")
    parser.add_argument(
        "--layer-aggregate",
        choices=["middle", "mean_standardized", "per_layer"],
        default="middle",
        help="How to reduce layer-wise projections for the aggregated output. Default is the lower middle available layer.",
    )
    parser.add_argument("--include-domain-balanced", action="store_true", help="Also create exploratory equal-bank neutral_domain_balanced rows.")
    parser.add_argument("--fail-on-missing-cells", action="store_true", help="Fail if any model/bank/layer/trait cell is missing.")
    parser.add_argument("--run-id")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--sync-to-hf", action="store_true")
    parser.add_argument("--dry-run-sync", action="store_true")
    args = parser.parse_args()

    run = RunContext.create(
        task="aggregate_projections",
        run_id=args.run_id,
        metadata={
            "input": args.input,
            "output": args.output,
            "standardized_output": args.standardized_output,
            "config": args.config,
            "layers": args.layer,
            "layer_aggregate": args.layer_aggregate,
            "include_domain_balanced": args.include_domain_balanced,
            "fail_on_missing_cells": args.fail_on_missing_cells,
        },
        resume=args.resume or bool(args.run_id),
    )
    try:
        result = aggregate_projection_file(
            input_path=args.input,
            output_path=args.output,
            standardized_output_path=args.standardized_output,
            config_path=args.config,
            layers=args.layer or None,
            layer_aggregate=args.layer_aggregate,
            include_domain_balanced=args.include_domain_balanced,
            fail_on_missing_cells=args.fail_on_missing_cells,
        )
        run.update_progress(
            counters={
                "raw_rows": result["raw_rows"],
                "standardized_rows": result["standardized_rows"],
                "aggregated_rows": result["aggregated_rows"],
                "models": result["models"],
            },
            completed_units=[str(args.output), str(args.standardized_output)],
        )
        run.mark_completed("projection aggregation complete")
        uploads = []
        if args.sync_to_hf or args.dry_run_sync:
            for path in (run.run_dir, args.output, args.standardized_output):
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
