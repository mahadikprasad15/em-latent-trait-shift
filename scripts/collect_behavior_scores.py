#!/usr/bin/env python3
"""Collect behavior eval aggregate CSVs into canonical results/behavior_scores.csv."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from em_latent_factors.artifacts import RunContext, upload_artifact_to_hf
from em_latent_factors.behavior_scores import collect_behavior_scores


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", default="artifacts/runs")
    parser.add_argument("--output", default="results/behavior_scores.csv")
    parser.add_argument("--base-model-id", default="base")
    parser.add_argument("--category", default="all")
    parser.add_argument("--include-categories", action="store_true")
    parser.add_argument("--allow-missing-base", action="store_true")
    parser.add_argument("--include-run-id", action="append", default=[])
    parser.add_argument("--run-id")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--sync-to-hf", action="store_true")
    parser.add_argument("--dry-run-sync", action="store_true")
    args = parser.parse_args()

    run = RunContext.create(
        task="collect_behavior_scores",
        run_id=args.run_id,
        metadata={
            "runs_root": args.runs_root,
            "output": args.output,
            "base_model_id": args.base_model_id,
            "category": args.category,
            "include_categories": args.include_categories,
            "allow_missing_base": args.allow_missing_base,
            "include_run_ids": args.include_run_id,
        },
        resume=args.resume or bool(args.run_id),
    )
    try:
        result = collect_behavior_scores(
            runs_root=args.runs_root,
            output_path=args.output,
            base_model_id=args.base_model_id,
            category=args.category,
            include_categories=args.include_categories,
            allow_missing_base=args.allow_missing_base,
            include_run_ids=args.include_run_id or None,
        )
        run.update_progress(counters={"behavior_score_rows": result["rows"], "models": result["models"], "evals": len(result["evals"])}, completed_units=[args.output])
        run.mark_completed("behavior score collection complete")
        uploads = []
        if args.sync_to_hf or args.dry_run_sync:
            for path in (run.run_dir, args.output):
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
