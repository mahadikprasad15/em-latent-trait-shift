#!/usr/bin/env python3
"""Aggregate behavior judge scores for one behavior eval run."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from em_latent_factors.artifacts import RunContext, upload_artifact_to_hf
from em_latent_factors.evaluation import aggregate_judge_scores


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-id", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--generations")
    parser.add_argument("--judge-scores")
    parser.add_argument("--run-id")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--sync-to-hf", action="store_true")
    parser.add_argument("--dry-run-sync", action="store_true")
    args = parser.parse_args()

    run = RunContext.create(
        task=f"behavior_aggregation_{args.eval_id}",
        model_id=args.model_id,
        run_id=args.run_id,
        metadata={
            "eval_id": args.eval_id,
            "generations": args.generations,
            "judge_scores": args.judge_scores,
        },
        resume=args.resume or bool(args.run_id),
    )
    try:
        generations_path = Path(args.generations) if args.generations else run.run_dir / "results" / "generations.jsonl"
        judge_scores_path = Path(args.judge_scores) if args.judge_scores else run.run_dir / "results" / "judge_scores.jsonl"
        aggregate = aggregate_judge_scores(
            judge_scores_path=judge_scores_path,
            generations_path=generations_path,
            output_json_path=run.run_dir / "results" / "aggregate_scores.json",
            output_csv_path=run.run_dir / "results" / "aggregate_scores.csv",
        )
        run.update_progress(counters={"n_scored_rows": aggregate["n_scored_rows"]})
        run.mark_completed("behavior aggregation complete")
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
