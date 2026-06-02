#!/usr/bin/env python3
"""Judge existing behavior generations without model loading."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from em_latent_factors.artifacts import RunContext, upload_artifact_to_hf
from em_latent_factors.evaluation import judge_generations


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-id", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--generations")
    parser.add_argument("--judge-backend", choices=["stub", "openai", "strongreject"], default="stub")
    parser.add_argument("--judge-model")
    parser.add_argument("--stub-score", type=float)
    parser.add_argument("--run-id")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--sync-to-hf", action="store_true")
    parser.add_argument("--dry-run-sync", action="store_true")
    args = parser.parse_args()

    run = RunContext.create(
        task=f"behavior_judging_{args.eval_id}",
        model_id=args.model_id,
        run_id=args.run_id,
        metadata={
            "eval_id": args.eval_id,
            "generations": args.generations,
            "judge_backend": args.judge_backend,
            "judge_model": args.judge_model,
        },
        resume=args.resume or bool(args.run_id),
    )
    try:
        generations_path = Path(args.generations) if args.generations else run.run_dir / "results" / "generations.jsonl"
        judge_scores_path = run.run_dir / "results" / "judge_scores.jsonl"
        n_written = judge_generations(
            generations_path=generations_path,
            output_path=judge_scores_path,
            eval_id=args.eval_id,
            judge_backend=args.judge_backend,
            judge_model=args.judge_model,
            stub_score=args.stub_score,
            limit=args.limit,
        )
        run.update_progress(counters={"new_judge_scores": n_written})
        run.mark_completed("behavior judging complete")
        if args.sync_to_hf or args.dry_run_sync:
            upload = upload_artifact_to_hf(run.run_dir, dry_run=args.dry_run_sync)
            run.update_progress(uploaded=[upload["remote_path"]])
            print(f"sync: {upload}")
        print(judge_scores_path)
    except BaseException as exc:
        run.mark_failed(exc)
        if args.sync_to_hf or args.dry_run_sync:
            upload = upload_artifact_to_hf(run.run_dir, dry_run=args.dry_run_sync)
            run.update_progress(uploaded=[upload["remote_path"]])
            print(f"sync_after_failure: {upload}")
        raise


if __name__ == "__main__":
    main()
