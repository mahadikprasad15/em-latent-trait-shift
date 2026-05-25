#!/usr/bin/env python3
"""Construct trait vectors from pooled rollout activation tensors."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from em_latent_factors.artifacts import RunContext, upload_artifact_to_hf
from em_latent_factors.trait_vectors import construct_trait_vectors, inspect_pooled_activation_metadata
from em_latent_factors.vectors import all_trait_ids


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pooled-activations", required=True)
    parser.add_argument("--trait-id", action="append", default=[])
    parser.add_argument("--all-traits", action="store_true")
    parser.add_argument("--model-id")
    parser.add_argument("--run-id")
    parser.add_argument("--question-kind", default="extraction")
    parser.add_argument("--min-count-per-pole", type=int, default=20)
    parser.add_argument("--output-root", default="artifacts/vectors")
    parser.add_argument("--inspect-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--sync-to-hf", action="store_true")
    parser.add_argument("--dry-run-sync", action="store_true")
    args = parser.parse_args()

    trait_ids = list(args.trait_id)
    if args.all_traits:
        trait_ids.extend(all_trait_ids())
    trait_ids = list(dict.fromkeys(trait_ids))

    if args.inspect_only:
        print(inspect_pooled_activation_metadata(args.pooled_activations))
        return

    if not trait_ids and not args.all_traits:
        parser.error("pass --trait-id or --all-traits")

    run = RunContext.create(
        task="construct_trait_vectors",
        model_id=args.model_id,
        run_id=args.run_id,
        metadata={
            "pooled_activations": args.pooled_activations,
            "trait_ids": trait_ids,
            "all_traits": args.all_traits,
            "question_kind": args.question_kind,
            "min_count_per_pole": args.min_count_per_pole,
            "output_root": args.output_root,
        },
        resume=args.resume or bool(args.run_id),
    )
    try:
        result = construct_trait_vectors(
            pooled_activations_path=args.pooled_activations,
            run=run,
            trait_ids=trait_ids,
            all_traits=args.all_traits,
            model_id=args.model_id,
            output_root=args.output_root,
            question_kind=args.question_kind,
            min_count_per_pole=args.min_count_per_pole,
        )
        run.mark_completed("trait vector construction complete")
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

