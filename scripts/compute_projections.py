#!/usr/bin/env python3
"""Compute raw per-layer projections of activation shifts onto trait vectors."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from em_latent_factors.artifacts import RunContext, upload_artifact_to_hf
from em_latent_factors.projections import compute_raw_projections
from em_latent_factors.vectors import all_trait_ids


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shifts", required=True)
    parser.add_argument("--vector-model-id", required=True)
    parser.add_argument("--trait-id", action="append", default=[])
    parser.add_argument("--all-traits", action="store_true")
    parser.add_argument("--vectors-root", default="artifacts/vectors")
    parser.add_argument("--output-root", default="results")
    parser.add_argument("--run-id")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--sync-to-hf", action="store_true")
    parser.add_argument("--dry-run-sync", action="store_true")
    args = parser.parse_args()

    trait_ids = list(args.trait_id)
    if args.all_traits:
        trait_ids.extend(all_trait_ids())
    trait_ids = list(dict.fromkeys(trait_ids))

    run = RunContext.create(
        task="compute_projections",
        run_id=args.run_id,
        metadata={
            "shifts": args.shifts,
            "vector_model_id": args.vector_model_id,
            "trait_ids": trait_ids,
            "all_traits": args.all_traits,
            "vectors_root": args.vectors_root,
            "output_root": args.output_root,
        },
        resume=args.resume or bool(args.run_id),
    )
    try:
        result = compute_raw_projections(
            shifts_path=args.shifts,
            vector_model_id=args.vector_model_id,
            run=run,
            trait_ids=trait_ids,
            all_traits=args.all_traits,
            vectors_root=args.vectors_root,
            output_root=args.output_root,
        )
        run.mark_completed("projection computation complete")
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

