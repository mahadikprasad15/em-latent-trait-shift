#!/usr/bin/env python3
"""Create analysis plots from regression result CSVs."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from em_latent_factors.artifacts import RunContext, upload_artifact_to_hf
from em_latent_factors.plots import plot_results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--figures-root", default="figures")
    parser.add_argument("--format", action="append", choices=["pdf", "png"], default=[])
    parser.add_argument("--run-id")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--sync-to-hf", action="store_true")
    parser.add_argument("--dry-run-sync", action="store_true")
    args = parser.parse_args()

    formats = tuple(args.format or ["pdf", "png"])
    run = RunContext.create(
        task="plot_results",
        run_id=args.run_id,
        metadata={
            "results_root": args.results_root,
            "figures_root": args.figures_root,
            "formats": list(formats),
        },
        resume=args.resume or bool(args.run_id),
    )
    try:
        result = plot_results(results_root=args.results_root, figures_root=args.figures_root, formats=formats)
        run.update_progress(counters={"plot_outputs": result["n_outputs"]}, completed_units=result["outputs"])
        run.mark_completed("plotting complete")
        uploads = []
        if args.sync_to_hf or args.dry_run_sync:
            for path in (run.run_dir, args.figures_root):
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
