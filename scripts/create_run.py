#!/usr/bin/env python3
"""Create or resume an artifact run directory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from em_latent_factors.artifacts import RunContext


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--model-id")
    parser.add_argument("--run-id")
    parser.add_argument("--output-root", default="artifacts/runs")
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--metadata-json", default="{}")
    args = parser.parse_args()

    metadata = json.loads(args.metadata_json)
    run = RunContext.create(
        task=args.task,
        model_id=args.model_id,
        run_id=args.run_id,
        output_root=args.output_root,
        config_path=args.config,
        metadata=metadata,
        resume=args.resume,
    )
    print(run.run_dir)
    print(run.manifest_path)
    print(run.status_path)
    print(run.progress_path)


if __name__ == "__main__":
    main()

