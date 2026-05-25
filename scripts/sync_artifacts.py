#!/usr/bin/env python3
"""Upload local artifacts to the configured Hugging Face dataset repo."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from em_latent_factors.artifacts import upload_artifact_to_hf


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    parser.add_argument("--repo-id")
    parser.add_argument("--repo-type", default="dataset")
    parser.add_argument("--remote-path")
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    result = upload_artifact_to_hf(
        local_path=args.path,
        repo_id=args.repo_id,
        repo_type=args.repo_type,
        remote_path=args.remote_path,
        config_path=args.config,
        dry_run=args.dry_run,
    )
    for key, value in result.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()

