#!/usr/bin/env python3
"""Validate all locally available normalized datasets."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from em_latent_factors.datasets.validate import validate_all


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/datasets.yaml")
    parser.add_argument("--all", action="store_true", help="validate registry entries and report missing datasets")
    parser.add_argument("--available", action="store_true", help="only print available datasets")
    args = parser.parse_args()

    include_missing = not args.available
    results = validate_all(args.config, include_missing=include_missing)
    for result in results:
        if result["status"] == "missing":
            print(f"{result['dataset_id']}: missing path={result['path']}")
        else:
            extras = " ".join(f"{k}={v}" for k, v in result.items() if k not in {"dataset_id", "path", "status"})
            print(f"{result['dataset_id']}: ok path={result['path']} {extras}")


if __name__ == "__main__":
    main()

