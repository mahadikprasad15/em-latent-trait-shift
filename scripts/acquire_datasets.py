#!/usr/bin/env python3
"""Acquire raw external datasets for the pilot."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from em_latent_factors.datasets.acquire import acquire_dataset
from em_latent_factors.datasets.registry import load_dataset_config


FT_DATASETS = [
    "ft_health_bad_advice",
    "ft_finance_bad_advice",
    "ft_insecure_code",
]
EVAL_DATASETS = [
    "eval_core_misalignment",
    "eval_extended_misalignment_by_category",
    "eval_hallucination_tool_deception",
    "eval_strongreject_unsafe_compliance",
    "eval_health_bad_advice",
    "eval_finance_risky_advice",
    "eval_code_insecurity",
    "eval_xstest_safe_overrefusal",
    "eval_sycophancy",
]
NEUTRAL_DATASETS = [
    "neutral_mtbench",
    "neutral_general_alpaca",
    "neutral_benign_advice",
    "neutral_benign_code",
    "neutral_safety_education",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/datasets.yaml")
    parser.add_argument("--dataset", action="append", default=[])
    parser.add_argument("--all-ft", action="store_true")
    parser.add_argument("--all-evals", action="store_true")
    parser.add_argument("--all-neutral", action="store_true")
    parser.add_argument("--hf-token")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    load_dataset_config(args.config)
    dataset_ids = list(args.dataset)
    if args.all_ft:
        dataset_ids.extend(FT_DATASETS)
    if args.all_evals:
        dataset_ids.extend(EVAL_DATASETS)
    if args.all_neutral:
        dataset_ids.extend(NEUTRAL_DATASETS)
    if not dataset_ids:
        parser.error("pass --dataset, --all-ft, --all-evals, or --all-neutral")

    for dataset_id in dict.fromkeys(dataset_ids):
        try:
            paths = acquire_dataset(dataset_id, hf_token=args.hf_token, force=args.force)
            print(f"{dataset_id}: acquired {', '.join(str(p) for p in paths)}")
        except Exception as exc:
            print(f"{dataset_id}: FAILED: {exc}", file=sys.stderr)
            raise


if __name__ == "__main__":
    main()
