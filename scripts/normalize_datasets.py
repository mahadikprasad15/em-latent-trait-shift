#!/usr/bin/env python3
"""Normalize acquired raw datasets into canonical JSONL files."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from em_latent_factors.datasets.normalize import normalize_dataset


EVAL_DATASETS = [
    "eval_core_misalignment",
    "eval_extended_misalignment_by_category",
    "eval_hallucination_tool_deception",
    "eval_strongreject_unsafe_compliance",
    "eval_health_bad_advice",
    "eval_finance_risky_advice",
    "eval_code_insecurity",
    "eval_xstest_safe_overrefusal",
    "eval_sycophancy_answer",
]
NEUTRAL_DATASETS = [
    "neutral_mtbench",
    "neutral_general_alpaca",
    "neutral_benign_advice",
    "neutral_benign_code",
    "neutral_safety_education",
    "neutral_all",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", action="append", default=[])
    parser.add_argument("--all-evals", action="store_true")
    parser.add_argument("--all-neutral", action="store_true")
    args = parser.parse_args()

    dataset_ids = list(args.dataset)
    if args.all_evals:
        dataset_ids.extend(EVAL_DATASETS)
    if args.all_neutral:
        dataset_ids.extend(NEUTRAL_DATASETS)
    if not dataset_ids:
        parser.error("pass --dataset, --all-evals, or --all-neutral")

    for dataset_id in dict.fromkeys(dataset_ids):
        result = normalize_dataset(dataset_id)
        print(f"{dataset_id}: normalized {result}")


if __name__ == "__main__":
    main()
