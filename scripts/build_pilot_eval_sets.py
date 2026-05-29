#!/usr/bin/env python3
"""Materialize deterministic capped behavior-evaluation views for the pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from em_latent_factors.datasets.pilot import build_pilot_eval_sets
from em_latent_factors.config import load_yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets-config", default="configs/datasets.yaml")
    parser.add_argument("--experiment-config", default="configs/experiment.yaml")
    parser.add_argument("--max-prompts", type=int)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()

    experiment = load_yaml(args.experiment_config)
    evaluation = experiment.get("behavior_evaluation", {})
    max_prompts = args.max_prompts if args.max_prompts is not None else int(evaluation.get("pilot_max_prompts_per_eval", 300))
    seed = args.seed if args.seed is not None else int(evaluation.get("pilot_sampling_seed", 0))
    manifest = build_pilot_eval_sets(args.datasets_config, max_prompts=max_prompts, seed=seed)
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
