#!/usr/bin/env python3
"""Inspect activation layer resolution for a model or explicit layer count."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from em_latent_factors.activations import infer_num_hidden_layers_from_model_config, resolve_layers_from_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name")
    parser.add_argument("--num-hidden-layers", type=int)
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--hf-token")
    args = parser.parse_args()

    if args.num_hidden_layers is None and not args.model_name:
        parser.error("pass --num-hidden-layers or --model-name")
    if args.num_hidden_layers is not None:
        num_hidden_layers = args.num_hidden_layers
    else:
        try:
            from transformers import AutoConfig
        except ModuleNotFoundError as exc:
            raise RuntimeError("transformers is required for --model-name") from exc
        token = args.hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        model_config = AutoConfig.from_pretrained(args.model_name, token=token)
        num_hidden_layers = infer_num_hidden_layers_from_model_config(model_config)
    selection = resolve_layers_from_config(num_hidden_layers=num_hidden_layers, config_path=args.config)
    print(json.dumps(selection.to_json(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

