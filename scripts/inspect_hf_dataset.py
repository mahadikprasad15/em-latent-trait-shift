#!/usr/bin/env python3
"""List Hugging Face dataset repo files visible to the current token."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("repo_id")
    parser.add_argument("--hf-token", default=None)
    args = parser.parse_args()

    try:
        from huggingface_hub import HfApi  # type: ignore
    except ModuleNotFoundError as exc:
        raise RuntimeError("huggingface_hub is required; install requirements.txt") from exc

    token = args.hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    api = HfApi(token=token)
    files = api.list_repo_files(args.repo_id, repo_type="dataset")
    for file in files:
        print(file)


if __name__ == "__main__":
    main()
