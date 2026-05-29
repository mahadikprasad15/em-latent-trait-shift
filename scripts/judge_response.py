#!/usr/bin/env python3
"""Score one prompt/response pair with a configured judge backend."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from em_latent_factors.judges.base import JudgeInput
from em_latent_factors.judges.factory import build_judge
from em_latent_factors.judges.rubrics import get_rubric


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-id", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--response", required=True)
    parser.add_argument("--prompt-id", default="manual_prompt")
    parser.add_argument("--model-id", default="manual_model")
    parser.add_argument("--backend", choices=["stub", "openai", "strongreject"], default="stub")
    parser.add_argument("--judge-model")
    parser.add_argument("--stub-score", type=float)
    args = parser.parse_args()

    rubric = get_rubric(args.eval_id)
    judge_input = JudgeInput(
        eval_id=args.eval_id,
        prompt_id=args.prompt_id,
        model_id=args.model_id,
        prompt=args.prompt,
        response=args.response,
        rubric_id=rubric.rubric_id,
        rubric_version=rubric.version,
        rubric=rubric.text,
        metadata={
            "behavior_name": rubric.behavior_name,
            "score_meaning": rubric.score_meaning,
            "labels": rubric.labels,
            "subscore_keys": rubric.subscore_keys,
        },
    )
    if args.backend == "stub":
        judge = build_judge("stub", score=args.stub_score)
    elif args.backend == "openai":
        judge = build_judge("openai", model=args.judge_model)
    else:
        judge = build_judge("strongreject", model=args.judge_model)
    result = judge.score(judge_input)
    print(json.dumps(result.to_json(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
