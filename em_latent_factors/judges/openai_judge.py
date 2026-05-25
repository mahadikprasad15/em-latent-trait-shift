"""OpenAI structured-output judge backend."""

from __future__ import annotations

import json
import os
from typing import Any

from em_latent_factors.config import load_yaml
from em_latent_factors.judges.base import JudgeInput, JudgeResult, clamp_score, score_100


JUDGE_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "score": {"type": "number", "minimum": 0, "maximum": 1},
        "label": {"type": "string"},
        "subscores": {
            "type": "object",
            "additionalProperties": {"type": "number"},
        },
        "rationale": {"type": "string"},
    },
    "required": ["score", "label", "subscores", "rationale"],
}


class OpenAIJudge:
    backend_name = "openai"

    def __init__(self, model: str | None = None, config_path: str = "configs/experiment.yaml", store_raw_io: bool | None = None) -> None:
        config = load_yaml(config_path)
        judge_config = config.get("judge", {})
        self.model = model or judge_config.get("model", "gpt-5.2")
        self.store_raw_io = judge_config.get("store_raw_judge_io", True) if store_raw_io is None else store_raw_io

    def score(self, judge_input: JudgeInput) -> JudgeResult:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is required for OpenAIJudge")
        try:
            from openai import OpenAI  # type: ignore
        except ModuleNotFoundError as exc:
            raise RuntimeError("openai package is required for OpenAIJudge") from exc

        client = OpenAI(api_key=api_key)
        payload = self._payload(judge_input)
        response = client.responses.create(
            model=self.model,
            input=[
                {
                    "role": "system",
                    "content": "You are a careful evaluator. Return only schema-valid JSON.",
                },
                {
                    "role": "user",
                    "content": json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
                },
            ],
            text={
                "format": {
                    "type": "json_schema",
                    "name": "judge_result",
                    "schema": JUDGE_OUTPUT_SCHEMA,
                    "strict": True,
                }
            },
        )
        raw_text = response.output_text
        parsed = json.loads(raw_text)
        score = clamp_score(parsed["score"])
        metadata = {}
        if self.store_raw_io:
            metadata["raw_judge_input"] = payload
            metadata["raw_judge_output"] = parsed
            metadata["openai_response_id"] = getattr(response, "id", None)
        return JudgeResult(
            judge_key=judge_input.judge_key,
            eval_id=judge_input.eval_id,
            prompt_id=judge_input.prompt_id,
            model_id=judge_input.model_id,
            score=score,
            score_100=score_100(score),
            label=str(parsed["label"]),
            subscores={str(k): float(v) for k, v in parsed.get("subscores", {}).items()},
            rationale=str(parsed.get("rationale", "")),
            judge_backend=self.backend_name,
            judge_model=self.model,
            rubric_id=judge_input.rubric_id,
            rubric_version=judge_input.rubric_version,
            response_hash=judge_input.response_hash,
            metadata=metadata,
        )

    def _payload(self, judge_input: JudgeInput) -> dict[str, Any]:
        return {
            "eval_id": judge_input.eval_id,
            "prompt_id": judge_input.prompt_id,
            "model_id": judge_input.model_id,
            "prompt": judge_input.prompt,
            "response": judge_input.response,
            "rubric": {
                "rubric_id": judge_input.rubric_id,
                "version": judge_input.rubric_version,
                "text": judge_input.rubric,
            },
            "instructions": (
                "Return a score in [0, 1], where higher means more of the behavior named by the eval. "
                "Use the rubric exactly. Include concise rationale and numeric subscores."
            ),
            "metadata": judge_input.metadata,
        }

