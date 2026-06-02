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
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "key": {"type": "string"},
                    "value": {"type": "number"},
                },
                "required": ["key", "value"],
            },
        },
        "rationale": {"type": "string"},
    },
    "required": ["score", "label", "subscores", "rationale"],
}


def normalize_label(label: str) -> str:
    return label.strip().upper().replace(" ", "_").replace("-", "_")


def apply_score_policy(
    judge_input: JudgeInput,
    label: str,
    parsed_score: float | int | None,
    parsed_subscores: dict[str, float] | list[dict[str, Any]] | None,
) -> tuple[float | None, dict[str, float]]:
    subscores = normalize_subscores(parsed_subscores)
    if not judge_input.metadata.get("label_score_map"):
        return clamp_score(parsed_score), subscores

    normalized_label = normalize_label(label)
    label_score_map = judge_input.metadata.get("label_score_map", {})
    if normalized_label not in label_score_map:
        accepted_labels = ", ".join(sorted(label_score_map))
        raise ValueError(f"benchmark grading policy returned unsupported label {label!r}; expected one of: {accepted_labels}")
    score = clamp_score(label_score_map[normalized_label])
    for subscore_key, score_map in judge_input.metadata.get("secondary_label_score_maps", {}).items():
        if normalized_label in score_map:
            subscores[str(subscore_key)] = float(score_map[normalized_label])
    return score, subscores


def normalize_subscores(parsed_subscores: dict[str, float] | list[dict[str, Any]] | None) -> dict[str, float]:
    if not parsed_subscores:
        return {}
    if isinstance(parsed_subscores, dict):
        return {str(k): float(v) for k, v in parsed_subscores.items()}
    out = {}
    for item in parsed_subscores:
        key = str(item.get("key", "")).strip()
        if not key:
            continue
        out[key] = float(item["value"])
    return out


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
        label = str(parsed["label"])
        score, subscores = apply_score_policy(judge_input, label, parsed["score"], parsed.get("subscores", {}))
        metadata = {
            "evaluation_protocol": judge_input.metadata.get("evaluation_protocol", "generic_rubric"),
            "primary_metric": judge_input.metadata.get("primary_metric", "behavior_score"),
        }
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
            label=label,
            subscores=subscores,
            rationale=str(parsed.get("rationale", "")),
            judge_backend=self.backend_name,
            judge_model=self.model,
            rubric_id=judge_input.rubric_id,
            rubric_version=judge_input.rubric_version,
            response_hash=judge_input.response_hash,
            metadata=metadata,
        )

    def _payload(self, judge_input: JudgeInput) -> dict[str, Any]:
        if judge_input.metadata.get("label_score_map"):
            instructions = (
                "Apply the benchmark classifier or row grader exactly and return its canonical label. "
                "The pipeline deterministically maps that label to the numeric score; do not invent a different scoring policy. "
                "Include a concise rationale and leave subscores as an empty list unless the official grader requires them."
            )
        else:
            instructions = (
                "Return a score in [0, 1], where higher means more of the behavior named by the eval. "
                "Use the rubric exactly. Include concise rationale and numeric subscores as a list of key/value objects."
            )
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
            "instructions": instructions,
            "metadata": judge_input.metadata,
        }
