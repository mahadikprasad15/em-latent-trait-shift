"""Native StrongREJECT rubric evaluator backend."""

from __future__ import annotations

import math

from em_latent_factors.judges.base import JudgeInput, JudgeResult, clamp_score, score_100


STRONGREJECT_EVAL_ID = "eval_strongreject_unsafe_compliance"


class StrongRejectJudge:
    """Wrap the official StrongREJECT rubric evaluator in the common judge schema."""

    backend_name = "strongreject"
    evaluator_name = "strongreject_rubric"

    def __init__(self, model: str | None = None) -> None:
        self.model = model

    def score(self, judge_input: JudgeInput) -> JudgeResult:
        if judge_input.eval_id != STRONGREJECT_EVAL_ID:
            raise ValueError(
                f"StrongRejectJudge only supports {STRONGREJECT_EVAL_ID}, got {judge_input.eval_id}"
            )
        try:
            from strong_reject.evaluate import strongreject_rubric  # type: ignore
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "StrongREJECT scoring requires the official strong_reject package; "
                "install project requirements before running this backend."
            ) from exc

        kwargs = {"models": [self.model]} if self.model else {}
        native_result = strongreject_rubric(judge_input.prompt, judge_input.response, **kwargs)
        raw_score = float(native_result["score"])
        if math.isnan(raw_score):
            raise RuntimeError("StrongREJECT rubric evaluator returned NaN score")
        score = clamp_score(raw_score)
        subscores = {
            str(key): float(value)
            for key, value in native_result.items()
            if key not in {"score", "judge_model"} and isinstance(value, (int, float))
        }
        native_judge_model = native_result.get("judge_model") or self.model or "official_default"
        return JudgeResult(
            judge_key=judge_input.judge_key,
            eval_id=judge_input.eval_id,
            prompt_id=judge_input.prompt_id,
            model_id=judge_input.model_id,
            score=score,
            score_100=score_100(score),
            label="strongreject_harmfulness",
            subscores=subscores,
            rationale="Scored by the official StrongREJECT rubric evaluator.",
            judge_backend=self.backend_name,
            judge_model=str(native_judge_model),
            rubric_id="strongreject_rubric_native",
            rubric_version="souly_2024_official",
            response_hash=judge_input.response_hash,
            metadata={
                "evaluation_protocol": "official_strongreject_rubric",
                "primary_metric": "harmfulness_score",
                "native_evaluator": self.evaluator_name,
                "native_result": native_result,
            },
        )
