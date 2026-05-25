"""Stub judge backend for plumbing and resume tests."""

from __future__ import annotations

from em_latent_factors.judges.base import JudgeInput, JudgeResult


class StubJudge:
    backend_name = "stub"

    def __init__(self, score: float | None = None, label: str = "stub_unscored") -> None:
        self.score_value = score
        self.label = label

    def score(self, judge_input: JudgeInput) -> JudgeResult:
        score_100 = None if self.score_value is None else self.score_value * 100.0
        return JudgeResult(
            judge_key=judge_input.judge_key,
            eval_id=judge_input.eval_id,
            prompt_id=judge_input.prompt_id,
            model_id=judge_input.model_id,
            score=self.score_value,
            score_100=score_100,
            label=self.label,
            subscores={},
            rationale="StubJudge does not score behavior.",
            judge_backend=self.backend_name,
            judge_model="stub",
            rubric_id=judge_input.rubric_id,
            rubric_version=judge_input.rubric_version,
            response_hash=judge_input.response_hash,
            metadata={"stub": True},
        )

