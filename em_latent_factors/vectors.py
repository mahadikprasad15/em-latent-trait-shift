"""Trait-vector config loading and rollout artifact helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from em_latent_factors.config import load_yaml
from em_latent_factors.io import ensure_parent, read_jsonl, write_jsonl


VECTOR_CONFIGS = {
    "v_toxic_reckless_persona": "configs/vectors/toxic_reckless_persona.yaml",
    "v_deception_concealment": "configs/vectors/deception_concealment.yaml",
    "v_hallucination": "configs/vectors/hallucination.yaml",
    "v_sycophancy": "configs/vectors/sycophancy.yaml",
    "v_refusal_gate": "configs/vectors/refusal_gate.yaml",
    "v_harmful_advice_continuation": "configs/vectors/harmful_advice_continuation.yaml",
    "v_insecure_code_continuation": "configs/vectors/insecure_code_continuation.yaml",
}


@dataclass(frozen=True)
class TraitConfig:
    trait_id: str
    config_path: str
    description: str
    orientation: str
    positive_instructions: list[str]
    negative_instructions: list[str]
    extraction_questions: list[str]
    validation_questions: list[str]
    judge_rubric: str
    question_set_ref: str | None = None


def load_trait_config(trait_id: str) -> TraitConfig:
    try:
        config_path = VECTOR_CONFIGS[trait_id]
    except KeyError as exc:
        raise KeyError(f"unknown trait_id: {trait_id}") from exc
    raw = load_yaml(config_path)
    question_set_ref = raw.get("question_set_ref")
    if question_set_ref:
        shared = load_yaml(question_set_ref)
        extraction_questions = list(shared["extraction_questions"])
        validation_questions = list(shared["validation_questions"])
    else:
        extraction_questions = list(raw["extraction_questions"])
        validation_questions = list(raw["validation_questions"])
    return TraitConfig(
        trait_id=raw["trait_id"],
        config_path=config_path,
        description=raw["description"],
        orientation=raw["orientation"],
        positive_instructions=list(raw["positive_instructions"]),
        negative_instructions=list(raw["negative_instructions"]),
        extraction_questions=extraction_questions,
        validation_questions=validation_questions,
        judge_rubric=raw["judge_rubric"],
        question_set_ref=question_set_ref,
    )


def all_trait_ids() -> list[str]:
    return list(VECTOR_CONFIGS)


def build_rollout_prompt_rows(
    trait: TraitConfig,
    rollouts_per_pair: int = 1,
    include_validation: bool = False,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    question_kind_to_questions = [("extraction", trait.extraction_questions)]
    if include_validation:
        question_kind_to_questions.append(("validation", trait.validation_questions))
    for pole, instructions in (("positive", trait.positive_instructions), ("negative", trait.negative_instructions)):
        for instruction_id, instruction in enumerate(instructions):
            for question_kind, questions in question_kind_to_questions:
                for question_id, question in enumerate(questions):
                    for rollout_id in range(rollouts_per_pair):
                        prompt_id = f"{trait.trait_id}:{pole}:i{instruction_id}:q{question_id}:r{rollout_id}:{question_kind}"
                        rows.append(
                            {
                                "prompt_id": prompt_id,
                                "trait_id": trait.trait_id,
                                "pole": pole,
                                "instruction_id": instruction_id,
                                "question_id": question_id,
                                "question_kind": question_kind,
                                "rollout_id": rollout_id,
                                "instruction": instruction,
                                "question": question,
                                "messages": [
                                    {"role": "system", "content": instruction},
                                    {"role": "user", "content": question},
                                ],
                                "prompt": question,
                                "metadata": {
                                    "trait_config_path": trait.config_path,
                                    "question_set_ref": trait.question_set_ref,
                                    "orientation": trait.orientation,
                                    "description": trait.description,
                                },
                            }
                        )
    return rows


def generation_to_rollout(generation: dict) -> dict:
    input_row = generation.get("metadata", {}).get("input_row", {})
    return {
        "trait_id": input_row.get("trait_id"),
        "pole": input_row.get("pole"),
        "instruction_id": input_row.get("instruction_id"),
        "question_id": input_row.get("question_id"),
        "question_kind": input_row.get("question_kind"),
        "rollout_id": input_row.get("rollout_id"),
        "prompt_id": generation.get("prompt_id"),
        "messages": generation.get("messages"),
        "instruction": input_row.get("instruction"),
        "question": input_row.get("question"),
        "response": generation.get("response"),
        "model_id": generation.get("model_id"),
        "model_name": generation.get("model_name"),
        "generation_config": generation.get("generation_config"),
        "dry_run": generation.get("dry_run"),
        "metadata": {
            "generation_metadata": generation.get("metadata", {}),
            "trait_metadata": input_row.get("metadata", {}),
        },
    }


def write_rollout_prompt_file(path: str | Path, rows: Iterable[dict]) -> int:
    return write_jsonl(path, rows)


def write_canonical_rollout_splits(
    generations_path: str | Path,
    output_root: str | Path = "data/vector_rollouts",
    trait_ids: set[str] | None = None,
) -> dict[str, int]:
    grouped: dict[tuple[str, str], list[dict]] = {}
    for generation in read_jsonl(generations_path):
        rollout = generation_to_rollout(generation)
        trait_id = rollout.get("trait_id")
        pole = rollout.get("pole")
        if not trait_id or not pole:
            continue
        if trait_ids and trait_id not in trait_ids:
            continue
        grouped.setdefault((str(trait_id), str(pole)), []).append(rollout)
    counts: dict[str, int] = {}
    for (trait_id, pole), rows in grouped.items():
        path = ensure_parent(Path(output_root) / trait_id / f"{pole}.jsonl")
        counts[str(path)] = write_jsonl(path, rows)
    return counts

