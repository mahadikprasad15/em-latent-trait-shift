#!/usr/bin/env python3
"""Lightweight validation for v1 trait-vector YAML configs.

This intentionally uses a tiny parser for the simple YAML subset in these files
so the check works before project dependencies are installed.
"""

from __future__ import annotations

from pathlib import Path


VECTOR_DIR = Path("configs/vectors")
BROAD_TRAITS = {
    "v_toxic_reckless_persona",
    "v_deception_concealment",
    "v_hallucination",
    "v_sycophancy",
}
FAMILY_TRAITS = {
    "v_refusal_gate",
    "v_harmful_advice_continuation",
    "v_insecure_code_continuation",
}


def parse_simple_yaml(path: Path) -> dict:
    data: dict[str, object] = {}
    current_key: str | None = None
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip("\n")
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if line.startswith("  - "):
                if current_key is None:
                    raise ValueError(f"{path}: list item without key: {line}")
                data.setdefault(current_key, [])
                if not isinstance(data[current_key], list):
                    raise ValueError(f"{path}: mixed scalar/list key {current_key}")
                data[current_key].append(stripped[2:].strip())
                continue
            if ":" not in line:
                raise ValueError(f"{path}: unsupported line: {line}")
            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip()
            current_key = key
            data[key] = value if value else []
    return data


def require_list_count(path: Path, data: dict, key: str, expected: int) -> None:
    value = data.get(key)
    if not isinstance(value, list):
        raise ValueError(f"{path}: expected list key {key}")
    if len(value) != expected:
        raise ValueError(f"{path}: expected {expected} items for {key}, found {len(value)}")
    if len(set(value)) != len(value):
        raise ValueError(f"{path}: duplicate entries in {key}")
    if any(not item for item in value):
        raise ValueError(f"{path}: empty entry in {key}")


def main() -> None:
    shared_path = VECTOR_DIR / "shared_broad_persona_questions.yaml"
    shared = parse_simple_yaml(shared_path)
    require_list_count(shared_path, shared, "applies_to", 4)
    require_list_count(shared_path, shared, "extraction_questions", 20)
    require_list_count(shared_path, shared, "validation_questions", 20)

    seen_traits: set[str] = set()
    for path in sorted(VECTOR_DIR.glob("*.yaml")):
        if path.name == "shared_broad_persona_questions.yaml":
            continue
        data = parse_simple_yaml(path)
        trait_id = data.get("trait_id")
        if not isinstance(trait_id, str) or not trait_id:
            raise ValueError(f"{path}: missing trait_id")
        seen_traits.add(trait_id)
        for scalar_key in ["description", "orientation", "judge_rubric"]:
            if not isinstance(data.get(scalar_key), str) or not data[scalar_key]:
                raise ValueError(f"{path}: missing scalar key {scalar_key}")
        require_list_count(path, data, "positive_instructions", 5)
        require_list_count(path, data, "negative_instructions", 5)
        if trait_id in BROAD_TRAITS:
            if data.get("question_set_ref") != str(shared_path):
                raise ValueError(f"{path}: broad trait must reference {shared_path}")
            if "extraction_questions" in data or "validation_questions" in data:
                raise ValueError(f"{path}: broad trait should use shared question_set_ref, not inline questions")
        elif trait_id in FAMILY_TRAITS:
            require_list_count(path, data, "extraction_questions", 20)
            require_list_count(path, data, "validation_questions", 20)
        else:
            raise ValueError(f"{path}: unexpected trait_id {trait_id}")

    expected = BROAD_TRAITS | FAMILY_TRAITS
    if seen_traits != expected:
        raise ValueError(f"trait set mismatch: seen={sorted(seen_traits)} expected={sorted(expected)}")
    print(f"validated {len(seen_traits)} vector configs plus shared broad-persona question set")


if __name__ == "__main__":
    main()

