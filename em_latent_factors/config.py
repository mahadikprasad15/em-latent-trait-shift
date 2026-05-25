"""Config loading helpers.

The repo uses YAML configs. In the intended environment PyYAML should be
installed, but this fallback parser handles the small YAML subset used by the
current configs so validation and data plumbing work before dependency setup.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    try:
        import yaml  # type: ignore

        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return data or {}
    except ModuleNotFoundError:
        return _load_simple_yaml(path)


def _load_simple_yaml(path: Path) -> dict[str, Any]:
    root: dict[str, Any] = {}
    stack: list[tuple[int, Any]] = [(-1, root)]
    key_stack: list[tuple[int, str]] = []

    lines = path.read_text(encoding="utf-8").splitlines()
    for raw in lines:
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        text = raw.strip()

        while stack and indent <= stack[-1][0]:
            stack.pop()
        while key_stack and indent <= key_stack[-1][0]:
            key_stack.pop()

        parent = stack[-1][1]
        if text.startswith("- "):
            item_text = text[2:].strip()
            if not isinstance(parent, list):
                if not key_stack:
                    raise ValueError(f"{path}: list item has no list parent: {raw}")
                grandparent = stack[-2][1]
                key = key_stack[-1][1]
                new_list: list[Any] = []
                grandparent[key] = new_list
                stack[-1] = (stack[-1][0], new_list)
                parent = new_list
            item = _parse_scalar(item_text) if item_text else {}
            parent.append(item)
            if isinstance(item, dict):
                stack.append((indent, item))
            continue

        if ":" not in text:
            raise ValueError(f"{path}: unsupported YAML line: {raw}")
        key, value_text = text.split(":", 1)
        key = key.strip()
        value_text = value_text.strip()
        if not isinstance(parent, dict):
            raise ValueError(f"{path}: mapping under non-dict parent: {raw}")

        if value_text:
            parent[key] = _parse_scalar(value_text)
        else:
            child: dict[str, Any] = {}
            parent[key] = child
            stack.append((indent, child))
            key_stack.append((indent, key))
            continue
        key_stack.append((indent, key))

    return root


def _parse_scalar(value: str) -> Any:
    value = value.strip()
    if value in {"true", "false"}:
        return value == "true"
    if value == "null":
        return None
    if value.startswith("[") or value.startswith("{"):
        try:
            return ast.literal_eval(value)
        except Exception:
            # YAML-style inline maps/lists in this repo are simple enough that
            # adding quotes around bare keys is not worth doing in the fallback.
            return value
    try:
        if "." in value or "e" in value.lower():
            return float(value)
        return int(value)
    except ValueError:
        return value.strip('"').strip("'")

