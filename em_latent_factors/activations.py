"""Activation extraction primitives.

This module intentionally separates cheap, testable layer/token-span logic from
expensive model forward passes. Extraction scripts should use these helpers and
pool immediately rather than storing full token hidden states.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Any

from em_latent_factors.config import load_yaml


@dataclass(frozen=True)
class LayerSelection:
    num_hidden_layers: int
    logical_layers: list[int]
    hidden_state_indices: list[int]
    quantiles: list[float]
    indexing: str = "transformer_blocks_1_based"

    def to_json(self) -> dict[str, Any]:
        return {
            "num_hidden_layers": self.num_hidden_layers,
            "logical_layers": self.logical_layers,
            "hidden_state_indices": self.hidden_state_indices,
            "quantiles": self.quantiles,
            "indexing": self.indexing,
        }


@dataclass(frozen=True)
class TokenSpan:
    start: int
    end: int
    name: str

    def to_slice(self) -> slice:
        return slice(self.start, self.end)

    @property
    def length(self) -> int:
        return max(0, self.end - self.start)


def resolve_layers_from_config(
    num_hidden_layers: int,
    config_path: str = "configs/experiment.yaml",
) -> LayerSelection:
    config = load_yaml(config_path).get("activation_extraction", {})
    indexing = config.get("layer_indexing", "transformer_blocks_1_based")
    if config.get("layers") == "auto" or config.get("layer_selection") == "quantiles":
        quantiles = [float(q) for q in config.get("quantiles", [0.25, 0.5, 0.75, 1.0])]
        logical_layers = resolve_quantile_layers(num_hidden_layers, quantiles)
    else:
        logical_layers = [int(x) for x in config["layers"]]
        quantiles = []
    hidden_state_indices = logical_to_hidden_state_indices(logical_layers, num_hidden_layers, indexing=indexing)
    return LayerSelection(
        num_hidden_layers=num_hidden_layers,
        logical_layers=logical_layers,
        hidden_state_indices=hidden_state_indices,
        quantiles=quantiles,
        indexing=indexing,
    )


def resolve_quantile_layers(num_hidden_layers: int, quantiles: list[float]) -> list[int]:
    if num_hidden_layers <= 0:
        raise ValueError("num_hidden_layers must be positive")
    layers = []
    for quantile in quantiles:
        if quantile <= 0 or quantile > 1:
            raise ValueError(f"quantile must be in (0, 1], got {quantile}")
        # 1-based transformer block index. ceil gives 0.25 of 28 -> 7.
        layer = max(1, min(num_hidden_layers, ceil(num_hidden_layers * quantile)))
        layers.append(layer)
    return sorted(dict.fromkeys(layers))


def logical_to_hidden_state_indices(
    logical_layers: list[int],
    num_hidden_layers: int,
    indexing: str = "transformer_blocks_1_based",
) -> list[int]:
    if indexing != "transformer_blocks_1_based":
        raise ValueError(f"unsupported layer indexing: {indexing}")
    indices = []
    for layer in logical_layers:
        if layer < 1 or layer > num_hidden_layers:
            raise ValueError(f"layer {layer} outside 1..{num_hidden_layers}")
        # HF hidden_states[0] is embedding output, hidden_states[k] is block k.
        indices.append(layer)
    return indices


def infer_num_hidden_layers_from_model_config(model_config: Any) -> int:
    for attr in ("num_hidden_layers", "n_layer", "num_layers"):
        value = getattr(model_config, attr, None)
        if value is not None:
            return int(value)
    raise ValueError("could not infer num_hidden_layers from model config")


def prompt_last_span(attention_mask_row) -> TokenSpan:
    length = int(attention_mask_row.sum().item())
    if length <= 0:
        raise ValueError("cannot select last token from empty prompt")
    return TokenSpan(start=length - 1, end=length, name="prompt_last")


def prompt_avg_span(attention_mask_row) -> TokenSpan:
    length = int(attention_mask_row.sum().item())
    if length <= 0:
        raise ValueError("cannot select prompt span from empty prompt")
    return TokenSpan(start=0, end=length, name="prompt_avg")


def response_span_from_lengths(prompt_len: int, full_len: int) -> TokenSpan:
    if prompt_len < 0 or full_len < 0:
        raise ValueError("token lengths must be non-negative")
    if full_len <= prompt_len:
        raise ValueError(f"full sequence length {full_len} must exceed prompt length {prompt_len}")
    return TokenSpan(start=prompt_len, end=full_len, name="response_avg")


def token_ids(tokenizer, text: str) -> list[int]:
    encoded = tokenizer(text, add_special_tokens=False)
    return list(encoded["input_ids"])


def pool_hidden_state(hidden_state, span: TokenSpan, mode: str):
    """Pool one sequence hidden-state tensor over a token span.

    `hidden_state` is expected to have shape [seq_len, hidden_dim].
    """
    if span.length <= 0:
        raise ValueError(f"empty token span for {span.name}")
    selected = hidden_state[span.to_slice()]
    if mode in {"response_avg", "prompt_avg"}:
        return selected.mean(dim=0)
    if mode == "prompt_last":
        return selected[-1]
    raise ValueError(f"unsupported pooling mode: {mode}")


def build_prompt_and_full_ids_for_response(tokenizer, messages: list[dict[str, str]], response: str) -> tuple[list[int], list[int], TokenSpan]:
    """Tokenize prompt-only and full conversation for response-token pooling."""
    response = str(response)
    if not response.strip():
        raise ValueError("cannot extract response activations from an empty response")
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prompt_ids = token_ids(tokenizer, prompt_text)
    full_messages = list(messages) + [{"role": "assistant", "content": response}]
    full_text = tokenizer.apply_chat_template(full_messages, tokenize=False, add_generation_prompt=False)
    full_ids = token_ids(tokenizer, full_text)
    response_char_start = full_text.rfind(response)
    if response_char_start >= 0:
        response_start = len(token_ids(tokenizer, full_text[:response_char_start]))
        span = response_span_from_lengths(prompt_len=response_start, full_len=len(full_ids))
    else:
        span = response_span_from_lengths(prompt_len=len(prompt_ids), full_len=len(full_ids))
    return prompt_ids, full_ids, span
