"""Model context limits for max_new_tokens validation and clamping."""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# openbmb/MiniCPM5-1B model card: context length 131,072
FALLBACK_MAX_POSITION_EMBEDDINGS = 131_072

_CONFIG_KEYS = (
    "max_position_embeddings",
    "model_max_length",
    "max_seq_len",
    "max_sequence_length",
)


def _read_positive_int_from_config(data: object) -> int | None:
    if not isinstance(data, dict):
        return None
    for key in _CONFIG_KEYS:
        value = data.get(key)
        if isinstance(value, int) and value > 0:
            return value
    return None


def resolve_max_position_embeddings(model_path: Path | None = None) -> int:
    """Return max context length from model config.json or the baked-in fallback."""
    if model_path is None:
        return FALLBACK_MAX_POSITION_EMBEDDINGS

    config_path = model_path / "config.json"
    if not config_path.is_file():
        return FALLBACK_MAX_POSITION_EMBEDDINGS

    try:
        raw = config_path.read_text(encoding="utf-8")
        data = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        logger.debug("Could not read model config at %s: %s", config_path, exc)
        return FALLBACK_MAX_POSITION_EMBEDDINGS

    resolved = _read_positive_int_from_config(data)
    if resolved is None:
        return FALLBACK_MAX_POSITION_EMBEDDINGS
    return resolved


def effective_max_new_tokens(
    requested: int,
    *,
    max_position_embeddings: int,
    input_token_count: int,
) -> int:
    """Clamp requested new tokens to remaining context window."""
    remaining = max_position_embeddings - input_token_count
    if remaining < 1:
        return 1
    return max(1, min(requested, remaining))
