"""Tests for model context limit resolution and generation clamping."""

from __future__ import annotations

import json
from pathlib import Path

from minicpm_container.model_limits import (
    FALLBACK_MAX_POSITION_EMBEDDINGS,
    effective_max_new_tokens,
    resolve_max_position_embeddings,
)


def test_resolve_max_position_embeddings_uses_fallback_when_path_missing() -> None:
    assert resolve_max_position_embeddings(None) == FALLBACK_MAX_POSITION_EMBEDDINGS
    assert resolve_max_position_embeddings(Path("/nonexistent/model")) == (
        FALLBACK_MAX_POSITION_EMBEDDINGS
    )


def test_resolve_max_position_embeddings_reads_config_json(tmp_path: Path) -> None:
    model_dir = tmp_path / "MiniCPM5-1B"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps({"max_position_embeddings": 65536}),
        encoding="utf-8",
    )
    assert resolve_max_position_embeddings(model_dir) == 65536


def test_resolve_max_position_embeddings_falls_back_on_invalid_config(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("not json", encoding="utf-8")
    assert resolve_max_position_embeddings(model_dir) == FALLBACK_MAX_POSITION_EMBEDDINGS


def test_effective_max_new_tokens_returns_requested_when_room_available() -> None:
    assert (
        effective_max_new_tokens(
            512,
            max_position_embeddings=131_072,
            input_token_count=100,
        )
        == 512
    )


def test_effective_max_new_tokens_clamps_to_remaining_context() -> None:
    assert (
        effective_max_new_tokens(
            10_000,
            max_position_embeddings=131_072,
            input_token_count=130_000,
        )
        == 1_072
    )


def test_effective_max_new_tokens_returns_at_least_one() -> None:
    assert (
        effective_max_new_tokens(
            500,
            max_position_embeddings=100,
            input_token_count=200,
        )
        == 1
    )


def test_fallback_matches_minicpm5_context_length() -> None:
    assert FALLBACK_MAX_POSITION_EMBEDDINGS == 131_072
