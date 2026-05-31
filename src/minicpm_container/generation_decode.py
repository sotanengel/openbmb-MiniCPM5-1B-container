"""Decode model token ids without stripping MiniCPM5 tool-call tokens."""

from __future__ import annotations

# MiniCPM5 registers XML/tool markup as added special tokens. skip_special_tokens=True
# removes <function>, <param>, <tool_call>, etc., breaking the tool parser.
import re

_THINKING_BLOCK_RE = re.compile(
    r"<think>[\s\S]*?</think>",
    re.IGNORECASE,
)

_GENERATION_PREFIXES = (
    "<think>\n\n</think>\n\n",
    "<think>\n</think>\n\n",
)

_TRAILING_MARKERS = (
    "<|im_end|>",
    "<|im_start|>",
    "</s>",
    "<s>",
)


def strip_thinking_blocks(text: str) -> str:
    """Remove MiniCPM5 reasoning blocks from model output."""
    without_blocks = _THINKING_BLOCK_RE.sub("", text)
    if "</think>" in without_blocks:
        without_blocks = without_blocks.split("</think>")[-1]
    if "<think>" in without_blocks:
        without_blocks = without_blocks.split("<think>")[0]
    return without_blocks


def strip_generation_artifacts(text: str) -> str:
    """Remove empty thinking blocks and stray chat control tokens from decoded text."""
    stripped = strip_thinking_blocks(text)
    changed = True
    while changed:
        changed = False
        for prefix in _GENERATION_PREFIXES:
            if stripped.startswith(prefix):
                stripped = stripped[len(prefix) :]
                changed = True
        stripped = stripped.lstrip("\n")
    for marker in _TRAILING_MARKERS:
        if stripped.endswith(marker):
            stripped = stripped[: -len(marker)]
            changed = True
    return stripped.rstrip()


def decode_generated_text(tokenizer, token_ids) -> str:
    """Decode generated tokens preserving tool-call special tokens."""
    text = tokenizer.decode(token_ids, skip_special_tokens=False)
    return strip_generation_artifacts(text).strip()
