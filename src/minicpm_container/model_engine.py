"""Transformers-based inference engine for MiniCPM5."""

from __future__ import annotations

import logging
from pathlib import Path

from minicpm_container.generation_decode import decode_generated_text
from minicpm_container.model_limits import (
    effective_max_new_tokens,
    resolve_max_position_embeddings,
)
from minicpm_container.protocol import ChatRequest, ChatResponse

logger = logging.getLogger(__name__)


class ModelEngine:
    def __init__(self, model_path: str) -> None:
        self.model_path = model_path
        self._tokenizer = None
        self._model = None
        self._max_position_embeddings = resolve_max_position_embeddings(Path(model_path))

    def load(self) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info("Loading model from %s", self.model_path)
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=False,
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype="auto",
            device_map="cpu",
            trust_remote_code=False,
        )
        config_max = getattr(self._model.config, "max_position_embeddings", None)
        if isinstance(config_max, int) and config_max > 0:
            self._max_position_embeddings = config_max
        logger.info(
            "Model loaded (max_position_embeddings=%s)",
            self._max_position_embeddings,
        )

    def generate(self, request: ChatRequest) -> ChatResponse:
        if self._tokenizer is None or self._model is None:
            return ChatResponse(content="", error="model not loaded")

        try:
            template_kwargs: dict[str, object] = {}
            if request.tools:
                template_kwargs["tools"] = request.tools
            if request.enable_thinking is not None:
                template_kwargs["enable_thinking"] = request.enable_thinking
            inputs = self._tokenizer.apply_chat_template(
                request.messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
                **template_kwargs,
            )
            inputs = inputs.to(self._model.device)
            input_length = inputs["input_ids"].shape[-1]
            max_new_tokens = effective_max_new_tokens(
                request.max_new_tokens,
                max_position_embeddings=self._max_position_embeddings,
                input_token_count=input_length,
            )
            if max_new_tokens != request.max_new_tokens:
                logger.info(
                    "Clamped max_new_tokens from %s to %s (input_tokens=%s, context=%s)",
                    request.max_new_tokens,
                    max_new_tokens,
                    input_length,
                    self._max_position_embeddings,
                )
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=request.do_sample,
                temperature=request.temperature,
                top_p=request.top_p,
            )
            generated = outputs[0][input_length:]
            content = decode_generated_text(self._tokenizer, generated)
            return ChatResponse(content=content)
        except Exception as exc:  # noqa: BLE001 - return safe error to client
            logger.exception("Generation failed")
            return ChatResponse(content="", error=str(exc))
