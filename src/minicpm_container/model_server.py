"""Unix socket model inference worker."""

from __future__ import annotations

import logging
import os
import socket
import sys
from pathlib import Path

from minicpm_container.protocol import (
    DEFAULT_SOCKET_PATH,
    ChatRequest,
    ChatResponse,
    ProtocolError,
)

logger = logging.getLogger(__name__)

MODEL_PATH = os.environ.get("MODEL_PATH", "/models/MiniCPM5-1B")
SOCKET_PATH = os.environ.get("MODEL_SOCKET_PATH", DEFAULT_SOCKET_PATH)
SOCKET_GROUP = os.environ.get("MODEL_SOCKET_GROUP", "chat")


class ModelEngine:
    def __init__(self, model_path: str) -> None:
        self.model_path = model_path
        self._tokenizer = None
        self._model = None

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
        logger.info("Model loaded")

    def generate(self, request: ChatRequest) -> ChatResponse:
        if self._tokenizer is None or self._model is None:
            return ChatResponse(content="", error="model not loaded")

        try:
            inputs = self._tokenizer.apply_chat_template(
                request.messages,
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=request.enable_thinking,
                return_dict=True,
                return_tensors="pt",
            )
            inputs = inputs.to(self._model.device)
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=request.max_new_tokens,
                do_sample=request.do_sample,
                temperature=request.temperature,
                top_p=request.top_p,
            )
            input_length = inputs["input_ids"].shape[-1]
            generated = outputs[0][input_length:]
            content = self._tokenizer.decode(generated, skip_special_tokens=True)
            return ChatResponse(content=content.strip())
        except Exception as exc:  # noqa: BLE001 - return safe error to client
            logger.exception("Generation failed")
            return ChatResponse(content="", error=str(exc))


def _configure_socket_permissions(socket_path: Path) -> None:
    socket_path.chmod(0o660)
    if hasattr(os, "chown"):
        import grp

        group = grp.getgrnam(SOCKET_GROUP)
        os.chown(socket_path, -1, group.gr_gid)


def _handle_client(connection: socket.socket, engine: ModelEngine) -> None:
    chunks: list[bytes] = []
    while True:
        chunk = connection.recv(65536)
        if not chunk:
            break
        chunks.append(chunk)

    if not chunks:
        response = ChatResponse(content="", error="empty request")
        connection.sendall(response.to_json().encode("utf-8"))
        return

    try:
        request = ChatRequest.from_json(b"".join(chunks).decode("utf-8"))
    except ProtocolError as exc:
        response = ChatResponse(content="", error=str(exc))
        connection.sendall(response.to_json().encode("utf-8"))
        return

    response = engine.generate(request)
    connection.sendall(response.to_json().encode("utf-8"))


def serve_forever(
    engine: ModelEngine,
    socket_path: str = SOCKET_PATH,
    backlog: int = 4,
) -> None:
    path = Path(socket_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()

    engine.load()

    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
        server.bind(str(path))
        _configure_socket_permissions(path)
        server.listen(backlog)
        logger.info("Model server listening on %s", path)

        while True:
            connection, _ = server.accept()
            with connection:
                _handle_client(connection, engine)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )

    model_path = Path(MODEL_PATH)
    if not model_path.is_dir():
        logger.error("Model path does not exist: %s", model_path)
        raise SystemExit(1)

    engine = ModelEngine(str(model_path))
    try:
        serve_forever(engine)
    except KeyboardInterrupt:
        logger.info("Shutting down model server")
        raise SystemExit(0) from None


if __name__ == "__main__":
    main()
