"""Shared Unix socket read/write helpers."""

from __future__ import annotations

import socket

_DEFAULT_CHUNK_SIZE = 65536


def recv_all(sock: socket.socket, chunk_size: int = _DEFAULT_CHUNK_SIZE) -> bytes:
    """Read from a socket until the peer closes the write side."""
    chunks: list[bytes] = []
    while True:
        chunk = sock.recv(chunk_size)
        if not chunk:
            break
        chunks.append(chunk)
    return b"".join(chunks)


def send_json(sock: socket.socket, payload: str) -> None:
    """Send a UTF-8 encoded JSON payload over a connected socket."""
    sock.sendall(payload.encode("utf-8"))
