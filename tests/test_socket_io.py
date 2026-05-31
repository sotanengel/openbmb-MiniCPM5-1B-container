"""Tests for shared Unix socket helpers."""

from __future__ import annotations

from unittest.mock import MagicMock

from minicpm_container.protocol.socket_io import recv_all, send_json


def test_recv_all_reads_until_peer_closes() -> None:
    sock = MagicMock()
    sock.recv.side_effect = [b"hello ", b"world", b""]
    assert recv_all(sock) == b"hello world"


def test_recv_all_returns_empty_when_no_data() -> None:
    sock = MagicMock()
    sock.recv.return_value = b""
    assert recv_all(sock) == b""


def test_send_json_encodes_utf8() -> None:
    sock = MagicMock()
    send_json(sock, '{"content":"名探偵"}')
    sock.sendall.assert_called_once_with('{"content":"名探偵"}'.encode())
