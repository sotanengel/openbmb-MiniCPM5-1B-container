"""Tests for local model directory preparation and validation."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from minicpm_container.model_cache import (
    DEFAULT_MODEL_ID,
    ensure_model_dir,
    find_hf_cache_snapshot,
    is_complete_model_dir,
    materialize_model_dir,
)


def _write_minimal_model_dir(model_dir: Path, *, with_weights: bool = True) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "config.json").write_text(
        json.dumps({"model_type": "llama"}),
        encoding="utf-8",
    )
    if with_weights:
        (model_dir / "model-00000-of-00001.safetensors").write_bytes(b"weights")


def test_is_complete_model_dir_false_when_missing() -> None:
    assert is_complete_model_dir(Path("/nonexistent/model")) is False


def test_is_complete_model_dir_false_when_empty(tmp_path: Path) -> None:
    model_dir = tmp_path / "empty"
    model_dir.mkdir()
    assert is_complete_model_dir(model_dir) is False


def test_is_complete_model_dir_false_when_config_only(tmp_path: Path) -> None:
    model_dir = tmp_path / "partial"
    _write_minimal_model_dir(model_dir, with_weights=False)
    assert is_complete_model_dir(model_dir) is False


def test_is_complete_model_dir_true_with_safetensors(tmp_path: Path) -> None:
    model_dir = tmp_path / "complete"
    _write_minimal_model_dir(model_dir)
    assert is_complete_model_dir(model_dir) is True


def test_is_complete_model_dir_true_with_index_json(tmp_path: Path) -> None:
    model_dir = tmp_path / "indexed"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}", encoding="utf-8")
    (model_dir / "model.safetensors.index.json").write_text("{}", encoding="utf-8")
    assert is_complete_model_dir(model_dir) is True


def test_materialize_model_dir_copies_files(tmp_path: Path) -> None:
    source = tmp_path / "source"
    target = tmp_path / "target"
    _write_minimal_model_dir(source)
    (source / "tokenizer_config.json").write_text("{}", encoding="utf-8")

    materialize_model_dir(source, target)

    assert is_complete_model_dir(target)
    assert (target / "tokenizer_config.json").is_file()
    assert not (target / "config.json").is_symlink()


def test_materialize_model_dir_resolves_symlinks(tmp_path: Path) -> None:
    blob = tmp_path / "blob"
    blob.mkdir()
    (blob / "weights.bin").write_bytes(b"data")

    source = tmp_path / "source"
    source.mkdir()
    (source / "config.json").write_text("{}", encoding="utf-8")
    (source / "model-00000-of-00001.safetensors").symlink_to(blob / "weights.bin")

    target = tmp_path / "target"
    materialize_model_dir(source, target)

    assert (target / "model-00000-of-00001.safetensors").is_file()
    assert not (target / "model-00000-of-00001.safetensors").is_symlink()
    assert (target / "model-00000-of-00001.safetensors").read_bytes() == b"data"


def test_find_hf_cache_snapshot_returns_none_when_cache_missing() -> None:
    with patch.dict(os.environ, {"HF_HOME": "/nonexistent/hf"}, clear=False):
        assert find_hf_cache_snapshot(DEFAULT_MODEL_ID) is None


def test_find_hf_cache_snapshot_finds_latest_revision(tmp_path: Path) -> None:
    repo_cache = tmp_path / "hub" / "models--openbmb--MiniCPM5-1B" / "snapshots"
    older = repo_cache / "aaa111"
    newer = repo_cache / "bbb222"
    _write_minimal_model_dir(older)
    _write_minimal_model_dir(newer)
    older.touch()
    import time

    time.sleep(0.02)
    newer.touch()

    with patch.dict(os.environ, {"HF_HOME": str(tmp_path)}, clear=False):
        found = find_hf_cache_snapshot(DEFAULT_MODEL_ID)

    assert found is not None
    assert found.name == "bbb222"


def test_ensure_model_dir_returns_true_when_already_complete(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    _write_minimal_model_dir(model_dir)

    assert ensure_model_dir(model_dir) is True


def test_ensure_model_dir_materializes_from_hf_cache(tmp_path: Path) -> None:
    hf_home = tmp_path / "hf"
    repo_cache = hf_home / "hub" / "models--openbmb--MiniCPM5-1B" / "snapshots" / "rev1"
    _write_minimal_model_dir(repo_cache)

    model_dir = tmp_path / "model"
    with patch.dict(os.environ, {"HF_HOME": str(hf_home)}, clear=False):
        assert ensure_model_dir(model_dir) is True

    assert is_complete_model_dir(model_dir)


def test_ensure_model_dir_extracts_from_docker_image(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    source = tmp_path / "extracted"
    _write_minimal_model_dir(source)

    def fake_run(
        cmd: list[str],
        *,
        check: bool,
        capture_output: bool,
        text: bool,
    ) -> subprocess.CompletedProcess[str]:
        del check, capture_output, text
        if cmd[:3] == ["docker", "image", "inspect"]:
            return subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr="")
        if cmd[:2] == ["docker", "create"]:
            return subprocess.CompletedProcess(cmd, 0, stdout="cid123\n", stderr="")
        if cmd[:2] == ["docker", "cp"]:
            container_spec, dest = cmd[2], Path(cmd[3])
            assert container_spec == "cid123:/models/MiniCPM5-1B/."
            materialize_model_dir(source, dest)
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        if cmd[:2] == ["docker", "rm"]:
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        raise AssertionError(f"unexpected command: {cmd}")

    with patch("minicpm_container.model_cache.subprocess.run", side_effect=fake_run):
        assert ensure_model_dir(model_dir, image="minicpm5-1b-chat:latest") is True

    assert is_complete_model_dir(model_dir)


def test_ensure_model_dir_returns_false_when_no_sources(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    with patch.dict(os.environ, {"HF_HOME": str(tmp_path / "empty-hf")}, clear=False):
        with patch(
            "minicpm_container.model_cache.subprocess.run",
            side_effect=FileNotFoundError("docker"),
        ):
            assert ensure_model_dir(model_dir, image="missing:latest") is False


def test_ensure_model_dir_skips_docker_when_image_missing(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"

    def fake_run(
        cmd: list[str],
        *,
        check: bool,
        capture_output: bool,
        text: bool,
    ) -> subprocess.CompletedProcess[str]:
        del capture_output, text
        if cmd[:3] == ["docker", "image", "inspect"]:
            raise subprocess.CalledProcessError(1, cmd)
        raise AssertionError(f"unexpected command: {cmd}")

    with patch.dict(os.environ, {"HF_HOME": str(tmp_path / "empty-hf")}, clear=False):
        with patch("minicpm_container.model_cache.subprocess.run", side_effect=fake_run):
            assert ensure_model_dir(model_dir, image="missing:latest") is False


def test_model_cache_cli_main(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    model_dir = tmp_path / "model"
    _write_minimal_model_dir(model_dir)

    from minicpm_container.model_cache import main

    with patch("sys.argv", ["model_cache", "ensure", str(model_dir)]):
        assert main() == 0

    output = capsys.readouterr().out
    assert "already complete" in output.lower() or "complete" in output.lower()
