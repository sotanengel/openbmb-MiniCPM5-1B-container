"""Tests for inference device resolution."""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import pytest

from minicpm_container.inference_device import (
    effective_device_map,
    host_has_nvidia_gpu,
    log_runtime_device_info,
    resolve_device_map,
)


@pytest.mark.parametrize(
    ("env", "expected"),
    [
        ({}, "cpu"),
        ({"MODEL_USE_GPU": "1"}, "auto"),
        ({"MODEL_USE_GPU": "true"}, "auto"),
        ({"MODEL_USE_GPU": "0"}, "cpu"),
        ({"MODEL_DEVICE_MAP": "auto"}, "auto"),
        ({"MODEL_DEVICE_MAP": "cpu"}, "cpu"),
        ({"MODEL_DEVICE_MAP": "cuda"}, "cuda"),
        ({"MODEL_DEVICE_MAP": "AUTO"}, "auto"),
    ],
)
def test_resolve_device_map(env: dict[str, str], expected: str) -> None:
    assert resolve_device_map(env) == expected


def test_resolve_device_map_device_map_overrides_use_gpu() -> None:
    env = {"MODEL_USE_GPU": "1", "MODEL_DEVICE_MAP": "cpu"}
    assert resolve_device_map(env) == "cpu"


def test_resolve_device_map_invalid_device_map_ignored() -> None:
    env = {"MODEL_DEVICE_MAP": "invalid", "MODEL_USE_GPU": "1"}
    assert resolve_device_map(env) == "auto"


@patch("minicpm_container.inference_device.shutil.which", return_value=None)
def test_host_has_nvidia_gpu_false_when_smi_missing(_which: MagicMock) -> None:
    assert host_has_nvidia_gpu() is False


@patch("minicpm_container.inference_device.subprocess.run")
@patch("minicpm_container.inference_device.shutil.which", return_value="/usr/bin/nvidia-smi")
def test_host_has_nvidia_gpu_true_when_smi_succeeds(
    _which: MagicMock,
    run: MagicMock,
) -> None:
    run.return_value = MagicMock(returncode=0)
    assert host_has_nvidia_gpu() is True


@patch("minicpm_container.inference_device.subprocess.run")
@patch("minicpm_container.inference_device.shutil.which", return_value="/usr/bin/nvidia-smi")
def test_host_has_nvidia_gpu_false_when_smi_fails(
    _which: MagicMock,
    run: MagicMock,
) -> None:
    run.return_value = MagicMock(returncode=1)
    assert host_has_nvidia_gpu() is False


def test_effective_device_map_falls_back_to_cpu_without_cuda() -> None:
    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = False

    with patch.dict("sys.modules", {"torch": mock_torch}):
        assert effective_device_map("auto") == "cpu"
        assert effective_device_map("cuda") == "cpu"
        assert effective_device_map("cpu") == "cpu"


def test_effective_device_map_keeps_gpu_when_cuda_available() -> None:
    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = True

    with patch.dict("sys.modules", {"torch": mock_torch}):
        assert effective_device_map("auto") == "auto"
        assert effective_device_map("cuda") == "cuda"


def test_log_runtime_device_info_cuda_available() -> None:
    import sys

    logger = logging.getLogger("test.inference_device")
    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = True
    mock_torch.cuda.device_count.return_value = 1
    mock_torch.cuda.get_device_name.return_value = "Test GPU"

    with patch.object(logger, "info") as info, patch.dict(sys.modules, {"torch": mock_torch}):
        log_runtime_device_info(logger)

    assert "CUDA available" in info.call_args[0][0]


def test_log_runtime_device_info_cpu_only() -> None:
    import sys

    logger = logging.getLogger("test.inference_device.cpu")
    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = False

    with patch.object(logger, "info") as info, patch.dict(sys.modules, {"torch": mock_torch}):
        log_runtime_device_info(logger)

    assert "CPU" in info.call_args[0][0]
