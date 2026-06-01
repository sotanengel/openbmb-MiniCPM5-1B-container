"""Resolve inference device (CPU vs NVIDIA GPU) from environment and host."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from collections.abc import Mapping

from minicpm_container.env_utils import parse_env_bool

_VALID_DEVICE_MAPS = frozenset({"cpu", "auto", "cuda"})


def resolve_device_map(
    environ: Mapping[str, str] | None = None,
) -> str:
    """Return Hugging Face ``device_map`` value for model loading."""
    env = os.environ if environ is None else environ

    raw_map = env.get("MODEL_DEVICE_MAP")
    if raw_map is not None:
        normalized = raw_map.strip().lower()
        if normalized in _VALID_DEVICE_MAPS:
            return normalized

    use_gpu = parse_env_bool(env.get("MODEL_USE_GPU"), default=False)
    if use_gpu:
        return "auto"
    return "cpu"


def effective_device_map(requested: str) -> str:
    """Apply runtime CUDA availability; fall back to CPU when GPU was requested but unavailable."""
    if requested not in ("auto", "cuda"):
        return requested

    try:
        import torch
    except ImportError:
        return "cpu"

    if torch.cuda.is_available():
        return requested
    return "cpu"


def host_has_nvidia_gpu() -> bool:
    """True when ``nvidia-smi`` exists and runs successfully (build-time host check)."""
    smi = shutil.which("nvidia-smi")
    if smi is None:
        return False
    try:
        result = subprocess.run(
            [smi],
            check=False,
            capture_output=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0


def log_runtime_device_info(logger: logging.Logger) -> None:
    """Log whether CUDA is available and the active device name."""
    try:
        import torch
    except ImportError:
        logger.warning("PyTorch not installed; inference device unknown")
        return

    if torch.cuda.is_available():
        count = torch.cuda.device_count()
        name = torch.cuda.get_device_name(0) if count > 0 else "unknown"
        logger.info("CUDA available: device_count=%s, device0=%s", count, name)
    else:
        logger.info("CUDA not available; using CPU inference")
