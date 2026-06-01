"""Tests for docker-compose security hardening."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
COMPOSE_PATH = ROOT / "docker-compose.yml"
GPU_COMPOSE_PATH = ROOT / "docker-compose.gpu.yml"


def _merge_service(base: dict, override: dict) -> dict:
    merged = dict(base)
    for key, value in override.items():
        if key == "environment":
            base_env = merged.get("environment") or {}
            if isinstance(base_env, list):
                base_env = {
                    item.split("=", 1)[0]: item.split("=", 1)[1]
                    for item in base_env
                    if "=" in item
                }
            override_env = value if isinstance(value, dict) else {}
            merged["environment"] = {**base_env, **override_env}
        else:
            merged[key] = value
    return merged


def _load_merged_service(*compose_paths: Path) -> dict:
    service: dict = {}
    for path in compose_paths:
        with path.open(encoding="utf-8") as handle:
            doc = yaml.safe_load(handle)
        override = doc["services"]["minicpm"]
        service = _merge_service(service, override) if service else dict(override)
    return service


@pytest.fixture
def compose_config() -> dict:
    assert COMPOSE_PATH.exists(), "docker-compose.yml is missing"
    with COMPOSE_PATH.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


@pytest.fixture
def gpu_merged_service() -> dict:
    assert GPU_COMPOSE_PATH.exists(), "docker-compose.gpu.yml is missing"
    return _load_merged_service(COMPOSE_PATH, GPU_COMPOSE_PATH)


def test_compose_uses_network_none(compose_config: dict) -> None:
    service = compose_config["services"]["minicpm"]
    assert service.get("network_mode") == "none"


def test_compose_is_read_only(compose_config: dict) -> None:
    service = compose_config["services"]["minicpm"]
    assert service.get("read_only") is True


def test_compose_drops_all_capabilities(compose_config: dict) -> None:
    service = compose_config["services"]["minicpm"]
    assert service.get("cap_drop") == ["ALL"]


def test_compose_disables_new_privileges(compose_config: dict) -> None:
    service = compose_config["services"]["minicpm"]
    security_opt = service.get("security_opt", [])
    assert "no-new-privileges:true" in security_opt


def test_compose_uses_tmpfs_for_runtime_dirs(compose_config: dict) -> None:
    service = compose_config["services"]["minicpm"]
    tmpfs = service.get("tmpfs", [])
    assert any(str(entry).startswith("/tmp") for entry in tmpfs)
    assert any(str(entry).startswith("/run") for entry in tmpfs)


def test_compose_does_not_publish_ports(compose_config: dict) -> None:
    service = compose_config["services"]["minicpm"]
    assert "ports" not in service


def test_gpu_compose_enables_gpus(gpu_merged_service: dict) -> None:
    assert gpu_merged_service.get("gpus") == "all"
    assert gpu_merged_service.get("environment", {}).get("MODEL_USE_GPU") == "1"


def test_gpu_compose_preserves_network_none(gpu_merged_service: dict) -> None:
    assert gpu_merged_service.get("network_mode") == "none"


def test_gpu_compose_preserves_read_only_and_capabilities(gpu_merged_service: dict) -> None:
    assert gpu_merged_service.get("read_only") is True
    assert gpu_merged_service.get("cap_drop") == ["ALL"]
    security_opt = gpu_merged_service.get("security_opt", [])
    assert "no-new-privileges:true" in security_opt
