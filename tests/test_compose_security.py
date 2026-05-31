"""Tests for docker-compose security hardening."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

COMPOSE_PATH = Path(__file__).resolve().parents[1] / "docker-compose.yml"


@pytest.fixture
def compose_config() -> dict:
    assert COMPOSE_PATH.exists(), "docker-compose.yml is missing"
    with COMPOSE_PATH.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


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
    assert "/tmp" in tmpfs
    assert "/run" in tmpfs


def test_compose_does_not_publish_ports(compose_config: dict) -> None:
    service = compose_config["services"]["minicpm"]
    assert "ports" not in service
