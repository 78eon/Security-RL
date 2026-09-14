"""Static security boundary checks for the optional Neo4j service."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
OVERLAY = ROOT / "docker-compose.neo4j.yml"
pytestmark = pytest.mark.skipif(
    not OVERLAY.exists(),
    reason="Phase 20 deployment files are mounted only in the Neo4j client image",
)


def _compose() -> dict:
    return yaml.safe_load(OVERLAY.read_text())


def test_neo4j_image_is_exactly_pinned() -> None:
    image = _compose()["services"]["neo4j"]["image"]
    assert image == "docker.io/library/neo4j:5.26.0-community"


def test_client_driver_is_pinned_and_final_user_is_non_root() -> None:
    dockerfile = (ROOT / "Dockerfile.neo4j").read_text()
    assert "neo4j==5.28.5" in dockerfile
    final_user = [line for line in dockerfile.splitlines() if line.startswith("USER ")][-1]
    assert final_user not in {"USER 0", "USER root", "USER 0:0"}


def test_neo4j_credentials_are_required_without_password_default() -> None:
    raw = OVERLAY.read_text()
    assert "${NEO4J_USER:?" in raw
    assert "${NEO4J_PASSWORD:?" in raw
    assert "${NEO4J_PASSWORD:-" not in raw


def test_neo4j_ports_are_loopback_only() -> None:
    ports = _compose()["services"]["neo4j"]["ports"]
    assert ports
    assert all(str(port).startswith("127.0.0.1:") for port in ports)


def test_neo4j_is_non_root_and_capability_dropped() -> None:
    service = _compose()["services"]["neo4j"]
    assert service["user"].split(":")[0] not in {"0", "root"}
    assert service["cap_drop"] == ["ALL"]
    assert "no-new-privileges:true" in service["security_opt"]


def test_neo4j_has_no_canonical_source_or_evidence_mounts() -> None:
    volumes = _compose()["services"]["neo4j"].get("volumes", [])
    assert all(not str(value).startswith("./") for value in volumes)


def test_neo4j_uses_the_offline_internal_network() -> None:
    base = yaml.safe_load((ROOT / "docker-compose.yml").read_text())
    service = _compose()["services"]["neo4j"]
    assert service["networks"] == ["rlredteam-internal"]
    assert base["networks"]["rlredteam-internal"]["internal"] is True
