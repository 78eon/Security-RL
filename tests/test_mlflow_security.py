"""Static security boundary checks for optional MLflow observability."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
OVERLAY = ROOT / "docker-compose.mlflow.yml"
pytestmark = pytest.mark.skipif(
    not OVERLAY.exists(),
    reason="Phase 21 deployment files are mounted only in the MLflow image",
)


def _compose() -> dict:
    return yaml.safe_load(OVERLAY.read_text())


def test_mlflow_package_is_exactly_pinned_and_final_user_is_non_root() -> None:
    dockerfile = (ROOT / "Dockerfile.mlflow").read_text()
    assert "mlflow==3.16.0" in dockerfile
    final_user = [line for line in dockerfile.splitlines() if line.startswith("USER ")][
        -1
    ]
    assert final_user not in {"USER 0", "USER root", "USER 0:0"}


def test_mlflow_port_is_loopback_only() -> None:
    ports = _compose()["services"]["mlflow"]["ports"]
    assert ports
    assert all(str(port).startswith("127.0.0.1:") for port in ports)


def test_mlflow_is_non_root_and_capability_dropped() -> None:
    service = _compose()["services"]["mlflow"]
    assert service["user"].split(":")[0] not in {"0", "root"}
    assert service["cap_drop"] == ["ALL"]
    assert "no-new-privileges:true" in service["security_opt"]


def test_mlflow_server_has_no_canonical_source_or_evidence_mounts() -> None:
    volumes = _compose()["services"]["mlflow"].get("volumes", [])
    assert volumes
    assert all(not str(value).startswith("./") for value in volumes)
    assert all("runs" not in str(value) for value in volumes)
    assert all("results" not in str(value) for value in volumes)


def test_mlflow_uses_the_offline_internal_network() -> None:
    base = yaml.safe_load((ROOT / "docker-compose.yml").read_text())
    service = _compose()["services"]["mlflow"]
    assert service["networks"] == ["rlredteam-internal"]
    assert base["networks"]["rlredteam-internal"]["internal"] is True


def test_mlflow_does_not_replace_postgresql_or_canonical_evidence() -> None:
    command = _compose()["services"]["mlflow"]["command"]
    backend_index = command.index("--backend-store-uri")
    assert command[backend_index + 1] == "sqlite:////mlflow/mlflow.db"
    assert all("postgres" not in str(value).lower() for value in command)
    assert "--default-artifact-root" in command


def test_mlflow_restricts_allowed_host_headers() -> None:
    command = _compose()["services"]["mlflow"]["command"]
    host_index = command.index("--allowed-hosts")
    assert command[host_index + 1] == "mlflow:5000,localhost,127.0.0.1"
    assert "*" not in command[host_index + 1]


def test_mlflow_has_one_worker_and_bounded_numeric_threads() -> None:
    service = _compose()["services"]["mlflow"]
    command = service["command"]
    worker_index = command.index("--workers")
    assert command[worker_index + 1] == "1"
    assert service["environment"] == {
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }
