"""Live Podman MLflow mirror checks against PostgreSQL source facts."""

from __future__ import annotations

import os
from pathlib import Path

import psycopg
import pytest

from rlredteam.observability import (
    connect_mlflow_client,
    load_observable_runs,
    sync_observable_run,
)
from rlredteam.storage.postgres_logger import connection_string, ensure_schema


def _record(root: Path):
    with psycopg.connect(connection_string()) as connection:
        ensure_schema(connection)
        records = load_observable_runs(connection, root, limit=1)
    if not records:
        pytest.skip("PostgreSQL contains no runs to mirror")
    return records[0]


@pytest.mark.skipif(
    not os.environ.get("MLFLOW_TRACKING_URI"), reason="MLflow service not configured"
)
def test_live_sync_is_idempotent_and_does_not_mutate_source() -> None:
    root = Path(__file__).resolve().parents[1]
    before = _record(root)
    source_before = before.to_dict()
    client = connect_mlflow_client()

    first = sync_observable_run(client, before, experiment_prefix="Security-RL-tests")
    second = sync_observable_run(client, before, experiment_prefix="Security-RL-tests")
    mirrored = client.get_run(first["run_id"])
    after = _record(root)

    assert first["run_id"] == second["run_id"]
    assert second["status"] == "unchanged"
    assert after.to_dict() == source_before
    assert mirrored.data.tags["security_rl_authority"] == (
        "postgresql_and_canonical_files"
    )
    assert mirrored.data.tags["security_rl_designation"] == before.designation
    assert mirrored.data.params["config_hash"] == before.parameters["config_hash"]
    assert mirrored.data.params["policy_hash"] == before.parameters["policy_hash"]

