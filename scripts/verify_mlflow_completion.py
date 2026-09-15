#!/usr/bin/env python3
"""Verify the MLflow mirror is idempotent and non-authoritative."""

from __future__ import annotations

import os
from pathlib import Path

import psycopg

from rlredteam.observability import (
    connect_mlflow_client,
    load_observable_runs,
    sync_observable_run,
)
from rlredteam.storage.postgres_logger import connection_string, ensure_schema


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    with psycopg.connect(connection_string()) as connection:
        ensure_schema(connection)
        before = load_observable_runs(connection, root, limit=3)
    if not before:
        raise SystemExit("no PostgreSQL run exists for MLflow verification")
    source_before = [record.to_dict() for record in before]
    client = connect_mlflow_client()
    prefix = os.environ.get("MLFLOW_EXPERIMENT_PREFIX") or "Security-RL"
    first = [sync_observable_run(client, item, experiment_prefix=prefix) for item in before]
    second = [sync_observable_run(client, item, experiment_prefix=prefix) for item in before]
    with psycopg.connect(connection_string()) as connection:
        after = load_observable_runs(connection, root, limit=3)
    source_after = [record.to_dict() for record in after]
    run_ids = [item["run_id"] for item in first]
    mirrored = [client.get_run(run_id) for run_id in run_ids]
    checks = {
        "postgresql_source_unchanged": source_before == source_after,
        "canonical_artifact_hashes_unchanged": [
            [artifact.sha256 for artifact in item.artifact_references] for item in before
        ]
        == [[artifact.sha256 for artifact in item.artifact_references] for item in after],
        "stable_mlflow_run_ids": run_ids == [item["run_id"] for item in second],
        "second_sync_is_idempotent": all(item["status"] == "unchanged" for item in second),
        "designation_preserved": all(
            run.data.params["designation"] == record.designation
            for run, record in zip(mirrored, before, strict=True)
        ),
        "hashes_logged": all(
            all(
                key in run.data.params
                for key in (
                    "config_hash",
                    "topology_config_hash",
                    "cve_snapshot_hash",
                    "git_commit",
                    "policy_hash",
                )
            )
            for run in mirrored
        ),
        "evaluation_and_training_remain_distinct": all(
            run.data.tags["security_rl_designation"] == record.designation
            for run, record in zip(mirrored, before, strict=True)
        ),
        "mlflow_declared_non_authoritative": all(
            run.data.tags["security_rl_authority"] == "postgresql_and_canonical_files"
            for run in mirrored
        ),
    }
    for name, passed in checks.items():
        print(f"[{'PASS' if passed else 'FAIL'}] {name}")
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
