"""Optional MLflow mirror of PostgreSQL-authoritative experiment facts."""

from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import psycopg
from psycopg.rows import dict_row

try:  # Present only in the dedicated Phase 21 Podman image.
    from mlflow import MlflowClient
    from mlflow.entities import Metric, Param, RunTag
except ModuleNotFoundError:  # pragma: no cover - canonical image boundary
    MlflowClient = None  # type: ignore[assignment,misc]
    Metric = Param = RunTag = None  # type: ignore[assignment,misc]

MLFLOW_MIRROR_SCHEMA_VERSION = "security-rl-mlflow-mirror-v1"


class ObservabilityError(RuntimeError):
    """MLflow mirroring cannot preserve the authoritative source semantics."""


@dataclass(frozen=True, slots=True)
class ArtifactReference:
    path: str
    sha256: str
    size_bytes: int
    kind: str


@dataclass(frozen=True, slots=True)
class ObservableRun:
    source_key: str
    postgres_experiment_id: int
    postgres_run_id: int
    experiment_name: str
    run_name: str
    designation: str
    status: str
    parameters: dict[str, str]
    metrics: dict[str, float]
    artifact_references: tuple[ArtifactReference, ...]
    mirror_hash: str

    def to_dict(self) -> dict[str, Any]:
        return json.loads(json.dumps(asdict(self), sort_keys=True, allow_nan=False))


def load_observable_runs(
    connection: psycopg.Connection,
    repo_root: Path,
    *,
    run_id: int | None = None,
    limit: int = 20,
) -> tuple[ObservableRun, ...]:
    if run_id is None and not 1 <= int(limit) <= 500:
        raise ValueError("MLflow synchronization limit must be between 1 and 500")
    where = "WHERE r.id = %s" if run_id is not None else ""
    tail = "" if run_id is not None else "ORDER BY r.id DESC LIMIT %s"
    params = (int(run_id),) if run_id is not None else (int(limit),)
    query = f"""
        SELECT e.id AS experiment_id, r.id AS run_id, e.name, e.condition,
               e.algorithm, e.reward_mode, e.config_hash,
               e.topology_config_hash, e.topology_id, e.topology_hash,
               e.cve_manifest_sha256, e.git_sha, e.seed_set,
               e.hyperparameters, r.seed, r.designation, r.status,
               r.evaluation_seeds, r.checkpoint_path,
               count(ep.id) AS episode_count,
               avg(ep.goal_reached::int) AS success_rate,
               avg(ep.total_reward) AS mean_total_reward,
               avg(ep.native_reward) AS mean_native_reward,
               avg(ep.length) AS mean_episode_length,
               avg(ep.discovery_coverage) AS mean_discovery_coverage,
               sum(ep.failed_actions) AS failed_action_count,
               max(ep.hosts_compromised) AS max_hosts_compromised
        FROM runs r
        JOIN experiments e ON e.id = r.experiment_id
        LEFT JOIN episodes ep ON ep.run_id = r.id
        {where}
        GROUP BY e.id, r.id
        {tail}
    """
    with connection.cursor(row_factory=dict_row) as cursor:
        cursor.execute(query, params)
        rows = cursor.fetchall()
    return tuple(_observable_run(row, repo_root) for row in rows)


def _observable_run(row: dict[str, Any], repo_root: Path) -> ObservableRun:
    experiment_id = int(row["experiment_id"])
    run_id = int(row["run_id"])
    source_key = f"postgresql:{experiment_id}:{run_id}"
    designation = str(row["designation"])
    parameters = {
        "algorithm": str(row["algorithm"]),
        "reward_mode": str(row["reward_mode"]),
        "condition": str(row["condition"] or row["reward_mode"]),
        "designation": designation,
        "training_seed": str(row["seed"]),
        "seed_set": _canonical(list(row["seed_set"] or [])),
        "evaluation_seeds": _canonical(list(row["evaluation_seeds"] or [])),
        "config_hash": str(row["config_hash"]),
        "topology_config_hash": str(row["topology_config_hash"]),
        "topology_id": str(row["topology_id"] or "unknown"),
        "topology_hash": str(row["topology_hash"] or "unknown"),
        "cve_snapshot_hash": str(row["cve_manifest_sha256"]),
        "git_commit": str(row["git_sha"] or "unknown"),
        "ppo_hyperparameters": _canonical(dict(row["hyperparameters"] or {})),
    }
    artifact_references = _artifact_references(row.get("checkpoint_path"), repo_root)
    policy = next((item.sha256 for item in artifact_references if item.kind == "policy"), None)
    parameters["checkpoint_path"] = str(row.get("checkpoint_path") or "unknown")
    parameters["policy_hash"] = policy or "unknown"
    metrics = {
        "episode_count": float(row["episode_count"] or 0),
        "success_rate": _number(row["success_rate"]),
        "mean_total_reward": _number(row["mean_total_reward"]),
        "mean_native_reward": _number(row["mean_native_reward"]),
        "mean_episode_length": _number(row["mean_episode_length"]),
        "mean_discovery_coverage": _number(row["mean_discovery_coverage"]),
        "failed_action_count": _number(row["failed_action_count"]),
        "max_hosts_compromised": _number(row["max_hosts_compromised"]),
    }
    body = {
        "schema_version": MLFLOW_MIRROR_SCHEMA_VERSION,
        "source_key": source_key,
        "experiment_name": str(row["name"]),
        "designation": designation,
        "status": str(row["status"]),
        "parameters": parameters,
        "metrics": metrics,
        "artifact_references": [asdict(item) for item in artifact_references],
    }
    return ObservableRun(
        source_key=source_key,
        postgres_experiment_id=experiment_id,
        postgres_run_id=run_id,
        experiment_name=str(row["name"]),
        run_name=f"{row['name']}:{designation}:run-{run_id}",
        designation=designation,
        status=str(row["status"]),
        parameters=parameters,
        metrics=metrics,
        artifact_references=artifact_references,
        mirror_hash=_sha256(body),
    )


def sync_observable_run(
    client: Any,
    record: ObservableRun,
    *,
    experiment_prefix: str = "Security-RL",
) -> dict[str, Any]:
    """Idempotently mirror one run without uploading canonical artifacts."""
    experiment_name = f"{experiment_prefix}/{record.experiment_name}"
    experiment = client.get_experiment_by_name(experiment_name)
    experiment_id = (
        client.create_experiment(experiment_name)
        if experiment is None
        else experiment.experiment_id
    )
    matches = client.search_runs(
        [str(experiment_id)],
        filter_string=f"tags.security_rl_source_key = '{record.source_key}'",
        max_results=2,
    )
    if len(matches) > 1:
        raise ObservabilityError(f"duplicate MLflow mirrors for {record.source_key}")
    if matches:
        run = matches[0]
        if run.data.tags.get("security_rl_mirror_hash") == record.mirror_hash:
            return {"run_id": run.info.run_id, "status": "unchanged"}
        conflicts = {
            key: (run.data.params[key], value)
            for key, value in record.parameters.items()
            if key in run.data.params and run.data.params[key] != value
        }
        if conflicts:
            raise ObservabilityError(f"immutable MLflow parameter drift: {conflicts}")
        mlflow_run_id = run.info.run_id
    else:
        created = client.create_run(
            str(experiment_id),
            tags={
                "mlflow.runName": record.run_name,
                "security_rl_source_key": record.source_key,
                "security_rl_authority": "postgresql_and_canonical_files",
                "security_rl_designation": record.designation,
            },
        )
        mlflow_run_id = created.info.run_id
    client.log_batch(
        mlflow_run_id,
        metrics=[Metric(key, value, 0, 0) for key, value in sorted(record.metrics.items())],
        params=[Param(key, value) for key, value in sorted(record.parameters.items())],
        tags=[
            RunTag("security_rl_mirror_hash", record.mirror_hash),
            RunTag("security_rl_source_status", record.status),
            RunTag("security_rl_schema", MLFLOW_MIRROR_SCHEMA_VERSION),
        ],
    )
    reference_path = f"security_rl/references/{record.mirror_hash}.json"
    client.log_dict(mlflow_run_id, record.to_dict(), reference_path)
    return {
        "run_id": mlflow_run_id,
        "status": "created" if not matches else "updated",
        "reference_manifest": reference_path,
    }


def connect_mlflow_client(*, wait_seconds: float = 45.0) -> Any:
    if MlflowClient is None:
        raise ObservabilityError("MLflow is available only in the Phase 21 client image")
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        raise ObservabilityError("MLFLOW_TRACKING_URI is required")
    deadline = time.monotonic() + max(0.0, wait_seconds)
    health_url = f"{tracking_uri.rstrip('/')}/health"
    while True:
        try:
            with urllib.request.urlopen(health_url, timeout=2.0) as response:
                if response.status != 200:
                    raise ObservabilityError(
                        f"MLflow health check returned HTTP {response.status}"
                    )
            break
        except urllib.error.URLError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(min(1.0, max(0.0, deadline - time.monotonic())))
    client = MlflowClient(tracking_uri=tracking_uri)
    client.search_experiments(max_results=1)
    return client


def _artifact_references(checkpoint: Any, repo_root: Path) -> tuple[ArtifactReference, ...]:
    if not checkpoint:
        return ()
    root = repo_root.resolve()
    raw = Path(str(checkpoint))
    if raw.is_absolute() and str(raw).startswith("/app/"):
        raw = root / raw.relative_to("/app")
    elif not raw.is_absolute():
        raw = root / raw
    try:
        resolved = raw.resolve(strict=False)
        resolved.relative_to(root)
    except ValueError as exc:
        raise ObservabilityError("checkpoint reference escapes repository") from exc
    candidates = [resolved]
    if resolved.is_file():
        candidates.extend(
            resolved.parent / name
            for name in ("manifest.json", "summary.json", "config.snapshot.json")
        )
    output = []
    for path in sorted({value for value in candidates if value.is_file()}):
        output.append(
            ArtifactReference(
                path=str(path.relative_to(root)),
                sha256=_sha256_file(path),
                size_bytes=path.stat().st_size,
                kind="policy" if path == resolved else "provenance",
            )
        )
    return tuple(output)


def _number(value: Any) -> float:
    return 0.0 if value is None else float(value)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


__all__ = [
    "MLFLOW_MIRROR_SCHEMA_VERSION",
    "ArtifactReference",
    "ObservableRun",
    "ObservabilityError",
    "connect_mlflow_client",
    "load_observable_runs",
    "sync_observable_run",
]
