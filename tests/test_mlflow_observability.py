"""Deterministic, reference-only MLflow observability records."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from rlredteam.observability import ObservabilityError, _artifact_references, _observable_run


def _row(checkpoint_path: str) -> dict:
    return {
        "experiment_id": 12,
        "run_id": 34,
        "name": "frozen-evaluation",
        "condition": "shaped",
        "algorithm": "PPO",
        "reward_mode": "shaped",
        "config_hash": "config-sha",
        "topology_config_hash": "topology-config-sha",
        "topology_id": "enterprise-small",
        "topology_hash": "topology-sha",
        "cve_manifest_sha256": "cve-sha",
        "git_sha": "commit-sha",
        "seed_set": [11, 12],
        "hyperparameters": {"learning_rate": 0.0003, "n_steps": 128},
        "seed": 42,
        "designation": "evaluation",
        "status": "completed",
        "evaluation_seeds": [1001, 1002],
        "checkpoint_path": checkpoint_path,
        "episode_count": 2,
        "success_rate": 0.5,
        "mean_total_reward": 3.25,
        "mean_native_reward": 2.75,
        "mean_episode_length": 7.0,
        "mean_discovery_coverage": 0.75,
        "failed_action_count": 3,
        "max_hosts_compromised": 2,
    }


def test_observable_run_is_deterministic_and_reference_only(tmp_path: Path) -> None:
    checkpoint = tmp_path / "runs" / "policy.zip"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"frozen-policy-weights")
    (checkpoint.parent / "manifest.json").write_text('{"frozen": true}\n')
    source_before = {
        path: path.read_bytes() for path in checkpoint.parent.iterdir() if path.is_file()
    }

    first = _observable_run(_row("runs/policy.zip"), tmp_path)
    second = _observable_run(_row("runs/policy.zip"), tmp_path)

    assert first == second
    assert first.mirror_hash == second.mirror_hash
    assert first.source_key == "postgresql:12:34"
    assert first.designation == "evaluation"
    assert first.parameters["evaluation_seeds"] == "[1001,1002]"
    assert first.parameters["policy_hash"] == hashlib.sha256(
        b"frozen-policy-weights"
    ).hexdigest()
    assert {item.kind for item in first.artifact_references} == {
        "policy",
        "provenance",
    }
    assert all(not hasattr(item, "content") for item in first.artifact_references)
    assert source_before == {
        path: path.read_bytes() for path in checkpoint.parent.iterdir() if path.is_file()
    }


def test_missing_checkpoint_is_recorded_as_unknown_without_fabrication(
    tmp_path: Path,
) -> None:
    record = _observable_run(_row("runs/missing.zip"), tmp_path)
    assert record.artifact_references == ()
    assert record.parameters["policy_hash"] == "unknown"


@pytest.mark.parametrize(
    "checkpoint",
    ["../outside.zip", "/tmp/outside.zip", "/app/../outside.zip"],
)
def test_artifact_reference_cannot_escape_repository(
    tmp_path: Path, checkpoint: str
) -> None:
    with pytest.raises(ObservabilityError, match="escapes repository"):
        _artifact_references(checkpoint, tmp_path)

