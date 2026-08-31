"""Evidence gate for the hidden-topology on-prem research milestone.

This module verifies existing, gated research artifacts.  It does not train a
policy, access a target network, or mutate the database.
"""

from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path
from typing import Any

from rlredteam.enterprise.generalisation import (
    OnPremExperimentConfig,
    distribution_manifest,
    sha256_file,
    synthetic_vulnerability_manifest,
)
from rlredteam.enterprise.onprem import OnPremGeneralisationSplit, OnPremTopologyConfig


class CompletionVerificationError(RuntimeError):
    """Raised when canonical evidence does not satisfy a completion invariant."""


def require(condition: bool, message: str) -> None:
    """Fail closed with a stable, operator-readable error."""
    if not condition:
        raise CompletionVerificationError(message)


def _json(path: Path) -> dict[str, Any] | list[Any]:
    require(path.is_file(), f"missing evidence file: {path}")
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise CompletionVerificationError(f"invalid JSON evidence: {path}: {exc}") from exc


def _csv(path: Path) -> list[dict[str, str]]:
    require(path.is_file(), f"missing evidence file: {path}")
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _git(repo_root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            ["git", *args],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise CompletionVerificationError(f"cannot verify Git provenance: {exc}") from exc


def _verify_training(repo_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    run_dir = repo_root / "runs" / "onprem-generalisation"
    manifest = _json(run_dir / "training_manifest.json")
    require(isinstance(manifest, dict), "training manifest must be a JSON object")
    checkpoint = run_dir / "model.zip"
    require(checkpoint.is_file(), "canonical on-prem checkpoint is absent")

    config = OnPremExperimentConfig.from_yaml(
        repo_root / "configs" / "experiments" / "onprem_generalisation.yaml"
    )
    topology_config = OnPremTopologyConfig.from_yaml(repo_root / "configs" / "onprem_topology.yaml")
    split = OnPremGeneralisationSplit()
    expected = {
        "algorithm": "MaskablePPO",
        "action_mask_source": "AgentKnowledge",
        "git_dirty": False,
        "experiment_config_hash": config.digest(),
        "topology_config_hash": topology_config.digest(),
        "train_topology_seeds": list(split.train),
        "validation_topology_seeds": list(split.validation),
        "test_topology_seeds": list(split.test),
        "checkpoint_sha256": sha256_file(checkpoint),
        "distribution": distribution_manifest(split, topology_config),
        "synthetic_vulnerability_manifest_sha256": synthetic_vulnerability_manifest(
            split, topology_config
        ),
    }
    for field, value in expected.items():
        require(manifest.get(field) == value, f"training provenance mismatch: {field}")
    require(
        int(manifest.get("actual_training_timesteps", 0))
        >= int(manifest.get("training_timesteps", 0))
        >= config.total_timesteps,
        "canonical training budget was not completed",
    )
    groups = [set(split.train), set(split.validation), set(split.test)]
    require(
        not (groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2]),
        "topology seed splits overlap",
    )
    training_commit = str(manifest.get("git_commit", ""))
    require(bool(training_commit), "training commit is absent from the manifest")
    require(
        _git(repo_root, "cat-file", "-e", f"{training_commit}^{{commit}}").returncode == 0,
        "training commit is not present in repository history",
    )
    require(
        _git(repo_root, "merge-base", "--is-ancestor", training_commit, "HEAD").returncode == 0,
        "training commit is not an ancestor of the current code",
    )
    return manifest, {
        "training_commit": training_commit,
        "checkpoint_sha256": expected["checkpoint_sha256"],
        "policy_sha256": manifest["policy_sha256"],
        "training_timesteps": manifest["actual_training_timesteps"],
        "topology_count": sum(len(group) for group in groups),
    }


def _verify_evaluation(
    repo_root: Path,
    split_name: str,
    manifest: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, int]]:
    output = repo_root / "results" / "onprem-generalisation" / split_name
    metadata = _json(output / "evaluation_metadata.json")
    summary = _json(output / "summary.json")
    attack_paths = _json(output / "attack_paths.json")
    rows = _csv(output / "episodes.csv")
    require(isinstance(metadata, dict), f"{split_name} metadata must be an object")
    require(isinstance(summary, dict), f"{split_name} summary must be an object")
    require(isinstance(attack_paths, list), f"{split_name} attack paths must be a list")

    expected_seeds = list(manifest[f"{split_name}_topology_seeds"])
    require(metadata.get("phase") == "frozen_evaluation", f"{split_name} is not frozen evaluation")
    require(metadata.get("split") == split_name, f"{split_name} metadata split mismatch")
    require(metadata.get("deterministic") is True, f"{split_name} evaluation is not deterministic")
    require(metadata.get("gradient_updates") is False, f"{split_name} evaluation updated gradients")
    require(
        metadata.get("action_masks_supplied_to_predict") is True,
        f"{split_name} evaluation did not supply AgentKnowledge masks",
    )
    require(metadata.get("topology_seeds") == expected_seeds, f"{split_name} seed set mismatch")
    require(
        metadata.get("policy_sha256_before")
        == metadata.get("policy_sha256_after")
        == manifest.get("policy_sha256"),
        f"{split_name} policy changed during evaluation",
    )
    require(
        metadata.get("checkpoint_sha256") == manifest.get("checkpoint_sha256"),
        f"{split_name} checkpoint hash mismatch",
    )
    embedded = metadata.get("training_manifest", {})
    for field in (
        "git_commit",
        "experiment_config_hash",
        "topology_config_hash",
        "checkpoint_sha256",
        "policy_sha256",
        "synthetic_vulnerability_manifest_sha256",
    ):
        require(embedded.get(field) == manifest.get(field), f"{split_name} manifest drift: {field}")

    require(len(rows) == len(expected_seeds), f"{split_name} episode count mismatch")
    row_seeds = [int(row["topology_seed"]) for row in rows]
    require(row_seeds == expected_seeds, f"{split_name} episode seed order/content mismatch")
    expected_hashes = manifest["distribution"][split_name]
    require(
        all(row["topology_hash"] == expected_hashes[row["topology_seed"]] for row in rows),
        f"{split_name} topology hash mismatch",
    )
    require(all(row["goal_reached"] == "True" for row in rows), f"{split_name} has failed goals")
    require(
        all(row["terminal_reason"] == "goal" for row in rows),
        f"{split_name} has non-goal terminals",
    )
    require(
        sum(int(row["invalid_mask_selections"]) for row in rows) == 0,
        f"{split_name} selected invalid masked actions",
    )

    trajectory_count = 0
    trajectory_keys: set[tuple[int, int]] = set()
    trajectories = output / "trajectories.jsonl"
    require(trajectories.is_file(), f"missing {split_name} trajectories")
    for line_number, line in enumerate(trajectories.read_text().splitlines(), start=1):
        try:
            step = json.loads(line)
        except json.JSONDecodeError as exc:
            raise CompletionVerificationError(
                f"invalid {split_name} trajectory line {line_number}: {exc}"
            ) from exc
        seed = int(step["topology_seed"])
        require(seed in expected_seeds, f"{split_name} trajectory contains an unknown seed")
        require(
            step["topology_hash"] == expected_hashes[str(seed)],
            f"{split_name} trajectory topology hash mismatch",
        )
        trajectory_keys.add((seed, int(step["evaluation_seed"])))
        trajectory_count += 1
    require(
        trajectory_count == int(metadata["database_step_count"]),
        f"{split_name} trajectory/database step count mismatch",
    )
    require(len(trajectory_keys) == len(rows), f"{split_name} trajectory episode coverage mismatch")

    require(len(attack_paths) == len(rows), f"{split_name} attack-path count mismatch")
    for path in attack_paths:
        steps = path.get("steps", [])
        require(bool(steps), f"{split_name} contains an empty attack path")
        require(steps[-1].get("action_kind") == "access_asset", "attack path lacks asset access")
        require(steps[-1].get("goal_reached") is True, "attack path does not reach its goal")
        require(all(step.get("success") is True for step in steps), "attack path contains failure")
        require(
            all(step.get("state_changed") is True for step in steps),
            "attack path contains a non-causal step",
        )

    require(int(summary["episode_count"]) == len(rows), f"{split_name} summary count mismatch")
    require(
        int(summary["topology_count"]) == len(expected_seeds),
        f"{split_name} topology count mismatch",
    )
    require(float(summary["success_rate"]) == 1.0, f"{split_name} success rate is incomplete")
    require(
        int(summary["invalid_mask_selections"]) == 0,
        f"{split_name} summary has invalid actions",
    )
    require(
        int(metadata["database_episode_count"]) == len(rows),
        f"{split_name} database episode count mismatch",
    )
    return metadata, {
        "episodes": len(rows),
        "steps": trajectory_count,
        "attack_paths": len(attack_paths),
    }


def _verify_fixed_baseline(repo_root: Path) -> dict[str, Any]:
    root = repo_root / "results" / "experiment_01"
    experiment = _json(root / "metadata" / "experiment.json")
    analysis = _json(root / "summaries" / "analysis.json")
    require(isinstance(experiment, dict), "fixed baseline metadata must be an object")
    require(isinstance(analysis, dict), "fixed baseline analysis must be an object")
    require(experiment.get("complete") is True, "fixed baseline package is incomplete")
    require(
        experiment.get("result_phase") == "dedicated_frozen_policy_evaluation",
        "fixed baseline does not contain dedicated evaluation",
    )
    config = experiment["experiment_config"]
    expected_training = list(config["training_seeds"])
    expected_evaluation = list(config["evaluation_seeds"])
    require(set(config["arms"]) == {"sparse", "shaped"}, "baseline arms changed")
    reward_hashes = experiment["frozen_inputs"]["reward_config_hash"]
    require(set(reward_hashes) == {"sparse", "shaped"}, "baseline reward hashes are incomplete")
    require(reward_hashes["sparse"] != reward_hashes["shaped"], "baseline rewards are identical")

    manifests: dict[tuple[str, int], dict[str, Any]] = {}
    raw = root / "raw"
    for arm in ("sparse", "shaped"):
        for seed in expected_training:
            run_name = f"experiment_01-{arm}-s{seed}-t{config['topology_seed']}"
            run_dir = raw / run_name
            metadata = _json(run_dir / "evaluation_metadata.json")
            rows = _csv(run_dir / "evaluation.csv")
            require(isinstance(metadata, dict), f"invalid baseline metadata: {run_name}")
            require(
                metadata.get("gradient_updates") is False,
                f"baseline updated gradients: {run_name}",
            )
            require(
                metadata.get("policy_sha256_before") == metadata.get("policy_sha256_after"),
                f"baseline policy changed in evaluation: {run_name}",
            )
            require(
                metadata.get("evaluation_seeds") == expected_evaluation,
                f"eval seeds drifted: {run_name}",
            )
            require(
                len(rows) == len(expected_evaluation),
                f"baseline episode count mismatch: {run_name}",
            )
            require(
                [int(row["evaluation_seed"]) for row in rows] == expected_evaluation,
                f"baseline raw evaluation seeds drifted: {run_name}",
            )
            manifests[(arm, seed)] = metadata["manifest"]

    controlled_fields = (
        "cve_manifest_sha256",
        "dependency_lock_hash",
        "environment_config_hash",
        "git_commit",
        "git_dirty",
        "ppo_config_hash",
        "topology_config_hash",
        "topology_hash",
        "topology_seed",
        "training_budget",
        "training_seed",
    )
    for seed in expected_training:
        sparse = manifests[("sparse", seed)]
        shaped = manifests[("shaped", seed)]
        for field in controlled_fields:
            require(
                sparse.get(field) == shaped.get(field),
                f"reward isolation drift: seed {seed}, {field}",
            )
        require(sparse.get("reward_mode") == "sparse", f"sparse label drift: seed {seed}")
        require(shaped.get("reward_mode") == "shaped", f"shaped label drift: seed {seed}")
        require(
            sparse.get("reward_config_hash") != shaped.get("reward_config_hash"),
            f"reward configuration is not isolated: seed {seed}",
        )

    require(analysis.get("complete") is True, "baseline paired analysis is incomplete")
    require(analysis.get("arms") == {"sparse": 10, "shaped": 10}, "baseline arm counts changed")
    require(analysis.get("expected_seeds") == expected_training, "paired seed set changed")
    protocol = analysis.get("protocol", {})
    require(
        protocol.get("evaluation", {}).get("episode_seeds") == expected_evaluation,
        "statistics are not tied to dedicated evaluation seeds",
    )
    require(protocol.get("statistics", {}).get("test") == "paired_t", "paired test changed")
    require((root / "tables" / "statistics.csv").is_file(), "baseline statistics table is absent")
    return {
        "runs": len(manifests),
        "evaluation_episodes": len(manifests) * len(expected_evaluation),
        "analysis_phase": "dedicated_frozen_policy_evaluation",
        "interpretation": "provisional" if analysis.get("not_converged") else "confirmatory",
        "non_converged_runs": len(analysis.get("not_converged", [])),
    }


def _verify_simulation_and_weight_boundaries(repo_root: Path) -> dict[str, Any]:
    simulator_files = [
        repo_root / "src" / "rlredteam" / "enterprise" / name
        for name in (
            "onprem.py",
            "environment.py",
            "state.py",
            "generalisation.py",
            "trajectory.py",
        )
    ]
    forbidden = ("import socket", "import requests", "import urllib", "from urllib")
    for path in simulator_files:
        source = path.read_text()
        require(
            not any(token in source for token in forbidden),
            f"network API imported by simulator: {path}",
        )
    ignore = (repo_root / ".gitignore").read_text().splitlines()
    require("runs/" in ignore and "results/" in ignore, "generated weights/results are not ignored")
    tracked = _git(repo_root, "ls-files", "runs/onprem-generalisation/model.zip")
    require(
        tracked.returncode == 0 and not tracked.stdout.strip(),
        "canonical weights are tracked by Git",
    )
    return {
        "runtime": "simulation-only",
        "checked_modules": len(simulator_files),
        "checkpoint_release": "gated and untracked",
    }


def _verify_postgres(metadata_by_split: dict[str, dict[str, Any]]) -> dict[str, Any]:
    import psycopg

    from rlredteam.storage.postgres_logger import connection_string

    reconstructed: dict[str, Any] = {}
    try:
        connection = psycopg.connect(connection_string())
    except Exception as exc:
        raise CompletionVerificationError(f"cannot connect to PostgreSQL: {exc}") from exc
    with connection:
        for split_name, metadata in metadata_by_split.items():
            experiment_id = int(metadata["database_experiment_id"])
            run_id = int(metadata["database_evaluation_run_id"])
            row = connection.execute(
                """
                SELECT r.status, count(DISTINCT ep.id), count(s.id),
                       bool_and(ep.goal_reached), coalesce(sum(ep.invalid_mask_selections), 0)
                FROM runs r
                LEFT JOIN episodes ep ON ep.run_id = r.id
                LEFT JOIN steps s ON s.episode_id = ep.id
                WHERE r.id = %s AND r.experiment_id = %s
                GROUP BY r.status
                """,
                (run_id, experiment_id),
            ).fetchone()
            require(row is not None, f"PostgreSQL has no {split_name} evaluation run")
            status, episodes, steps, goals, invalid = row
            require(status == "complete", f"PostgreSQL {split_name} run is not complete")
            require(
                episodes == int(metadata["database_episode_count"]),
                f"PostgreSQL {split_name} episode mismatch",
            )
            require(
                steps == int(metadata["database_step_count"]),
                f"PostgreSQL {split_name} step mismatch",
            )
            require(goals is True, f"PostgreSQL {split_name} includes failed goals")
            require(invalid == 0, f"PostgreSQL {split_name} includes invalid actions")
            reconstructed[split_name] = {
                "experiment_id": experiment_id,
                "run_id": run_id,
                "episodes": episodes,
                "steps": steps,
            }
    return reconstructed


def verify_onprem_completion(repo_root: Path, *, include_postgres: bool = False) -> dict[str, Any]:
    """Verify canonical research evidence and return a machine-readable verdict."""
    repo_root = Path(repo_root).resolve()
    manifest, training = _verify_training(repo_root)
    metadata_by_split: dict[str, dict[str, Any]] = {}
    evaluations: dict[str, dict[str, int]] = {}
    for split_name in ("validation", "test"):
        metadata, evidence = _verify_evaluation(repo_root, split_name, manifest)
        metadata_by_split[split_name] = metadata
        evaluations[split_name] = evidence
    report: dict[str, Any] = {
        "schema_version": 1,
        "complete": True,
        "scope": "hidden seeded on-prem topology generalisation",
        "checks": {
            "training_provenance": "pass",
            "hidden_topology_distribution": "pass",
            "agent_knowledge_action_masks": "pass",
            "frozen_unseen_seed_evaluation": "pass",
            "auditable_attack_paths": "pass",
            "fixed_baseline_preserved": "pass",
            "simulation_boundary": "pass",
            "weight_release_gate": "pass",
            "postgres_reconstruction": "pass" if include_postgres else "not-requested",
        },
        "training": training,
        "evaluations": evaluations,
        "fixed_baseline": _verify_fixed_baseline(repo_root),
        "boundaries": _verify_simulation_and_weight_boundaries(repo_root),
    }
    if include_postgres:
        report["postgres"] = _verify_postgres(metadata_by_split)
    return report
