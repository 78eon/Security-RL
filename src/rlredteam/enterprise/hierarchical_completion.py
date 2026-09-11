"""Read-only completion gate for the canonical Phase 12 hierarchical study."""

from __future__ import annotations

import csv
import json
import math
import subprocess
from pathlib import Path
from typing import Any

from rlredteam.enterprise.curriculum import canonical_digest
from rlredteam.enterprise.curriculum_study import (
    aggregate_seed_metrics,
    analyse_seed_metrics,
)
from rlredteam.enterprise.generalisation import evaluation_provenance, sha256_file
from rlredteam.enterprise.graph_policy_study import observation_schema
from rlredteam.enterprise.hierarchical_policy import (
    OBJECTIVES,
    HierarchicalResearchConfig,
    action_to_objective,
    fixed_action_catalogue,
)
from rlredteam.enterprise.hierarchical_study import (
    current_input_manifest,
    validate_paired_training_isolation,
)


class HierarchicalCompletionError(RuntimeError):
    """Raised when Phase 12 evidence fails a completion invariant."""


def require(condition: bool, message: str) -> None:
    """Fail closed when a required proof is absent."""
    if not condition:
        raise HierarchicalCompletionError(message)


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


def _json(path: Path) -> Any:
    require(path.is_file(), f"missing evidence file: {path}")
    try:
        return json.loads(path.read_text(), parse_constant=_reject_constant)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise HierarchicalCompletionError(f"invalid JSON evidence: {path}: {exc}") from exc


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
        raise HierarchicalCompletionError(f"cannot verify Git provenance: {exc}") from exc


def _finite_number(value: Any, message: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise HierarchicalCompletionError(message) from exc
    require(math.isfinite(result), message)
    return result


def _close(actual: Any, expected: Any, message: str) -> None:
    require(
        math.isclose(float(actual), float(expected), rel_tol=1e-12, abs_tol=1e-12),
        message,
    )


def _verify_protocol(
    repo_root: Path,
) -> tuple[HierarchicalResearchConfig, dict[str, Any], dict[str, Any]]:
    config = HierarchicalResearchConfig.from_yaml(
        repo_root / "configs/experiments/hierarchical_policy.yaml"
    )
    frozen = _json(repo_root / "configs/frozen_hierarchical_policy.json")
    study = _json(repo_root / "results" / config.experiment_id / "test/metadata/study.json")
    require(isinstance(frozen, dict), "frozen inputs must be a JSON object")
    require(isinstance(study, dict), "study metadata must be a JSON object")

    current = current_input_manifest(config)
    require(
        current == {key: frozen.get(key) for key in current},
        "current Phase 12 inputs differ from the frozen protocol",
    )
    hierarchy = frozen.get("hierarchy", {})
    catalogue = fixed_action_catalogue()
    require(len(catalogue) == 968, "fixed public action catalogue size drifted")
    require(
        hierarchy.get("action_to_objective") == list(action_to_objective(config)),
        "frozen objective mapping differs from the public action catalogue",
    )
    require(
        hierarchy.get("action_catalogue_sha256") == canonical_digest(catalogue),
        "frozen public action catalogue hash drifted",
    )
    require(
        hierarchy.get("source") == "static_public_slot_catalogue",
        "hierarchy was not derived from the static public action catalogue",
    )
    require(
        hierarchy.get("transition_semantics")
        == "one_concrete_action_per_environment_step",
        "hierarchical transition semantics drifted",
    )

    expected = {
        "study_id": config.experiment_id,
        "phase": "canonical_test",
        "complete": True,
        "study_config_hash": config.digest(),
        "scientific_config_hash": config.scientific_digest(),
        "topology_split": "test",
        "topology_seeds": list(config.topology_splits["test"]),
        "evaluation_episode_seeds": list(config.evaluation_episode_seeds),
        "training_timesteps": config.total_timesteps,
        "parallel_training_workers": config.parallel_training_workers,
        "runtime_cap_minutes": config.runtime_cap_minutes,
        "frozen_inputs": frozen,
    }
    for field, value in expected.items():
        require(study.get(field) == value, f"study metadata mismatch: {field}")
    projected = _finite_number(
        study.get("projected_canonical_minutes"), "canonical runtime projection is absent"
    )
    wall_seconds = _finite_number(
        study.get("training_wall_seconds"), "canonical training wall time is absent"
    )
    require(0 < projected <= config.runtime_cap_minutes, "runtime projection exceeds frozen cap")
    require(
        0 < wall_seconds <= config.runtime_cap_minutes * 60,
        "canonical training wall time exceeds frozen cap",
    )

    for commit, label in (
        (str(frozen.get("protocol_commit", "")), "protocol"),
        (str(study.get("code_commit", "")), "execution"),
    ):
        require(bool(commit), f"canonical {label} commit is absent")
        require(
            _git(repo_root, "cat-file", "-e", f"{commit}^{{commit}}").returncode == 0,
            f"canonical {label} commit is absent from Git history",
        )
        require(
            _git(repo_root, "merge-base", "--is-ancestor", commit, "HEAD").returncode == 0,
            f"canonical {label} commit is not an ancestor of HEAD",
        )
    require(
        _git(
            repo_root,
            "merge-base",
            "--is-ancestor",
            str(frozen["protocol_commit"]),
            str(study["code_commit"]),
        ).returncode
        == 0,
        "execution commit predates the frozen protocol",
    )

    schema = observation_schema()
    require(schema.get("source") == "AgentKnowledge", "observation source is unsafe")
    require(schema.get("hidden_topology_fields") == 0, "observation contains hidden topology")
    require(current.get("observation_schema") == schema, "frozen observation schema drifted")
    return config, frozen, study


def _expected_runs(config: HierarchicalResearchConfig) -> set[tuple[str, int]]:
    return {(arm, seed) for arm in config.arms for seed in config.training_seeds}


def _verify_training_audit(
    config: HierarchicalResearchConfig, manifest: dict[str, Any], *, run_name: str
) -> None:
    seed = int(manifest["training_seed"])
    reset_count = int(manifest.get("training_reset_count", 0))
    require(
        manifest.get("environment_seed") == seed * 100 + 1,
        f"environment seed drift: {run_name}",
    )
    require(reset_count > 0, f"training contains no reset evidence: {run_name}")
    require(bool(manifest.get("reset_trace_sha256")), f"reset trace absent: {run_name}")
    exposure = manifest.get("exposure_counts", {})
    possible = {
        f"{profile.value}:{topology_seed}"
        for profile in config.train_profiles
        for topology_seed in config.topology_splits["train"]
    }
    require(bool(exposure), f"training exposure evidence absent: {run_name}")
    require(set(exposure) <= possible, f"undeclared training exposure: {run_name}")
    require(sum(map(int, exposure.values())) == reset_count, f"reset count drift: {run_name}")
    profile_counts = manifest.get("observed_profile_counts", {})
    require(
        set(profile_counts) == {item.value for item in config.train_profiles},
        f"profile coverage drift: {run_name}",
    )
    require(
        all(int(value) > 0 for value in profile_counts.values()),
        f"training omitted a deployment profile: {run_name}",
    )
    require(
        sum(map(int, profile_counts.values())) == reset_count,
        f"profile count drift: {run_name}",
    )


def _read_trajectories(path: Path, *, run_name: str) -> list[dict[str, Any]]:
    require(path.is_file(), f"missing trajectories: {run_name}")
    rows = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        try:
            value = json.loads(line, parse_constant=_reject_constant)
        except (ValueError, json.JSONDecodeError) as exc:
            raise HierarchicalCompletionError(
                f"invalid trajectory {run_name}:{line_number}: {exc}"
            ) from exc
        require(isinstance(value, dict), f"trajectory is not an object: {run_name}")
        rows.append(value)
    return rows


def _verify_run(
    repo_root: Path,
    config: HierarchicalResearchConfig,
    frozen: dict[str, Any],
    study: dict[str, Any],
    manifest: dict[str, Any],
    metadata: dict[str, Any],
) -> tuple[dict[str, int], list[dict[str, Any]], list[dict[str, Any]]]:
    arm = str(manifest.get("arm"))
    seed = int(manifest.get("training_seed", -1))
    run_name = f"{config.experiment_id}-{arm}-s{seed}"
    checkpoint = repo_root / "runs" / config.experiment_id / "canonical" / run_name / "model.zip"
    output = repo_root / "results" / config.experiment_id / "test" / "raw" / run_name
    representation = (
        "agent_knowledge_graph_flat_action"
        if arm == "flat_graph"
        else "agent_knowledge_graph_objective_conditional_action"
    )
    expected_manifest = {
        "study_id": config.experiment_id,
        "experiment_id": run_name,
        "algorithm": "MaskablePPO",
        "policy": "AgentKnowledgeGraphPolicy",
        "representation": representation,
        "action_mask_source": "AgentKnowledge",
        "observation_source": "AgentKnowledge",
        "observation_schema": observation_schema(),
        "transition_semantics": "one_concrete_action_per_environment_step",
        "development": False,
        "training_seed": seed,
        "training_timesteps": config.total_timesteps,
        "actual_training_timesteps": config.total_timesteps,
        "train_profiles": [item.value for item in config.train_profiles],
        "train_topology_seeds": list(config.topology_splits["train"]),
        "validation_topology_seeds": list(config.topology_splits["validation"]),
        "test_topology_seeds": list(config.topology_splits["test"]),
        "study_config_hash": config.digest(),
        "scientific_config_hash": config.scientific_digest(),
        "base_profile_config_hash": frozen["base_profile_config_sha256"],
        "dependency_lock_hash": frozen["dependency_lock_hash"],
        "ppo": config.common_ppo,
        "graph_policy_config": config.graph_policy,
        "hierarchy_config": config.hierarchy if arm == "hierarchical_graph" else None,
        "frozen_inputs_sha256": canonical_digest(frozen),
        "git_commit": study["code_commit"],
        "git_dirty": False,
        "torch_num_threads": 1,
        "base_vulnerability_snapshot_sha256": frozen["vulnerability_snapshot_sha256"],
        "weights_release": "gated; runs/ is gitignored",
    }
    for field, value in expected_manifest.items():
        require(manifest.get(field) == value, f"manifest {field} drift: {run_name}")
    require(
        canonical_digest(manifest.get("base_distribution")) == frozen["distribution_sha256"],
        f"training distribution drift: {run_name}",
    )
    require(int(manifest.get("parameter_count", 0)) > 0, f"parameter count absent: {run_name}")
    elapsed = _finite_number(
        manifest.get("training_elapsed_seconds"), f"training duration absent: {run_name}"
    )
    require(0 < elapsed <= config.runtime_cap_minutes * 60, f"training duration drift: {run_name}")
    require(checkpoint.is_file(), f"missing checkpoint: {run_name}")
    require(
        manifest.get("checkpoint_sha256") == sha256_file(checkpoint),
        f"checkpoint drift: {run_name}",
    )
    require(len(str(manifest.get("policy_sha256", ""))) == 64, f"policy hash absent: {run_name}")
    _verify_training_audit(config, manifest, run_name=run_name)

    require(metadata.get("training_manifest") == manifest, f"embedded manifest drift: {run_name}")
    expected_episodes = (
        len(config.train_profiles)
        * len(config.topology_splits["test"])
        * len(config.evaluation_episode_seeds)
    )
    for field, value in {
        "phase": "phase12_frozen_hierarchical_policy_comparison",
        "split": "test",
        "gradient_updates": False,
        "evaluation_reset_count": expected_episodes,
        "invalid_mask_selections": 0,
        "checkpoint_sha256": manifest["checkpoint_sha256"],
    }.items():
        require(metadata.get(field) == value, f"evaluation {field} drift: {run_name}")
    environment_steps = int(metadata.get("environment_steps", -1))
    require(environment_steps > 0, f"evaluation contains no transitions: {run_name}")
    require(
        int(metadata.get("knowledge_mask_checks", -1)) == environment_steps,
        f"AgentKnowledge mask audit drift: {run_name}",
    )
    expected_options = environment_steps if arm == "hierarchical_graph" else 0
    require(
        int(metadata.get("option_decisions", -1)) == expected_options,
        f"hierarchical objective-decision count drift: {run_name}",
    )
    require(
        metadata.get("policy_sha256_before")
        == metadata.get("policy_sha256_after")
        == manifest.get("policy_sha256"),
        f"policy changed during evaluation: {run_name}",
    )

    rows = _csv(output / "episodes.csv")
    expected_keys = {
        (profile.value, topology_seed, evaluation_seed)
        for profile in config.train_profiles
        for topology_seed in config.topology_splits["test"]
        for evaluation_seed in config.evaluation_episode_seeds
    }
    observed_keys = {
        (row["profile"], int(row["topology_seed"]), int(row["evaluation_seed"]))
        for row in rows
    }
    require(
        len(rows) == expected_episodes and observed_keys == expected_keys,
        f"held-out evaluation grid mismatch: {run_name}",
    )
    distribution = manifest["base_distribution"]["test"]
    episode_success: dict[tuple[str, int, int], bool] = {}
    episode_lengths: dict[tuple[str, int, int], int] = {}
    for row in rows:
        key = (row["profile"], int(row["topology_seed"]), int(row["evaluation_seed"]))
        require(row["arm"] == arm, f"episode arm drift: {run_name}")
        require(int(row["training_seed"]) == seed, f"episode seed drift: {run_name}")
        require(int(row["invalid_mask_selections"]) == 0, f"invalid action row: {run_name}")
        require(int(row["failed_actions"]) == 0, f"failed action row: {run_name}")
        require(
            row["topology_hash"] == distribution[key[0]][str(key[1])],
            f"episode topology drift: {run_name}",
        )
        require(int(row["true_nodes"]) > 0, f"true-node count absent: {run_name}")
        require(
            0 <= int(row["known_nodes"]) <= int(row["true_nodes"]),
            f"knowledge count invalid: {run_name}",
        )
        succeeded = row["goal_reached"] == "True"
        require(
            (row["terminal_reason"] == "goal") == succeeded,
            f"terminal outcome drift: {run_name}",
        )
        episode_success[key] = succeeded
        episode_lengths[key] = int(row["episode_length"])

    trajectory_events: dict[tuple[str, int, int], set[tuple[Any, ...]]] = {}
    trajectories = _read_trajectories(output / "trajectories.jsonl", run_name=run_name)
    for step in trajectories:
        key = (step["profile"], int(step["topology_seed"]), int(step["evaluation_seed"]))
        require(key in expected_keys, f"trajectory grid drift: {run_name}")
        require(step["arm"] == arm, f"trajectory arm drift: {run_name}")
        require(int(step["training_seed"]) == seed, f"trajectory seed drift: {run_name}")
        require(
            step["topology_hash"] == distribution[key[0]][str(key[1])],
            f"trajectory topology drift: {run_name}",
        )
        require(step.get("success") is True, f"failed trajectory action: {run_name}")
        trajectory_events.setdefault(key, set()).add(
            (
                int(step["step"]),
                step["action"],
                step["success"],
                step["state_changed"],
                step["goal_reached"],
                step["topology_hash"],
            )
        )
    require(set(trajectory_events) == expected_keys, f"trajectory coverage mismatch: {run_name}")
    require(
        len(trajectories) == sum(episode_lengths.values()) == environment_steps,
        f"trajectory length drift: {run_name}",
    )
    require(
        len(trajectories) == int(metadata.get("database_step_count", -1)),
        f"database step metadata drift: {run_name}",
    )

    attack_paths = _json(output / "attack_paths.json")
    require(
        isinstance(attack_paths, list) and len(attack_paths) == expected_episodes,
        f"attack-path count drift: {run_name}",
    )
    path_keys = set()
    for path in attack_paths:
        key = (path["profile"], int(path["topology_seed"]), int(path["evaluation_seed"]))
        require(key in expected_keys, f"attack-path grid drift: {run_name}")
        path_keys.add(key)
        steps = path.get("steps", [])
        if episode_success[key]:
            require(bool(steps), f"successful episode has empty attack path: {run_name}")
            require(steps[-1].get("goal_reached") is True, f"attack path lacks goal: {run_name}")
            require(
                steps[-1].get("action_kind") == "access_asset",
                f"attack path lacks asset access: {run_name}",
            )
        require(
            all(step.get("success") is True for step in steps),
            f"attack path contains failure: {run_name}",
        )
        require(
            all(step.get("state_changed") is True for step in steps),
            f"attack path contains non-causal event: {run_name}",
        )
        for step in steps:
            fingerprint = (
                int(step["step"]),
                step["action"],
                step["success"],
                step["state_changed"],
                step["goal_reached"],
                step["topology_hash"],
            )
            require(
                fingerprint in trajectory_events[key],
                f"unrecorded attack-path event: {run_name}",
            )
    require(path_keys == expected_keys, f"attack-path coverage mismatch: {run_name}")

    summary = _json(output / "summary.json")
    require(summary.get("episode_count") == expected_episodes, f"summary episode drift: {run_name}")
    require(
        summary.get("topology_count") == len(config.topology_splits["test"]),
        f"summary topology drift: {run_name}",
    )
    require(summary.get("invalid_mask_selections") == 0, f"summary invalid actions: {run_name}")
    _close(
        summary.get("success_rate"),
        sum(episode_success.values()) / expected_episodes,
        f"summary success rate drift: {run_name}",
    )
    require(
        metadata.get("database_episode_count") == expected_episodes,
        f"database episode metadata drift: {run_name}",
    )
    return (
        {
            "episodes": expected_episodes,
            "steps": len(trajectories),
            "attack_paths": expected_episodes,
            "successful_episodes": sum(episode_success.values()),
        },
        rows,
        trajectories,
    )


def _coerce_seed_metric(row: dict[str, str]) -> dict[str, Any]:
    return {
        "arm": row["arm"],
        "training_seed": int(row["training_seed"]),
        "episode_count": int(row["episode_count"]),
        "success_rate": float(row["success_rate"]),
        "penalized_steps": float(row["penalized_steps"]),
        "total_reward": float(row["total_reward"]),
        "discovery_coverage": float(row["discovery_coverage"]),
        "failed_actions": float(row["failed_actions"]),
    }


def _verify_analysis(
    repo_root: Path,
    config: HierarchicalResearchConfig,
    episode_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    root = repo_root / "results" / config.experiment_id / "test"
    analysis = _json(root / "summaries/analysis.json")
    require(analysis.get("complete") is True, "paired analysis is incomplete")
    require(analysis.get("primary_unit") == "training_seed", "paired unit drifted")
    require(analysis.get("arm_a") == config.arms[0], "analysis control arm drifted")
    require(analysis.get("arm_b") == config.arms[1], "analysis hierarchy arm drifted")
    require(
        analysis.get("expected_training_seeds") == list(config.training_seeds),
        "paired seeds drifted",
    )
    require(
        analysis.get("observed_training_seeds")
        == {arm: list(config.training_seeds) for arm in config.arms},
        "paired arm/seed grid is incomplete",
    )
    require(
        analysis.get("primary_metrics") == list(config.primary_metrics),
        "primary metrics drifted",
    )

    recomputed_rows = aggregate_seed_metrics(
        episode_rows,
        config,
        expected_topology_seeds=config.topology_splits["test"],
    )
    stored_rows = [
        _coerce_seed_metric(row) for row in _csv(root / "summaries/seed_metrics.csv")
    ]
    require(len(stored_rows) == 20, "seed-level metric grid is incomplete")
    require(stored_rows == recomputed_rows, "seed metrics do not reconstruct from raw episodes")
    recomputed_analysis = analyse_seed_metrics(
        recomputed_rows,
        config,
        expected_seeds=config.training_seeds,
    )
    require(analysis == recomputed_analysis, "statistics do not reconstruct from seed metrics")

    comparisons = analysis.get("comparisons", [])
    expected_metrics = set(config.primary_metrics + config.descriptive_metrics)
    require(
        {item.get("metric") for item in comparisons} == expected_metrics,
        "analysis metrics drifted",
    )
    for item in comparisons:
        metric = str(item.get("metric"))
        require(item.get("n_pairs") == len(config.training_seeds), "paired sample size drifted")
        for field in ("difference", "mean_a", "mean_b", "p_value"):
            _finite_number(item.get(field), f"non-finite statistic: {metric}.{field}")
        if metric in config.primary_metrics:
            _finite_number(
                item.get("p_bonferroni"), f"missing corrected p-value: {metric}"
            )

    statistics_rows = _csv(root / "tables/statistics.csv")
    require(len(statistics_rows) == len(comparisons), "statistics table row count drifted")
    require(
        [row["metric"] for row in statistics_rows]
        == [item["metric"] for item in comparisons],
        "statistics table metric order drifted",
    )
    return {
        "pairs": len(config.training_seeds),
        "comparisons": len(comparisons),
        "significant_primary_results": sum(
            bool(item["significant"])
            for item in comparisons
            if item["metric"] in config.primary_metrics
        ),
        "primary_results": {
            item["metric"]: {
                "flat_mean": item["mean_a"],
                "hierarchical_mean": item["mean_b"],
                "hierarchical_minus_flat": item["difference"],
                "p_bonferroni": item["p_bonferroni"],
                "significant": item["significant"],
            }
            for item in comparisons
            if item["metric"] in config.primary_metrics
        },
    }


def _bool_csv(value: str) -> bool:
    require(value in {"True", "False"}, f"invalid CSV boolean: {value!r}")
    return value == "True"


def _verify_postgres_run(
    connection,
    metadata: dict[str, Any],
    config: HierarchicalResearchConfig,
    file_episodes: list[dict[str, str]],
    file_steps: list[dict[str, Any]],
) -> tuple[int, int, int]:
    experiment_id = int(metadata["database_experiment_id"])
    run_id = int(metadata["database_evaluation_run_id"])
    manifest = metadata["training_manifest"]
    provenance = evaluation_provenance(manifest, "test")
    row = connection.execute(
        """
        SELECT r.status, r.designation, r.evaluation_seeds, r.seed,
               e.name, e.algorithm, e.git_sha, e.config_hash,
               e.topology_config_hash, e.topology_hash,
               e.cve_manifest_sha256, e.seed_set, e.condition,
               e.reward_mode, e.hyperparameters, r.checkpoint_path
        FROM runs r JOIN experiments e ON e.id=r.experiment_id
        WHERE r.id=%s AND e.id=%s
        """,
        (run_id, experiment_id),
    ).fetchone()
    require(row is not None, f"PostgreSQL run is absent: {run_id}")
    expected_prefix = (
        "complete",
        "evaluation",
        list(config.evaluation_episode_seeds),
        int(manifest["training_seed"]),
        f"{manifest['experiment_id']}-test",
        manifest["algorithm"],
        manifest["git_commit"],
        provenance["config_hash"],
        provenance["topology_config_hash"],
        provenance["topology_hash"],
        provenance["cve_manifest_sha256"],
        [int(manifest["training_seed"])],
        "unseen_topology_generalisation",
        "enterprise_shaped",
    )
    require(tuple(row[:14]) == expected_prefix, f"PostgreSQL provenance drift: {run_id}")
    require(row[14] == manifest["ppo"], f"PostgreSQL hyperparameter drift: {run_id}")
    require(
        row[15] == metadata["checkpoint"],
        f"PostgreSQL checkpoint path drift: {run_id}",
    )

    db_episodes = connection.execute(
        """
        SELECT id, seed, topology_seed, episode_idx, total_reward, native_reward,
               length, terminal_state, goal_reached, exploited_hosts,
               hosts_compromised, topology_hash, known_nodes, true_nodes,
               discovery_coverage, invalid_mask_selections, failed_actions,
               deployment_profile
        FROM episodes WHERE run_id=%s ORDER BY episode_idx
        """,
        (run_id,),
    ).fetchall()
    require(len(db_episodes) == len(file_episodes) == 60, f"PostgreSQL episode drift: {run_id}")
    grouped_steps: dict[tuple[str, int, int], list[dict[str, Any]]] = {}
    for step in file_steps:
        key = (step["profile"], int(step["topology_seed"]), int(step["evaluation_seed"]))
        grouped_steps.setdefault(key, []).append(step)

    successes = step_count = 0
    for index, (db_episode, episode) in enumerate(
        zip(db_episodes, file_episodes, strict=True)
    ):
        episode_id = int(db_episode[0])
        expected_episode = (
            int(episode["evaluation_seed"]),
            int(episode["topology_seed"]),
            index,
            float(episode["total_reward"]),
            float(episode["total_reward"]),
            int(episode["episode_length"]),
            episode["terminal_reason"],
            _bool_csv(episode["goal_reached"]),
            int(episode["hosts_compromised"]),
            episode["topology_hash"],
            int(episode["known_nodes"]),
            int(episode["true_nodes"]),
            float(episode["discovery_coverage"]),
            0,
            0,
            episode["profile"],
        )
        actual_episode = (
            int(db_episode[1]),
            int(db_episode[2]),
            int(db_episode[3]),
            float(db_episode[4]),
            float(db_episode[5]),
            int(db_episode[6]),
            db_episode[7],
            bool(db_episode[8]),
            int(db_episode[10]),
            db_episode[11],
            int(db_episode[12]),
            int(db_episode[13]),
            float(db_episode[14]),
            int(db_episode[15]),
            int(db_episode[16]),
            db_episode[17],
        )
        require(
            actual_episode == expected_episode,
            f"PostgreSQL episode row drift: {run_id}/{index}",
        )
        expected_exploits = [
            step["target_entity"]
            for step in grouped_steps[
                (
                    episode["profile"],
                    int(episode["topology_seed"]),
                    int(episode["evaluation_seed"]),
                )
            ]
            if step["action_kind"] == "exploit" and step["success"]
        ]
        require(db_episode[9] == expected_exploits, f"PostgreSQL exploit list drift: {run_id}")

        expected_steps = grouped_steps[
            (
                episode["profile"],
                int(episode["topology_seed"]),
                int(episode["evaluation_seed"]),
            )
        ]
        db_steps = connection.execute(
            """
            SELECT step_idx, action_name, action_kind, success, reward, native_reward,
                   cve_id, cvss_base, target_entity, state_changed,
                   prerequisites, outcomes, error
            FROM steps WHERE episode_id=%s ORDER BY step_idx
            """,
            (episode_id,),
        ).fetchall()
        require(len(db_steps) == len(expected_steps), f"PostgreSQL step count drift: {run_id}")
        for db_step, step in zip(db_steps, expected_steps, strict=True):
            actual_step = (
                int(db_step[0]),
                db_step[1],
                db_step[2],
                bool(db_step[3]),
                float(db_step[4]),
                float(db_step[5]),
                db_step[6],
                float(db_step[7]) if db_step[7] is not None else None,
                db_step[8],
                bool(db_step[9]),
                db_step[10],
                db_step[11],
                db_step[12],
            )
            expected_step = (
                int(step["step"]),
                step["action"],
                step["action_kind"],
                bool(step["success"]),
                float(step["reward"]),
                float(step["reward"]),
                step.get("cve_id"),
                float(step["cvss_base"]) if step.get("cvss_base") is not None else None,
                step["target_entity"],
                bool(step["state_changed"]),
                step["prerequisites"],
                step["outcomes"],
                step.get("reason"),
            )
            require(actual_step == expected_step, f"PostgreSQL step row drift: {run_id}")
        successes += int(_bool_csv(episode["goal_reached"]))
        step_count += len(db_steps)

    require(
        len(file_episodes) == metadata["database_episode_count"],
        f"PostgreSQL episode metadata mismatch: {run_id}",
    )
    require(
        step_count == metadata["database_step_count"],
        f"PostgreSQL step metadata mismatch: {run_id}",
    )
    return len(file_episodes), step_count, successes


def _verify_postgres(
    repo_root: Path,
    evaluations: list[dict[str, Any]],
    config: HierarchicalResearchConfig,
) -> dict[str, int]:
    import psycopg

    from rlredteam.storage.postgres_logger import connection_string

    try:
        connection = psycopg.connect(connection_string())
    except Exception as exc:
        raise HierarchicalCompletionError(f"cannot connect to PostgreSQL: {exc}") from exc
    episodes = steps = successes = 0
    experiment_ids: set[int] = set()
    run_ids: set[int] = set()
    with connection:
        for metadata in evaluations:
            manifest = metadata["training_manifest"]
            run_name = manifest["experiment_id"]
            output = repo_root / "results" / config.experiment_id / "test/raw" / run_name
            file_episodes = _csv(output / "episodes.csv")
            file_steps = _read_trajectories(output / "trajectories.jsonl", run_name=run_name)
            run_episodes, run_steps, run_successes = _verify_postgres_run(
                connection, metadata, config, file_episodes, file_steps
            )
            episodes += run_episodes
            steps += run_steps
            successes += run_successes
            experiment_ids.add(int(metadata["database_experiment_id"]))
            run_ids.add(int(metadata["database_evaluation_run_id"]))
        require(len(experiment_ids) == len(evaluations), "PostgreSQL experiment IDs are reused")
        require(len(run_ids) == len(evaluations), "PostgreSQL run IDs are reused")
        study_count = connection.execute(
            "SELECT count(*) FROM experiments WHERE name LIKE %s",
            (f"{config.experiment_id}-%",),
        ).fetchone()[0]
        require(study_count == len(evaluations), "unexpected Phase 12 PostgreSQL experiment count")
    return {
        "experiments": len(experiment_ids),
        "runs": len(run_ids),
        "episodes": episodes,
        "successful_episodes": successes,
        "steps": steps,
    }


def verify_hierarchical_completion(
    repo_root: Path, *, include_postgres: bool = False
) -> dict[str, Any]:
    """Verify every canonical Phase 12 artifact without mutating evidence."""
    repo_root = Path(repo_root).resolve()
    config, frozen, study = _verify_protocol(repo_root)
    manifests = study.get("training_manifests", [])
    evaluations = study.get("evaluation_metadata", [])
    expected = _expected_runs(config)
    require(
        len(manifests) == len(evaluations) == len(expected) == 20,
        "canonical run count mismatch",
    )
    manifests_by_key = {
        (item.get("arm"), int(item.get("training_seed", -1))): item for item in manifests
    }
    evaluations_by_key = {
        (
            item.get("training_manifest", {}).get("arm"),
            int(item.get("training_manifest", {}).get("training_seed", -1)),
        ): item
        for item in evaluations
    }
    require(set(manifests_by_key) == expected, "training manifest grid mismatch")
    require(set(evaluations_by_key) == expected, "evaluation metadata grid mismatch")
    for seed in config.training_seeds:
        validate_paired_training_isolation(
            manifests_by_key[(config.arms[0], seed)],
            manifests_by_key[(config.arms[1], seed)],
        )

    parameter_counts = {
        arm: {
            int(manifests_by_key[(arm, seed)]["parameter_count"])
            for seed in config.training_seeds
        }
        for arm in config.arms
    }
    require(
        all(len(values) == 1 for values in parameter_counts.values()),
        "parameter count drift within an arm",
    )
    stable_parameters = {arm: next(iter(values)) for arm, values in parameter_counts.items()}
    expected_delta = len(OBJECTIVES) * (int(config.graph_policy["feature_dim"]) + 1)
    require(
        stable_parameters["hierarchical_graph"] - stable_parameters["flat_graph"]
        == expected_delta,
        "hierarchical model differs by more than the preregistered objective head",
    )

    totals = {
        "runs": 0,
        "episodes": 0,
        "steps": 0,
        "attack_paths": 0,
        "successful_episodes": 0,
    }
    all_episode_rows: list[dict[str, Any]] = []
    for key in sorted(expected):
        counts, episode_rows, _ = _verify_run(
            repo_root,
            config,
            frozen,
            study,
            manifests_by_key[key],
            evaluations_by_key[key],
        )
        totals["runs"] += 1
        for field in ("episodes", "steps", "attack_paths", "successful_episodes"):
            totals[field] += counts[field]
        for row in episode_rows:
            all_episode_rows.append(
                {
                    **row,
                    "training_seed": int(row["training_seed"]),
                    "topology_seed": int(row["topology_seed"]),
                    "evaluation_seed": int(row["evaluation_seed"]),
                    "goal_reached": _bool_csv(row["goal_reached"]),
                    "steps_to_goal": (
                        int(row["steps_to_goal"]) if row["steps_to_goal"] else None
                    ),
                    "total_reward": float(row["total_reward"]),
                    "discovery_coverage": float(row["discovery_coverage"]),
                    "failed_actions": int(row["failed_actions"]),
                }
            )
    require(
        totals["runs"] * config.total_timesteps == 512_000,
        "aggregate training budget drift",
    )
    require(totals["episodes"] == totals["attack_paths"] == 1_200, "evidence grid drift")
    analysis = _verify_analysis(repo_root, config, all_episode_rows)
    postgres: dict[str, Any] = (
        _verify_postgres(repo_root, evaluations, config)
        if include_postgres
        else {"status": "not-requested"}
    )
    return {
        "complete": True,
        "study_id": config.experiment_id,
        "checks": {
            "frozen_prospective_protocol": "pass",
            "agent_knowledge_only_observation": "pass",
            "static_public_action_taxonomy": "pass",
            "matched_hierarchical_isolation": "pass",
            "exact_training_budgets": "pass",
            "deterministic_reset_audit": "pass",
            "agent_knowledge_action_masks": "pass",
            "frozen_evaluation": "pass",
            "policy_immutability": "pass",
            "hierarchical_objective_accounting": "pass",
            "causal_attack_paths": "pass",
            "paired_seed_statistics": "pass",
            "postgres_exact_reconstruction": "pass" if include_postgres else "not-requested",
        },
        "evidence": totals,
        "parameter_counts": stable_parameters,
        "objective_head_parameter_delta": expected_delta,
        "analysis": analysis,
        "postgres": postgres,
        "runtime": {
            "training_wall_minutes": study["training_wall_seconds"] / 60,
            "projected_minutes": study["projected_canonical_minutes"],
            "cap_minutes": config.runtime_cap_minutes,
        },
    }


__all__ = [
    "HierarchicalCompletionError",
    "require",
    "verify_hierarchical_completion",
]
