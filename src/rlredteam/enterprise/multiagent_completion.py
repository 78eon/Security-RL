"""Read-only, fail-closed completion gate for the Phase 13 red-blue study."""

from __future__ import annotations

import csv
import json
import math
import subprocess
from pathlib import Path
from typing import Any

from rlredteam.enterprise.curriculum import canonical_digest
from rlredteam.enterprise.generalisation import (
    evaluation_provenance,
    reconstruct_attack_path,
    sha256_file,
)
from rlredteam.enterprise.multiagent import MultiAgentResearchConfig, phase13_policy_manifest
from rlredteam.enterprise.multiagent_study import (
    aggregate_seed_metrics,
    analyse_seed_metrics,
    current_input_manifest,
    validate_paired_red_training_isolation,
)


class MultiAgentCompletionError(RuntimeError):
    """Raised when canonical Phase 13 evidence violates a frozen invariant."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise MultiAgentCompletionError(message)


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


def _json(path: Path) -> Any:
    require(path.is_file(), f"missing evidence file: {path}")
    try:
        return json.loads(path.read_text(), parse_constant=_reject_constant)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise MultiAgentCompletionError(f"invalid JSON evidence: {path}: {exc}") from exc


def _csv(path: Path) -> list[dict[str, str]]:
    require(path.is_file(), f"missing evidence file: {path}")
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _trajectories(path: Path) -> list[dict[str, Any]]:
    require(path.is_file(), f"missing trajectories: {path}")
    rows = []
    for number, line in enumerate(path.read_text().splitlines(), start=1):
        try:
            value = json.loads(line, parse_constant=_reject_constant)
        except (ValueError, json.JSONDecodeError) as exc:
            raise MultiAgentCompletionError(
                f"invalid trajectory {path}:{number}: {exc}"
            ) from exc
        require(isinstance(value, dict), f"trajectory row is not an object: {path}")
        rows.append(value)
    return rows


def _bool(value: str) -> bool:
    require(value in {"True", "False"}, f"invalid CSV boolean: {value!r}")
    return value == "True"


def _optional_int(value: str) -> int | None:
    return int(value) if value else None


def _finite(value: Any, message: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise MultiAgentCompletionError(message) from exc
    require(math.isfinite(result), message)
    return result


def _close(actual: Any, expected: Any, message: str) -> None:
    require(
        math.isclose(float(actual), float(expected), rel_tol=1e-12, abs_tol=1e-12),
        message,
    )


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            ["git", *args],
            cwd=repo,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise MultiAgentCompletionError(f"cannot verify Git provenance: {exc}") from exc


def _verify_protocol(repo: Path) -> tuple[MultiAgentResearchConfig, dict, dict]:
    config = MultiAgentResearchConfig.from_yaml(
        repo / "configs/experiments/multiagent_defense.yaml"
    )
    frozen = _json(repo / "configs/frozen_multiagent_defense.json")
    study = _json(repo / "results" / config.experiment_id / "test/metadata/study.json")
    defender_checkpoint = (
        repo / "runs" / config.experiment_id / "development"
        / f"{config.experiment_id}-defender/model.zip"
    )
    development = (
        repo / "results" / config.experiment_id / "development/metadata/study.json"
    )
    current = current_input_manifest(config, defender_checkpoint=defender_checkpoint)
    require(
        current == {key: frozen.get(key) for key in current},
        "current Phase 13 inputs differ from the frozen protocol",
    )
    require(
        frozen.get("development_study_sha256") == sha256_file(development),
        "excluded-seed development evidence drifted after freeze",
    )
    require(
        frozen.get("defender_checkpoint_sha256") == sha256_file(defender_checkpoint),
        "frozen defender checkpoint drifted",
    )
    require(
        frozen.get("policy_manifest") == phase13_policy_manifest(config),
        "partial-observation policy schema drifted",
    )
    expected = {
        "study_id": config.experiment_id,
        "phase": "canonical_test",
        "complete": True,
        "study_config_hash": config.digest(),
        "scientific_config_hash": config.scientific_digest(),
        "frozen_inputs": frozen,
        "topology_split": "test",
        "topology_seeds": list(config.topology_splits["test"]),
        "evaluation_episode_seeds": list(config.evaluation_episode_seeds),
        "red_training_timesteps": config.red_total_timesteps,
        "aggregate_red_training_timesteps": 20 * config.red_total_timesteps,
        "parallel_training_workers": config.parallel_training_workers,
        "runtime_cap_minutes": config.runtime_cap_minutes,
        "defender_checkpoint_sha256": sha256_file(defender_checkpoint),
    }
    for field, value in expected.items():
        require(study.get(field) == value, f"study metadata mismatch: {field}")
    wall = _finite(study.get("training_wall_seconds"), "canonical wall time absent")
    projection = _finite(
        study.get("projected_canonical_minutes"), "canonical projection absent"
    )
    require(0 < wall <= config.runtime_cap_minutes * 60, "canonical runtime cap exceeded")
    require(0 < projection <= config.runtime_cap_minutes, "runtime projection cap exceeded")
    for commit, label in (
        (str(frozen.get("protocol_commit", "")), "protocol"),
        (str(study.get("code_commit", "")), "execution"),
    ):
        require(bool(commit), f"{label} commit is absent")
        require(
            _git(repo, "cat-file", "-e", f"{commit}^{{commit}}").returncode == 0,
            f"{label} commit is absent from Git history",
        )
        require(
            _git(repo, "merge-base", "--is-ancestor", commit, "HEAD").returncode == 0,
            f"{label} commit is not an ancestor of HEAD",
        )
    require(
        _git(
            repo,
            "merge-base",
            "--is-ancestor",
            str(frozen["protocol_commit"]),
            str(study["code_commit"]),
        ).returncode
        == 0,
        "execution commit predates the frozen protocol",
    )
    return config, frozen, study


def _expected_runs(config: MultiAgentResearchConfig) -> set[tuple[str, int]]:
    return {(arm, seed) for arm in config.arms for seed in config.training_seeds}


def _verify_manifest(
    repo: Path,
    config: MultiAgentResearchConfig,
    frozen: dict,
    study: dict,
    manifest: dict,
) -> None:
    arm = str(manifest.get("arm"))
    seed = int(manifest.get("training_seed", -1))
    name = f"{config.experiment_id}-{arm}-s{seed}"
    checkpoint = repo / "runs" / config.experiment_id / "canonical" / name / "model.zip"
    expected = {
        "schema_version": 1,
        "study_id": config.experiment_id,
        "experiment_id": name,
        "arm": arm,
        "algorithm": "MaskablePPO",
        "policy": "HierarchicalMaskableActorCriticPolicy",
        "representation": "agent_knowledge_graph_objective_conditional_action",
        "red_observation_source": "AgentKnowledge",
        "red_action_mask_source": "AgentKnowledge",
        "defender_observation_source": "DefenderKnowledge",
        "defender_action_mask_source": "DefenderKnowledge",
        "policy_manifest": phase13_policy_manifest(config),
        "turn_semantics": "one_blue_decision_then_one_red_transition",
        "development": False,
        "training_seed": seed,
        "environment_seed": seed * 100 + 1,
        "red_training_timesteps": config.red_total_timesteps,
        "actual_red_training_timesteps": config.red_total_timesteps,
        "study_config_hash": config.digest(),
        "scientific_config_hash": config.scientific_digest(),
        "dependency_lock_hash": frozen["dependency_lock_hash"],
        "git_commit": study["code_commit"],
        "git_dirty": False,
        "red_ppo": config.red_ppo,
        "graph_policy_config": config.graph_policy,
        "hierarchy_config": config.hierarchy,
        "defense_config": config.defense,
        "torch_num_threads": 1,
        "frozen_inputs_sha256": canonical_digest(frozen),
        "checkpoint_sha256": sha256_file(checkpoint),
        "weights_release": "gated; runs/ is gitignored",
        "defender_decision_count": config.red_total_timesteps,
    }
    for field, value in expected.items():
        require(manifest.get(field) == value, f"training manifest drift: {name}.{field}")
    expected_defender = (
        frozen["defender_checkpoint_sha256"]
        if arm == "adaptive_defender_training"
        else None
    )
    require(
        manifest.get("defender_checkpoint_sha256") == expected_defender,
        f"defender assignment drift: {name}",
    )
    require(
        manifest.get("policy_sha256_before_training") != manifest.get("policy_sha256"),
        f"red policy did not update: {name}",
    )
    if arm == "adaptive_defender_training":
        require(
            manifest.get("defender_policy_sha256_before")
            == manifest.get("defender_policy_sha256_after"),
            f"frozen defender changed during red training: {name}",
        )
    require(int(manifest.get("red_parameter_count", 0)) > 0, f"parameters absent: {name}")
    require(int(manifest.get("training_reset_count", 0)) > 0, f"reset audit absent: {name}")
    require(bool(manifest.get("reset_trace_sha256")), f"reset hash absent: {name}")
    require(checkpoint.is_file(), f"checkpoint absent: {name}")


def _typed_episode(row: dict[str, str]) -> dict[str, Any]:
    return {
        **row,
        "training_seed": int(row["training_seed"]),
        "topology_seed": int(row["topology_seed"]),
        "evaluation_seed": int(row["evaluation_seed"]),
        "campaign_episode_index": int(row["campaign_episode_index"]),
        "goal_reached": _bool(row["goal_reached"]),
        "steps_to_goal": _optional_int(row["steps_to_goal"]),
        "episode_length": int(row["episode_length"]),
        "total_reward": float(row["total_reward"]),
        "native_total_reward": float(row["native_total_reward"]),
        "known_nodes": int(row["known_nodes"]),
        "true_nodes": int(row["true_nodes"]),
        "discovery_coverage": float(row["discovery_coverage"]),
        "invalid_mask_selections": int(row["invalid_mask_selections"]),
        "invalid_defender_selections": int(row["invalid_defender_selections"]),
        "failed_actions": int(row["failed_actions"]),
        "hosts_compromised": int(row["hosts_compromised"]),
        "path_events": int(row["path_events"]),
        "detected_actions": int(row["detected_actions"]),
        "detectable_actions": int(row["detectable_actions"]),
        "detection_rate": float(row["detection_rate"]),
        "evasion_rate": float(row["evasion_rate"]),
        "exploit_attempts": int(row["exploit_attempts"]),
        "mitigated_exploits": int(row["mitigated_exploits"]),
        "mitigation_block_rate": float(row["mitigation_block_rate"]),
        "defender_actions": int(row["defender_actions"]),
        "threshold_adjustments": int(row["threshold_adjustments"]),
        "patch_schedules": int(row["patch_schedules"]),
        "patch_activations": int(row["patch_activations"]),
    }


def _verify_run(
    repo: Path,
    config: MultiAgentResearchConfig,
    manifest: dict,
    metadata: dict,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    arm = manifest["arm"]
    seed = int(manifest["training_seed"])
    name = manifest["experiment_id"]
    root = repo / "results" / config.experiment_id / "test/raw" / name
    checkpoint = repo / "runs" / config.experiment_id / "canonical" / name / "model.zip"
    defender_policy = metadata.get("defender_policy_sha256_before")
    expected_episodes = (
        len(config.train_profiles)
        * len(config.topology_splits["test"])
        * len(config.evaluation_episode_seeds)
    )
    expected_metadata = {
        "phase": "phase13_frozen_adversarial_defender_comparison",
        "split": "test",
        "gradient_updates": False,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": manifest["checkpoint_sha256"],
        "training_manifest": manifest,
        "evaluation_reset_count": expected_episodes,
        "invalid_mask_selections": 0,
        "invalid_defender_selections": 0,
        "turn_semantics": "one_blue_decision_then_one_red_transition",
        "red_policy_sha256_before": manifest["policy_sha256"],
        "red_policy_sha256_after": manifest["policy_sha256"],
    }
    for field, value in expected_metadata.items():
        require(metadata.get(field) == value, f"evaluation metadata drift: {name}.{field}")
    require(
        defender_policy == metadata.get("defender_policy_sha256_after"),
        f"defender changed during evaluation: {name}",
    )
    environment_steps = int(metadata.get("environment_steps", 0))
    require(environment_steps > 0, f"evaluation has no transitions: {name}")
    for field in (
        "red_knowledge_mask_checks",
        "defender_knowledge_mask_checks",
        "defender_decisions",
    ):
        require(int(metadata.get(field, -1)) == environment_steps, f"turn audit drift: {name}")

    episode_rows = [_typed_episode(row) for row in _csv(root / "episodes.csv")]
    expected_keys = {
        (profile.value, topology, evaluation)
        for profile in config.train_profiles
        for topology in config.topology_splits["test"]
        for evaluation in config.evaluation_episode_seeds
    }
    observed_keys = {
        (row["profile"], row["topology_seed"], row["evaluation_seed"])
        for row in episode_rows
    }
    require(
        len(episode_rows) == expected_episodes and observed_keys == expected_keys,
        f"untouched test grid drift: {name}",
    )
    trajectories = _trajectories(root / "trajectories.jsonl")
    require(len(trajectories) == environment_steps, f"trajectory length drift: {name}")
    grouped: dict[tuple[str, int, int], list[dict[str, Any]]] = {}
    distribution = manifest["base_distribution"]["test"]
    for step in trajectories:
        key = (step["profile"], int(step["topology_seed"]), int(step["evaluation_seed"]))
        require(key in expected_keys, f"trajectory case drift: {name}")
        require(step["arm"] == arm, f"trajectory arm drift: {name}")
        require(int(step["training_seed"]) == seed, f"trajectory training seed drift: {name}")
        require(
            step["topology_hash"] == distribution[key[0]][str(key[1])],
            f"trajectory topology hash drift: {name}",
        )
        ids_level = int(step["ids_level"])
        probability = min(
            1.0,
            float(config.defense["base_detection_probability"][step["action_kind"]])
            * float(config.defense["ids_multipliers"][ids_level]),
        )
        _close(step["detection_probability"], probability, f"detection model drift: {name}")
        expected_reward = float(step["native_reward"]) + (
            float(config.defense["red_detection_penalty"]) if step["detected"] else 0.0
        )
        _close(step["reward"], expected_reward, f"red reward drift: {name}")
        require(
            step["defender_action_index"] >= 0 and step["ids_level_name"]
            == config.defense["ids_levels"][ids_level],
            f"defender telemetry drift: {name}",
        )
        grouped.setdefault(key, []).append(step)
    require(set(grouped) == expected_keys, f"trajectory coverage drift: {name}")

    for episode in episode_rows:
        key = (episode["profile"], episode["topology_seed"], episode["evaluation_seed"])
        values = grouped[key]
        require(
            episode["topology_hash"] == distribution[key[0]][str(key[1])],
            f"episode topology hash drift: {name}",
        )
        require(episode["episode_length"] == len(values), f"episode length drift: {name}")
        _close(
            episode["total_reward"],
            sum(float(step["reward"]) for step in values),
            f"episode red reward drift: {name}",
        )
        _close(
            episode["native_total_reward"],
            sum(float(step["native_reward"]) for step in values),
            f"episode native reward drift: {name}",
        )
        detected = sum(bool(step["detected"]) for step in values)
        attempts = sum(step["action_kind"] == "exploit" for step in values)
        blocks = sum(bool(step["mitigation_blocked"]) for step in values)
        require(episode["detected_actions"] == detected, f"detection count drift: {name}")
        require(episode["detectable_actions"] == len(values), f"detectable count drift: {name}")
        _close(episode["detection_rate"], detected / len(values), f"rate drift: {name}")
        _close(episode["evasion_rate"], 1 - detected / len(values), f"evasion drift: {name}")
        require(episode["exploit_attempts"] == attempts, f"exploit count drift: {name}")
        require(episode["mitigated_exploits"] == blocks, f"mitigation count drift: {name}")
        _close(
            episode["mitigation_block_rate"],
            blocks / max(1, attempts),
            f"mitigation rate drift: {name}",
        )
        require(
            episode["invalid_mask_selections"]
            == episode["invalid_defender_selections"]
            == 0,
            f"invalid policy action: {name}",
        )
    attack_paths = _json(root / "attack_paths.json")
    expected_paths = []
    for profile, topology, evaluation in sorted(expected_keys):
        expected_paths.append(
            {
                "profile": profile,
                "topology_seed": topology,
                "evaluation_seed": evaluation,
                "steps": reconstruct_attack_path(grouped[(profile, topology, evaluation)]),
            }
        )
    require(attack_paths == expected_paths, f"causal attack paths drift: {name}")
    return episode_rows, trajectories, len(attack_paths)


def _verify_analysis(
    repo: Path,
    config: MultiAgentResearchConfig,
    episodes: list[dict[str, Any]],
) -> dict[str, Any]:
    root = repo / "results" / config.experiment_id / "test"
    seed_metrics = aggregate_seed_metrics(
        episodes,
        config,
        expected_topology_seeds=config.topology_splits["test"],
    )
    report = analyse_seed_metrics(seed_metrics, config)
    require(report == _json(root / "summaries/analysis.json"), "analysis does not reconstruct")
    stored = _csv(root / "summaries/seed_metrics.csv")
    require(len(stored) == len(seed_metrics) == 20, "seed metric row count drifted")
    for raw, expected in zip(stored, seed_metrics, strict=True):
        for field, value in expected.items():
            if isinstance(value, str):
                require(raw[field] == value, f"seed metric drift: {field}")
            elif isinstance(value, int):
                require(int(raw[field]) == value, f"seed metric drift: {field}")
            else:
                _close(raw[field], value, f"seed metric drift: {field}")
    comparisons = report["comparisons"]
    require(report["complete"], "paired statistical grid is incomplete")
    require(
        all(item["n_pairs"] == 10 for item in comparisons),
        "statistics do not use ten matched red training seeds",
    )
    for item in comparisons:
        if item["metric"] in config.primary_metrics:
            _finite(item["p_bonferroni"], f"corrected p-value absent: {item['metric']}")
    return report


def _verify_postgres_run(connection, metadata: dict, episodes: list, steps: list, config) -> None:
    experiment_id = int(metadata["database_experiment_id"])
    run_id = int(metadata["database_evaluation_run_id"])
    manifest = metadata["training_manifest"]
    split_name = str(metadata["split"])
    provenance = evaluation_provenance(manifest, split_name)
    row = connection.execute(
        """
        SELECT r.status,r.designation,r.evaluation_seeds,r.seed,e.name,e.algorithm,
               e.git_sha,e.config_hash,e.topology_config_hash,e.topology_hash,
               e.cve_manifest_sha256,e.seed_set,e.condition,e.reward_mode,
               e.hyperparameters,r.checkpoint_path
        FROM runs r JOIN experiments e ON e.id=r.experiment_id
        WHERE r.id=%s AND e.id=%s
        """,
        (run_id, experiment_id),
    ).fetchone()
    expected = (
        "complete",
        "evaluation",
        list(config.evaluation_episode_seeds),
        int(manifest["training_seed"]),
        f"{manifest['experiment_id']}-{split_name}",
        "MaskablePPO",
        manifest["git_commit"],
        provenance["config_hash"],
        provenance["topology_config_hash"],
        provenance["topology_hash"],
        provenance["cve_manifest_sha256"],
        [int(manifest["training_seed"])],
        "unseen_topology_generalisation",
        "enterprise_shaped",
        config.red_ppo,
        metadata["checkpoint"],
    )
    require(row is not None and tuple(row) == expected, f"PostgreSQL provenance drift: {run_id}")
    db_episodes = connection.execute(
        """
        SELECT e.id,e.seed,e.topology_seed,e.episode_idx,e.total_reward,e.native_reward,
               e.length,e.terminal_state,e.goal_reached,e.hosts_compromised,
               e.topology_hash,e.known_nodes,e.true_nodes,e.discovery_coverage,
               e.invalid_mask_selections,e.failed_actions,e.deployment_profile,
               d.native_total_reward,d.detected_actions,d.detectable_actions,
               d.detection_rate,d.evasion_rate,d.exploit_attempts,d.mitigated_exploits,
               d.defender_actions,d.threshold_adjustments,d.patch_schedules,d.patch_activations
        FROM episodes e JOIN phase13_defense_episodes d ON d.episode_id=e.id
        WHERE e.run_id=%s ORDER BY e.episode_idx
        """,
        (run_id,),
    ).fetchall()
    require(
        len(db_episodes) == len(episodes),
        f"PostgreSQL episode count drift: {run_id}",
    )
    grouped: dict[tuple[str, int, int], list[dict]] = {}
    for step in steps:
        grouped.setdefault(
            (step["profile"], int(step["topology_seed"]), int(step["evaluation_seed"])), []
        ).append(step)
    total_steps = 0
    for index, (db, episode) in enumerate(zip(db_episodes, episodes, strict=True)):
        expected_episode = (
            int(episode["evaluation_seed"]), int(episode["topology_seed"]), index,
            float(episode["total_reward"]), float(episode["total_reward"]),
            int(episode["episode_length"]), episode["terminal_reason"],
            bool(episode["goal_reached"]), int(episode["hosts_compromised"]),
            episode["topology_hash"], int(episode["known_nodes"]), int(episode["true_nodes"]),
            float(episode["discovery_coverage"]), 0, int(episode["failed_actions"]),
            episode["profile"], float(episode["native_total_reward"]),
            int(episode["detected_actions"]), int(episode["detectable_actions"]),
            float(episode["detection_rate"]), float(episode["evasion_rate"]),
            int(episode["exploit_attempts"]), int(episode["mitigated_exploits"]),
            int(episode["defender_actions"]), int(episode["threshold_adjustments"]),
            int(episode["patch_schedules"]), int(episode["patch_activations"]),
        )
        actual_episode = tuple(
            float(value) if position in {3, 4, 12, 16, 19, 20} else value
            for position, value in enumerate(db[1:])
        )
        require(actual_episode == expected_episode, f"PostgreSQL episode drift: {run_id}/{index}")
        key = (episode["profile"], int(episode["topology_seed"]), int(episode["evaluation_seed"]))
        source = grouped[key]
        db_steps = connection.execute(
            """
            SELECT s.step_idx,s.action_name,s.action_kind,s.success,s.reward,s.native_reward,
                   s.cve_id,s.cvss_base,s.target_entity,s.state_changed,s.prerequisites,
                   s.outcomes,s.error,d.native_reward,d.detected,d.detection_probability,
                   d.ids_level,d.ids_level_name,d.defender_action,d.defender_action_index,
                   d.defender_action_changed,d.scheduled_vulnerability,
                   d.patch_activation_episode,d.pending_patch_count,d.active_patch_count,
                   d.mitigation_blocked,d.defender_reward
            FROM steps s JOIN phase13_defense_steps d ON d.step_id=s.id
            WHERE s.episode_id=%s ORDER BY s.step_idx
            """,
            (db[0],),
        ).fetchall()
        require(len(db_steps) == len(source), f"PostgreSQL step count drift: {run_id}")
        for stored, step in zip(db_steps, source, strict=True):
            expected_step = (
                int(step["step"]), step["action"], step["action_kind"], bool(step["success"]),
                float(step["reward"]), float(step["reward"]), step.get("cve_id"),
                float(step["cvss_base"]) if step.get("cvss_base") is not None else None,
                step["target_entity"], bool(step["state_changed"]), step["prerequisites"],
                step["outcomes"], step.get("reason"), float(step["native_reward"]),
                bool(step["detected"]), float(step["detection_probability"]),
                int(step["ids_level"]), step["ids_level_name"], step["defender_action"],
                int(step["defender_action_index"]), bool(step["defender_action_state_changed"]),
                step.get("scheduled_vulnerability"), step.get("patch_activation_episode"),
                int(step["pending_patch_count"]), int(step["active_patch_count"]),
                bool(step["mitigation_blocked"]), float(step["defender_reward"]),
            )
            actual_step = tuple(
                float(value) if position in {4, 5, 7, 13, 15, 26} and value is not None else value
                for position, value in enumerate(stored)
            )
            require(actual_step == expected_step, f"PostgreSQL step drift: {run_id}")
        total_steps += len(db_steps)
    require(
        metadata["database_episode_count"] == len(episodes),
        f"database metadata drift: {run_id}",
    )
    require(metadata["database_step_count"] == total_steps, f"database metadata drift: {run_id}")
    require(
        metadata["database_defense_episode_count"] == len(episodes),
        f"extension metadata drift: {run_id}",
    )
    require(
        metadata["database_defense_step_count"] == total_steps,
        f"extension metadata drift: {run_id}",
    )


def _verify_postgres(repo: Path, config, evaluations, run_data) -> dict[str, int]:
    import psycopg

    from rlredteam.storage.postgres_logger import connection_string

    try:
        connection = psycopg.connect(connection_string())
    except Exception as exc:
        raise MultiAgentCompletionError(f"cannot connect to PostgreSQL: {exc}") from exc
    experiments: set[int] = set()
    runs: set[int] = set()
    episode_count = step_count = 0
    with connection:
        for metadata in evaluations:
            key = (
                metadata["training_manifest"]["arm"],
                int(metadata["training_manifest"]["training_seed"]),
            )
            episodes, steps = run_data[key]
            _verify_postgres_run(connection, metadata, episodes, steps, config)
            experiments.add(int(metadata["database_experiment_id"]))
            runs.add(int(metadata["database_evaluation_run_id"]))
            episode_count += len(episodes)
            step_count += len(steps)
        require(len(experiments) == len(runs) == 20, "PostgreSQL IDs are not unique")
        study_count = connection.execute(
            "SELECT count(*) FROM experiments WHERE name LIKE %s",
            (f"{config.experiment_id}-%",),
        ).fetchone()[0]
        require(study_count == 20, "unexpected Phase 13 PostgreSQL experiment count")
    return {
        "experiments": len(experiments),
        "runs": len(runs),
        "episodes": episode_count,
        "steps": step_count,
    }


def verify_multiagent_completion(
    repo_root: Path, *, include_postgres: bool = False
) -> dict[str, Any]:
    repo = Path(repo_root).resolve()
    config, frozen, study = _verify_protocol(repo)
    manifests = study.get("red_training_manifests", [])
    evaluations = study.get("evaluation_metadata", [])
    expected = _expected_runs(config)
    require(len(manifests) == len(evaluations) == len(expected) == 20, "run count drift")
    by_manifest = {
        (item.get("arm"), int(item.get("training_seed", -1))): item for item in manifests
    }
    by_evaluation = {
        (
            item.get("training_manifest", {}).get("arm"),
            int(item.get("training_manifest", {}).get("training_seed", -1)),
        ): item
        for item in evaluations
    }
    require(set(by_manifest) == set(by_evaluation) == expected, "matched run grid drift")
    defender_manifest = study.get("defender_training_manifest", {})
    require(
        defender_manifest.get("actual_defender_training_timesteps")
        == config.defender_total_timesteps,
        "defender full training budget drifted",
    )
    require(
        defender_manifest.get("policy_sha256_before_training")
        != defender_manifest.get("policy_sha256"),
        "defender policy did not update",
    )
    for field in ("threshold_adjustments", "patch_schedules", "patch_activations"):
        require(int(defender_manifest.get(field, 0)) > 0, f"defender {field} absent")

    all_episodes: list[dict[str, Any]] = []
    run_data = {}
    paths = steps = 0
    for seed in config.training_seeds:
        validate_paired_red_training_isolation(
            by_manifest[(config.arms[0], seed)], by_manifest[(config.arms[1], seed)]
        )
    for key in sorted(expected):
        manifest = by_manifest[key]
        _verify_manifest(repo, config, frozen, study, manifest)
        episodes, trajectories, run_paths = _verify_run(
            repo, config, manifest, by_evaluation[key]
        )
        all_episodes.extend(episodes)
        run_data[key] = (episodes, trajectories)
        paths += run_paths
        steps += len(trajectories)
    report = _verify_analysis(repo, config, all_episodes)
    postgres = (
        _verify_postgres(repo, config, evaluations, run_data)
        if include_postgres
        else {"status": "not-requested"}
    )
    return {
        "complete": True,
        "study_id": config.experiment_id,
        "checks": {
            "prospective_protocol": "pass",
            "explicit_partial_knowledge": "pass",
            "matched_red_training": "pass",
            "alternating_frozen_opponents": "pass",
            "exact_training_budgets": "pass",
            "held_out_enterprise_profiles": "pass",
            "red_blue_mask_audits": "pass",
            "frozen_two_policy_evaluation": "pass",
            "dynamic_ids_delayed_mitigation": "pass",
            "causal_attack_paths": "pass",
            "paired_seed_statistics": "pass",
            "postgres_exact_reconstruction": (
                "pass" if include_postgres else "not-requested"
            ),
        },
        "evidence": {
            "runs": len(expected),
            "episodes": len(all_episodes),
            "steps": steps,
            "attack_paths": paths,
            "aggregate_red_training_steps": 20 * config.red_total_timesteps,
            "defender_training_steps": config.defender_total_timesteps,
        },
        "analysis": report,
        "postgres": postgres,
    }


__all__ = [
    "MultiAgentCompletionError",
    "require",
    "verify_multiagent_completion",
]
