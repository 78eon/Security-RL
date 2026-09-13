"""Read-only completion gate for the canonical Phase 10 graph-policy study."""

from __future__ import annotations

import csv
import json
import math
import subprocess
from pathlib import Path
from typing import Any

from rlredteam.enterprise.curriculum import canonical_digest
from rlredteam.enterprise.generalisation import evaluation_provenance, sha256_file
from rlredteam.enterprise.graph_policy import GraphPolicyResearchConfig
from rlredteam.enterprise.graph_policy_study import (
    current_input_manifest,
    observation_schema,
    validate_paired_training_isolation,
)
from rlredteam.historical_provenance import frozen_inputs_match


class GraphPolicyCompletionError(RuntimeError):
    """Raised when Phase 10 evidence fails a completion invariant."""


def require(condition: bool, message: str) -> None:
    """Fail closed when a required proof is absent."""
    if not condition:
        raise GraphPolicyCompletionError(message)


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


def _json(path: Path) -> Any:
    require(path.is_file(), f"missing evidence file: {path}")
    try:
        return json.loads(path.read_text(), parse_constant=_reject_constant)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise GraphPolicyCompletionError(f"invalid JSON evidence: {path}: {exc}") from exc


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
        raise GraphPolicyCompletionError(f"cannot verify Git provenance: {exc}") from exc


def _verify_protocol(repo_root: Path) -> tuple[GraphPolicyResearchConfig, dict, dict]:
    config = GraphPolicyResearchConfig.from_yaml(
        repo_root / "configs/experiments/graph_policy.yaml"
    )
    frozen = _json(repo_root / "configs/frozen_graph_policy.json")
    study = _json(repo_root / "results" / config.experiment_id / "test/metadata/study.json")
    require(isinstance(frozen, dict), "frozen inputs must be a JSON object")
    require(isinstance(study, dict), "study metadata must be a JSON object")
    current = current_input_manifest(config)
    frozen_matches, frozen_detail = frozen_inputs_match(repo_root, current, frozen)
    require(
        frozen_matches,
        f"Phase 10 frozen protocol cannot be reconstructed: {frozen_detail}",
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
    require(
        0 < float(study.get("projected_canonical_minutes", math.inf)) <= config.runtime_cap_minutes,
        "canonical runtime projection violates the frozen cap",
    )
    require(
        0 < float(study.get("training_wall_seconds", 0)) <= config.runtime_cap_minutes * 60,
        "canonical training wall time violates the frozen cap",
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
    schema = observation_schema()
    require(schema.get("source") == "AgentKnowledge", "observation source is unsafe")
    require(schema.get("hidden_topology_fields") == 0, "observation contains hidden topology")
    return config, frozen, study


def _expected_runs(config: GraphPolicyResearchConfig) -> set[tuple[str, int]]:
    return {(arm, seed) for arm in config.arms for seed in config.training_seeds}


def _verify_training_audit(
    config: GraphPolicyResearchConfig, manifest: dict[str, Any], *, run_name: str
) -> None:
    seed = int(manifest["training_seed"])
    reset_count = int(manifest.get("training_reset_count", 0))
    require(
        manifest.get("environment_seed") == seed * 100 + 1, f"environment seed drift: {run_name}"
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
        sum(map(int, profile_counts.values())) == reset_count, f"profile count drift: {run_name}"
    )


def _verify_run(
    repo_root: Path,
    config: GraphPolicyResearchConfig,
    frozen: dict[str, Any],
    manifest: dict[str, Any],
    metadata: dict[str, Any],
) -> dict[str, int]:
    arm = str(manifest.get("arm"))
    seed = int(manifest.get("training_seed", -1))
    run_name = f"{config.experiment_id}-{arm}-s{seed}"
    checkpoint = repo_root / "runs" / config.experiment_id / "canonical" / run_name / "model.zip"
    output = repo_root / "results" / config.experiment_id / "test" / "raw" / run_name
    representation = (
        "flat_agent_knowledge" if arm == "flat_mlp" else "agent_knowledge_message_passing_graph"
    )
    policy_config = config.baseline_policy if arm == "flat_mlp" else config.graph_policy
    expected_manifest = {
        "study_id": config.experiment_id,
        "experiment_id": run_name,
        "algorithm": "MaskablePPO",
        "policy": "MlpPolicy",
        "representation": representation,
        "action_mask_source": "AgentKnowledge",
        "observation_source": "AgentKnowledge",
        "observation_schema": observation_schema(),
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
        "ppo": config.common_ppo,
        "policy_config": policy_config,
        "frozen_inputs_sha256": canonical_digest(frozen),
        "git_dirty": False,
        "torch_num_threads": 1,
    }
    for field, value in expected_manifest.items():
        require(manifest.get(field) == value, f"manifest {field} drift: {run_name}")
    require(int(manifest.get("parameter_count", 0)) > 0, f"parameter count absent: {run_name}")
    require(checkpoint.is_file(), f"missing checkpoint: {run_name}")
    require(
        manifest.get("checkpoint_sha256") == sha256_file(checkpoint),
        f"checkpoint drift: {run_name}",
    )
    _verify_training_audit(config, manifest, run_name=run_name)

    require(metadata.get("training_manifest") == manifest, f"embedded manifest drift: {run_name}")
    for field, value in {
        "phase": "phase10_frozen_graph_representation_comparison",
        "split": "test",
        "gradient_updates": False,
        "evaluation_reset_count": 60,
        "invalid_mask_selections": 0,
        "checkpoint_sha256": manifest["checkpoint_sha256"],
    }.items():
        require(metadata.get(field) == value, f"evaluation {field} drift: {run_name}")
    require(int(metadata.get("knowledge_mask_checks", 0)) > 0, f"mask proof absent: {run_name}")
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
        (row["profile"], int(row["topology_seed"]), int(row["evaluation_seed"])) for row in rows
    }
    require(
        len(rows) == 60 and observed_keys == expected_keys, f"held-out grid mismatch: {run_name}"
    )
    distribution = manifest["base_distribution"]["test"]
    episode_success: dict[tuple[str, int, int], bool] = {}
    episode_lengths: dict[tuple[str, int, int], int] = {}
    for row in rows:
        key = (row["profile"], int(row["topology_seed"]), int(row["evaluation_seed"]))
        require(
            row["arm"] == arm and int(row["training_seed"]) == seed, f"episode drift: {run_name}"
        )
        require(int(row["invalid_mask_selections"]) == 0, f"invalid action row: {run_name}")
        require(int(row["failed_actions"]) == 0, f"failed action row: {run_name}")
        require(
            row["topology_hash"] == distribution[key[0]][str(key[1])], f"topology drift: {run_name}"
        )
        episode_success[key] = row["goal_reached"] == "True"
        episode_lengths[key] = int(row["episode_length"])

    trajectory_events: dict[tuple[str, int, int], set[tuple[Any, ...]]] = {}
    trajectory_count = 0
    trajectories = output / "trajectories.jsonl"
    require(trajectories.is_file(), f"missing trajectories: {run_name}")
    for line_number, line in enumerate(trajectories.read_text().splitlines(), start=1):
        try:
            step = json.loads(line, parse_constant=_reject_constant)
        except (ValueError, json.JSONDecodeError) as exc:
            raise GraphPolicyCompletionError(
                f"invalid trajectory {run_name}:{line_number}: {exc}"
            ) from exc
        key = (step["profile"], int(step["topology_seed"]), int(step["evaluation_seed"]))
        require(key in expected_keys, f"trajectory grid drift: {run_name}")
        require(
            step["arm"] == arm and int(step["training_seed"]) == seed,
            f"trajectory drift: {run_name}",
        )
        require(
            step["topology_hash"] == distribution[key[0]][str(key[1])],
            f"trajectory topology drift: {run_name}",
        )
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
        trajectory_count += 1
    require(set(trajectory_events) == expected_keys, f"trajectory coverage mismatch: {run_name}")
    require(
        trajectory_count == sum(episode_lengths.values()), f"trajectory length drift: {run_name}"
    )
    require(
        trajectory_count == int(metadata.get("database_step_count", -1)),
        f"step count drift: {run_name}",
    )

    attack_paths = _json(output / "attack_paths.json")
    require(
        isinstance(attack_paths, list) and len(attack_paths) == 60,
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
                fingerprint in trajectory_events[key], f"unrecorded attack-path event: {run_name}"
            )
    require(path_keys == expected_keys, f"attack-path coverage mismatch: {run_name}")

    summary = _json(output / "summary.json")
    require(summary.get("episode_count") == 60, f"summary episode drift: {run_name}")
    require(summary.get("topology_count") == 20, f"summary topology drift: {run_name}")
    require(summary.get("invalid_mask_selections") == 0, f"summary invalid actions: {run_name}")
    require(
        metadata.get("database_episode_count") == 60, f"database episode metadata drift: {run_name}"
    )
    return {
        "episodes": 60,
        "steps": trajectory_count,
        "attack_paths": 60,
        "successful_episodes": sum(episode_success.values()),
    }


def _verify_analysis(repo_root: Path, config: GraphPolicyResearchConfig) -> dict[str, Any]:
    root = repo_root / "results" / config.experiment_id / "test"
    analysis = _json(root / "summaries/analysis.json")
    require(analysis.get("complete") is True, "paired analysis is incomplete")
    require(analysis.get("primary_unit") == "training_seed", "paired unit drifted")
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
        analysis.get("primary_metrics") == list(config.primary_metrics), "primary metrics drifted"
    )
    comparisons = analysis.get("comparisons", [])
    expected_metrics = set(config.primary_metrics + config.descriptive_metrics)
    require(
        {item.get("metric") for item in comparisons} == expected_metrics, "analysis metrics drifted"
    )
    for item in comparisons:
        require(item.get("n_pairs") == len(config.training_seeds), "paired sample size drifted")
        for field in ("difference", "mean_a", "mean_b", "p_value"):
            require(
                math.isfinite(float(item[field])), f"non-finite statistic: {item['metric']}.{field}"
            )
        if item["metric"] in config.primary_metrics:
            require(
                math.isfinite(float(item["p_bonferroni"])),
                f"missing corrected p-value: {item['metric']}",
            )
    require(
        len(_csv(root / "summaries/seed_metrics.csv")) == 20, "seed-level metric grid is incomplete"
    )
    require((root / "tables/statistics.csv").is_file(), "statistics table is absent")
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
                "graph_mean": item["mean_b"],
                "graph_minus_flat": item["difference"],
                "p_bonferroni": item["p_bonferroni"],
                "significant": item["significant"],
            }
            for item in comparisons
            if item["metric"] in config.primary_metrics
        },
    }


def _verify_postgres(
    evaluations: list[dict[str, Any]], config: GraphPolicyResearchConfig
) -> dict[str, int]:
    import psycopg

    from rlredteam.storage.postgres_logger import connection_string

    try:
        connection = psycopg.connect(connection_string())
    except Exception as exc:
        raise GraphPolicyCompletionError(f"cannot connect to PostgreSQL: {exc}") from exc
    episodes = steps = successes = 0
    experiment_ids: set[int] = set()
    run_ids: set[int] = set()
    with connection:
        for metadata in evaluations:
            experiment_id = int(metadata["database_experiment_id"])
            run_id = int(metadata["database_evaluation_run_id"])
            manifest = metadata["training_manifest"]
            provenance = evaluation_provenance(manifest, "test")
            row = connection.execute(
                """
                SELECT r.status, r.designation, r.evaluation_seeds, r.seed,
                       e.name, e.algorithm, e.git_sha, e.config_hash,
                       e.topology_config_hash, e.topology_hash,
                       e.cve_manifest_sha256, e.seed_set, e.condition
                FROM runs r JOIN experiments e ON e.id=r.experiment_id
                WHERE r.id=%s AND e.id=%s
                """,
                (run_id, experiment_id),
            ).fetchone()
            require(row is not None, f"PostgreSQL run is absent: {run_id}")
            expected = (
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
            )
            require(tuple(row) == expected, f"PostgreSQL provenance drift: {run_id}")
            counts = connection.execute(
                """
                SELECT count(*), coalesce(sum((goal_reached)::int), 0),
                       coalesce(sum(invalid_mask_selections), 0),
                       coalesce(sum(failed_actions), 0), coalesce(sum(length), 0)
                FROM episodes WHERE run_id=%s
                """,
                (run_id,),
            ).fetchone()
            step_count = connection.execute(
                """SELECT count(*) FROM steps s JOIN episodes ep ON ep.id=s.episode_id
                   WHERE ep.run_id=%s""",
                (run_id,),
            ).fetchone()[0]
            require(
                counts[0] == metadata["database_episode_count"] == 60,
                f"PostgreSQL episode drift: {run_id}",
            )
            require(counts[2] == counts[3] == 0, f"PostgreSQL action failure: {run_id}")
            require(
                step_count == counts[4] == metadata["database_step_count"],
                f"PostgreSQL step drift: {run_id}",
            )
            episodes += counts[0]
            successes += counts[1]
            steps += step_count
            experiment_ids.add(experiment_id)
            run_ids.add(run_id)
        require(len(experiment_ids) == len(evaluations), "PostgreSQL experiment IDs are reused")
        require(len(run_ids) == len(evaluations), "PostgreSQL run IDs are reused")
        study_count = connection.execute(
            "SELECT count(*) FROM experiments WHERE name LIKE %s",
            (f"{config.experiment_id}-%",),
        ).fetchone()[0]
        require(study_count == len(evaluations), "unexpected Phase 10 PostgreSQL experiment count")
    return {
        "experiments": len(experiment_ids),
        "runs": len(run_ids),
        "episodes": episodes,
        "successful_episodes": successes,
        "steps": steps,
    }


def verify_graph_policy_completion(
    repo_root: Path, *, include_postgres: bool = False
) -> dict[str, Any]:
    """Verify all canonical Phase 10 evidence without mutating it."""
    repo_root = Path(repo_root).resolve()
    config, frozen, study = _verify_protocol(repo_root)
    manifests = study.get("training_manifests", [])
    evaluations = study.get("evaluation_metadata", [])
    require(len(manifests) == len(evaluations) == 20, "canonical run count mismatch")
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
    expected = _expected_runs(config)
    require(set(manifests_by_key) == expected, "training manifest grid mismatch")
    require(set(evaluations_by_key) == expected, "evaluation metadata grid mismatch")
    for seed in config.training_seeds:
        validate_paired_training_isolation(
            manifests_by_key[(config.arms[0], seed)],
            manifests_by_key[(config.arms[1], seed)],
        )
    parameter_counts = {
        arm: {
            int(manifests_by_key[(arm, seed)]["parameter_count"]) for seed in config.training_seeds
        }
        for arm in config.arms
    }
    require(
        all(len(values) == 1 for values in parameter_counts.values()),
        "parameter count drift within an arm",
    )

    totals = {"runs": 0, "episodes": 0, "steps": 0, "attack_paths": 0, "successful_episodes": 0}
    for key in sorted(expected):
        counts = _verify_run(
            repo_root, config, frozen, manifests_by_key[key], evaluations_by_key[key]
        )
        totals["runs"] += 1
        for field in ("episodes", "steps", "attack_paths", "successful_episodes"):
            totals[field] += counts[field]
    require(totals["runs"] * config.total_timesteps == 1_003_520, "aggregate training budget drift")
    analysis = _verify_analysis(repo_root, config)
    postgres: dict[str, Any] = (
        _verify_postgres(evaluations, config) if include_postgres else {"status": "not-requested"}
    )
    return {
        "complete": True,
        "study_id": config.experiment_id,
        "checks": {
            "frozen_protocol": "pass",
            "agent_knowledge_only_observation": "pass",
            "matched_representation_isolation": "pass",
            "exact_training_budgets": "pass",
            "deterministic_reset_audit": "pass",
            "agent_knowledge_masks": "pass",
            "frozen_evaluation": "pass",
            "policy_immutability": "pass",
            "causal_attack_paths": "pass",
            "paired_seed_analysis": "pass",
            "postgres_reconstruction": "pass" if include_postgres else "not-requested",
        },
        "evidence": totals,
        "parameter_counts": {arm: next(iter(values)) for arm, values in parameter_counts.items()},
        "analysis": analysis,
        "postgres": postgres,
        "runtime": {
            "training_wall_minutes": study["training_wall_seconds"] / 60,
            "projected_minutes": study["projected_canonical_minutes"],
            "cap_minutes": config.runtime_cap_minutes,
        },
    }
