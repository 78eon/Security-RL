"""Read-only completion gate for the canonical Phase 11 transfer study."""

from __future__ import annotations

import csv
import json
import math
import subprocess
from pathlib import Path
from typing import Any

from rlredteam.enterprise.curriculum import canonical_digest
from rlredteam.enterprise.generalisation import evaluation_provenance, sha256_file
from rlredteam.enterprise.transfer_learning import TransferResearchConfig, observation_schema
from rlredteam.enterprise.transfer_study import (
    current_input_manifest,
    validate_paired_target_isolation,
)


class TransferCompletionError(RuntimeError):
    """Raised when Phase 11 evidence fails a completion invariant."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise TransferCompletionError(message)


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


def _json(path: Path) -> Any:
    require(path.is_file(), f"missing evidence file: {path}")
    try:
        return json.loads(path.read_text(), parse_constant=_reject_constant)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise TransferCompletionError(f"invalid JSON evidence: {path}: {exc}") from exc


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
        raise TransferCompletionError(f"cannot verify Git provenance: {exc}") from exc


def _verify_protocol(repo_root: Path) -> tuple[TransferResearchConfig, dict, dict]:
    config = TransferResearchConfig.from_yaml(
        repo_root / "configs/experiments/transfer_learning.yaml"
    )
    frozen = _json(repo_root / "configs/frozen_transfer_learning.json")
    study = _json(repo_root / "results" / config.experiment_id / "test/metadata/study.json")
    current = current_input_manifest(config)
    require(
        current == {key: frozen.get(key) for key in current},
        "current Phase 11 inputs differ from the frozen protocol",
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
        "source_pretrain_timesteps": config.source_pretrain_timesteps,
        "target_adaptation_timesteps": config.target_adaptation_timesteps,
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
    require(schema["source"] == "AgentKnowledge", "observation source is unsafe")
    require(schema["hidden_topology_fields"] == 0, "observation contains hidden topology")
    return config, frozen, study


def _verify_audit(
    audit: dict[str, Any],
    distribution: dict[str, dict[str, str]],
    *,
    profiles: set[str],
    environment_seed: int,
    label: str,
) -> None:
    count = int(audit.get("reset_count", 0))
    require(count > 0, f"reset evidence absent: {label}")
    require(audit.get("environment_seed") == environment_seed, f"environment seed drift: {label}")
    require(bool(audit.get("reset_trace_sha256")), f"reset trace absent: {label}")
    profile_counts = audit.get("observed_profile_counts", {})
    require(set(profile_counts) == profiles, f"profile coverage drift: {label}")
    require(sum(map(int, profile_counts.values())) == count, f"profile reset count drift: {label}")
    exposure = audit.get("exposure_counts", {})
    possible = {f"{profile}:{seed}" for profile, seeds in distribution.items() for seed in seeds}
    require(bool(exposure) and set(exposure) <= possible, f"training exposure drift: {label}")
    require(sum(map(int, exposure.values())) == count, f"exposure reset count drift: {label}")


def _verify_training(
    repo_root: Path,
    config: TransferResearchConfig,
    frozen: dict,
    manifest: dict[str, Any],
) -> None:
    arm = str(manifest["arm"])
    seed = int(manifest["training_seed"])
    run_name = f"{config.experiment_id}-{arm}-s{seed}"
    checkpoint = repo_root / "runs" / config.experiment_id / "canonical" / run_name / "model.zip"
    expected = {
        "study_id": config.experiment_id,
        "experiment_id": run_name,
        "algorithm": "MaskablePPO",
        "policy": "MlpPolicy",
        "representation": "agent_knowledge_message_passing_graph",
        "action_mask_source": "AgentKnowledge",
        "observation_source": "AgentKnowledge",
        "observation_schema": observation_schema(),
        "development": False,
        "training_seed": seed,
        "target_adaptation_timesteps": config.target_adaptation_timesteps,
        "actual_target_adaptation_timesteps": config.target_adaptation_timesteps,
        "target_profiles": [item.value for item in config.target_profiles],
        "source_topology_seeds": list(config.topology_splits["source_train"]),
        "target_train_topology_seeds": list(config.topology_splits["target_train"]),
        "validation_topology_seeds": list(config.topology_splits["validation"]),
        "test_topology_seeds": list(config.topology_splits["test"]),
        "study_config_hash": config.digest(),
        "scientific_config_hash": config.scientific_digest(),
        "ppo": config.common_ppo,
        "policy_config": config.graph_policy,
        "frozen_inputs_sha256": canonical_digest(frozen),
        "git_dirty": False,
        "torch_num_threads": 1,
    }
    for field, value in expected.items():
        require(manifest.get(field) == value, f"manifest {field} drift: {run_name}")
    require(checkpoint.is_file(), f"missing checkpoint: {run_name}")
    require(
        manifest.get("checkpoint_sha256") == sha256_file(checkpoint),
        f"checkpoint drift: {run_name}",
    )
    require(
        manifest["policy_sha256"] != manifest["pre_adaptation_policy_sha256"],
        f"target policy did not adapt: {run_name}",
    )
    _verify_audit(
        manifest["target_training_audit"],
        manifest["base_distribution"]["target_train"],
        profiles={"hybrid"},
        environment_seed=seed * 100 + 2,
        label=f"{run_name}/target",
    )
    source = manifest.get("source_pretraining")
    if arm == "scratch_hybrid":
        require(
            manifest.get("initialization_type") == "random",
            f"scratch initialization drift: {run_name}",
        )
        require(source is None, f"scratch has source evidence: {run_name}")
    else:
        require(
            manifest.get("initialization_type") == "legacy_cloud_source_checkpoint",
            f"transfer initialization drift: {run_name}",
        )
        require(isinstance(source, dict), f"source evidence absent: {run_name}")
        require(source.get("profiles") == ["legacy", "cloud"], f"source profile drift: {run_name}")
        require(
            source.get("topology_seeds") == list(config.topology_splits["source_train"]),
            f"source topology drift: {run_name}",
        )
        require(
            source.get("training_timesteps")
            == source.get("actual_training_timesteps")
            == config.source_pretrain_timesteps,
            f"source budget drift: {run_name}",
        )
        source_checkpoint = checkpoint.with_name("source_model.zip")
        require(source_checkpoint.is_file(), f"source checkpoint absent: {run_name}")
        require(
            source.get("checkpoint_sha256") == sha256_file(source_checkpoint),
            f"source checkpoint drift: {run_name}",
        )
        require(
            source.get("policy_sha256") == manifest.get("pre_adaptation_policy_sha256"),
            f"source policy was not loaded: {run_name}",
        )
        require(
            source.get("policy_sha256") != source.get("initial_policy_sha256"),
            f"source policy did not train: {run_name}",
        )
        _verify_audit(
            source,
            manifest["base_distribution"]["source_train"],
            profiles={"legacy", "cloud"},
            environment_seed=seed * 100 + 1,
            label=f"{run_name}/source",
        )


def _verify_evaluation(
    repo_root: Path,
    config: TransferResearchConfig,
    manifest: dict[str, Any],
    metadata: dict[str, Any],
) -> dict[str, int]:
    arm = str(manifest["arm"])
    seed = int(manifest["training_seed"])
    run_name = manifest["experiment_id"]
    output = repo_root / "results" / config.experiment_id / "test/raw" / run_name
    expected_metadata = {
        "phase": "phase11_frozen_transfer_comparison",
        "split": "test",
        "gradient_updates": False,
        "evaluation_reset_count": 20,
        "invalid_mask_selections": 0,
        "checkpoint_sha256": manifest["checkpoint_sha256"],
    }
    require(metadata.get("training_manifest") == manifest, f"embedded manifest drift: {run_name}")
    for field, value in expected_metadata.items():
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
        ("hybrid", topology_seed, evaluation_seed)
        for topology_seed in config.topology_splits["test"]
        for evaluation_seed in config.evaluation_episode_seeds
    }
    observed_keys = {
        (row["profile"], int(row["topology_seed"]), int(row["evaluation_seed"])) for row in rows
    }
    require(
        len(rows) == 20 and observed_keys == expected_keys, f"held-out grid mismatch: {run_name}"
    )
    distribution = manifest["base_distribution"]["test"]
    success: dict[tuple[str, int, int], bool] = {}
    lengths: dict[tuple[str, int, int], int] = {}
    for row in rows:
        key = (row["profile"], int(row["topology_seed"]), int(row["evaluation_seed"]))
        require(
            row["arm"] == arm and int(row["training_seed"]) == seed,
            f"episode identity drift: {run_name}",
        )
        require(
            int(row["invalid_mask_selections"]) == int(row["failed_actions"]) == 0,
            f"invalid/failed action: {run_name}",
        )
        require(
            row["topology_hash"] == distribution["hybrid"][str(key[1])],
            f"topology drift: {run_name}",
        )
        success[key] = row["goal_reached"] == "True"
        lengths[key] = int(row["episode_length"])

    events: dict[tuple[str, int, int], set[tuple[Any, ...]]] = {}
    trajectories = output / "trajectories.jsonl"
    require(trajectories.is_file(), f"missing trajectories: {run_name}")
    count = 0
    for line_number, line in enumerate(trajectories.read_text().splitlines(), start=1):
        try:
            step = json.loads(line, parse_constant=_reject_constant)
        except (ValueError, json.JSONDecodeError) as exc:
            raise TransferCompletionError(
                f"invalid trajectory {run_name}:{line_number}: {exc}"
            ) from exc
        key = (step["profile"], int(step["topology_seed"]), int(step["evaluation_seed"]))
        require(key in expected_keys, f"trajectory grid drift: {run_name}")
        require(
            step["arm"] == arm and int(step["training_seed"]) == seed,
            f"trajectory identity drift: {run_name}",
        )
        require(
            step["topology_hash"] == distribution["hybrid"][str(key[1])],
            f"trajectory topology drift: {run_name}",
        )
        events.setdefault(key, set()).add(
            (
                int(step["step"]),
                step["action"],
                step["success"],
                step["state_changed"],
                step["goal_reached"],
                step["topology_hash"],
            )
        )
        count += 1
    require(set(events) == expected_keys, f"trajectory coverage mismatch: {run_name}")
    require(
        count == sum(lengths.values()) == int(metadata.get("database_step_count", -1)),
        f"step count drift: {run_name}",
    )

    paths = _json(output / "attack_paths.json")
    require(isinstance(paths, list) and len(paths) == 20, f"attack-path count drift: {run_name}")
    path_keys = set()
    for path in paths:
        key = (path["profile"], int(path["topology_seed"]), int(path["evaluation_seed"]))
        require(key in expected_keys, f"attack-path grid drift: {run_name}")
        path_keys.add(key)
        steps = path.get("steps", [])
        if success[key]:
            require(bool(steps), f"successful episode has empty path: {run_name}")
            require(steps[-1].get("goal_reached") is True, f"path lacks goal: {run_name}")
            require(
                steps[-1].get("action_kind") == "access_asset",
                f"path lacks asset access: {run_name}",
            )
        require(
            all(
                step.get("success") is True and step.get("state_changed") is True for step in steps
            ),
            f"path contains non-causal event: {run_name}",
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
            require(fingerprint in events[key], f"unrecorded path event: {run_name}")
    require(path_keys == expected_keys, f"attack-path coverage mismatch: {run_name}")
    summary = _json(output / "summary.json")
    require(
        summary.get("episode_count") == summary.get("topology_count") == 20,
        f"summary grid drift: {run_name}",
    )
    require(summary.get("invalid_mask_selections") == 0, f"summary invalid actions: {run_name}")
    require(metadata.get("database_episode_count") == 20, f"database metadata drift: {run_name}")
    return {
        "episodes": 20,
        "steps": count,
        "attack_paths": 20,
        "successful_episodes": sum(success.values()),
    }


def _verify_analysis(repo_root: Path, config: TransferResearchConfig) -> dict[str, Any]:
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
    comparisons = analysis.get("comparisons", [])
    expected_metrics = set(config.primary_metrics + config.descriptive_metrics)
    require(
        {item.get("metric") for item in comparisons} == expected_metrics, "analysis metrics drifted"
    )
    for item in comparisons:
        require(item.get("n_pairs") == 10, "paired sample size drifted")
        for field in ("difference", "mean_a", "mean_b", "p_value"):
            require(
                math.isfinite(float(item[field])), f"non-finite statistic: {item['metric']}.{field}"
            )
        if item["metric"] in config.primary_metrics:
            require(
                math.isfinite(float(item["p_bonferroni"])),
                f"missing corrected p-value: {item['metric']}",
            )
    require(len(_csv(root / "summaries/seed_metrics.csv")) == 20, "seed-level grid is incomplete")
    require((root / "tables/statistics.csv").is_file(), "statistics table is absent")
    return {
        "pairs": 10,
        "comparisons": len(comparisons),
        "significant_primary_results": sum(
            bool(item["significant"])
            for item in comparisons
            if item["metric"] in config.primary_metrics
        ),
        "primary_results": {
            item["metric"]: {
                "scratch_mean": item["mean_a"],
                "transfer_mean": item["mean_b"],
                "transfer_minus_scratch": item["difference"],
                "p_bonferroni": item["p_bonferroni"],
                "significant": item["significant"],
            }
            for item in comparisons
            if item["metric"] in config.primary_metrics
        },
    }


def _verify_postgres(
    evaluations: list[dict[str, Any]], config: TransferResearchConfig
) -> dict[str, int]:
    import psycopg

    from rlredteam.storage.postgres_logger import connection_string

    try:
        connection = psycopg.connect(connection_string())
    except Exception as exc:
        raise TransferCompletionError(f"cannot connect to PostgreSQL: {exc}") from exc
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
            require(
                row is not None and tuple(row) == expected, f"PostgreSQL provenance drift: {run_id}"
            )
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
                counts[0] == metadata["database_episode_count"] == 20,
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
        require(study_count == len(evaluations), "unexpected Phase 11 PostgreSQL experiment count")
    return {
        "experiments": len(experiment_ids),
        "runs": len(run_ids),
        "episodes": episodes,
        "successful_episodes": successes,
        "steps": steps,
    }


def verify_transfer_completion(
    repo_root: Path, *, include_postgres: bool = False
) -> dict[str, Any]:
    """Verify all canonical Phase 11 evidence without mutating it."""
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
    expected = {(arm, seed) for arm in config.arms for seed in config.training_seeds}
    require(
        set(manifests_by_key) == set(evaluations_by_key) == expected, "matched run grid mismatch"
    )
    for seed in config.training_seeds:
        validate_paired_target_isolation(
            manifests_by_key[(config.arms[0], seed)],
            manifests_by_key[(config.arms[1], seed)],
        )
    parameter_counts = {int(item["parameter_count"]) for item in manifests}
    require(len(parameter_counts) == 1, "architecture parameter count differs between arms")
    totals = {"runs": 0, "episodes": 0, "steps": 0, "attack_paths": 0, "successful_episodes": 0}
    for key in sorted(expected):
        manifest = manifests_by_key[key]
        _verify_training(repo_root, config, frozen, manifest)
        counts = _verify_evaluation(repo_root, config, manifest, evaluations_by_key[key])
        totals["runs"] += 1
        for field in ("episodes", "steps", "attack_paths", "successful_episodes"):
            totals[field] += counts[field]
    analysis = _verify_analysis(repo_root, config)
    postgres: dict[str, Any] = (
        _verify_postgres(evaluations, config) if include_postgres else {"status": "not-requested"}
    )
    return {
        "complete": True,
        "study_id": config.experiment_id,
        "checks": {
            "frozen_protocol": "pass",
            "source_target_disjointness": "pass",
            "agent_knowledge_only_observation": "pass",
            "architecture_matched": "pass",
            "source_checkpoint_loaded": "pass",
            "equal_target_budget": "pass",
            "profile_exposure_audit": "pass",
            "agent_knowledge_masks": "pass",
            "frozen_evaluation": "pass",
            "policy_immutability": "pass",
            "causal_attack_paths": "pass",
            "paired_seed_analysis": "pass",
            "postgres_reconstruction": "pass" if include_postgres else "not-requested",
        },
        "evidence": {
            **totals,
            "source_training_steps": len(config.training_seeds) * config.source_pretrain_timesteps,
            "target_training_steps": len(manifests) * config.target_adaptation_timesteps,
        },
        "parameter_count_per_arm": next(iter(parameter_counts)),
        "analysis": analysis,
        "postgres": postgres,
        "runtime": {
            "training_wall_minutes": study["training_wall_seconds"] / 60,
            "projected_minutes": study["projected_canonical_minutes"],
            "cap_minutes": config.runtime_cap_minutes,
        },
    }
