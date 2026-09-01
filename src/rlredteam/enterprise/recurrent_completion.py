"""Read-only completion gate for the canonical Phase 8 recurrent study."""

from __future__ import annotations

import csv
import json
import math
import subprocess
from pathlib import Path
from typing import Any

from rlredteam.enterprise.generalisation import sha256_file
from rlredteam.enterprise.recurrent import RecurrentResearchConfig, _canonical_digest
from rlredteam.enterprise.recurrent_study import current_input_manifest


class RecurrentCompletionError(RuntimeError):
    """Raised when Phase 8 evidence fails a completion invariant."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RecurrentCompletionError(message)


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


def _json(path: Path) -> Any:
    require(path.is_file(), f"missing evidence file: {path}")
    try:
        return json.loads(path.read_text(), parse_constant=_reject_constant)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise RecurrentCompletionError(f"invalid JSON evidence: {path}: {exc}") from exc


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
        raise RecurrentCompletionError(f"cannot verify Git provenance: {exc}") from exc


def _verify_protocol(repo_root: Path) -> tuple[RecurrentResearchConfig, dict, dict]:
    config = RecurrentResearchConfig.from_yaml(
        repo_root / "configs" / "experiments" / "recurrent_policy.yaml"
    )
    frozen = _json(repo_root / "configs" / "frozen_recurrent_policy.json")
    study = _json(repo_root / "results" / config.experiment_id / "test" / "metadata" / "study.json")
    require(isinstance(frozen, dict), "frozen inputs must be a JSON object")
    require(isinstance(study, dict), "study metadata must be a JSON object")
    current = current_input_manifest(config)
    require(
        current == {key: frozen.get(key) for key in current},
        "current Phase 8 inputs differ from the frozen protocol",
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
    for field in ("protocol_commit",):
        commit = str(frozen.get(field, ""))
        require(bool(commit), f"frozen {field} is absent")
        require(
            _git(repo_root, "cat-file", "-e", f"{commit}^{{commit}}").returncode == 0,
            f"frozen {field} is not present in Git history",
        )
        require(
            _git(repo_root, "merge-base", "--is-ancestor", commit, "HEAD").returncode == 0,
            f"frozen {field} is not an ancestor of HEAD",
        )
    execution_commit = str(study.get("code_commit", ""))
    require(bool(execution_commit), "canonical execution commit is absent")
    require(
        _git(repo_root, "merge-base", "--is-ancestor", execution_commit, "HEAD").returncode == 0,
        "canonical execution commit is not an ancestor of HEAD",
    )
    return config, frozen, study


def _expected_runs(config: RecurrentResearchConfig) -> set[tuple[str, int]]:
    return {(arm, seed) for arm in config.arms for seed in config.training_seeds}


def _verify_run(
    repo_root: Path,
    config: RecurrentResearchConfig,
    frozen: dict,
    manifest: dict,
    metadata: dict,
) -> dict[str, int]:
    arm = str(manifest.get("arm"))
    seed = int(manifest.get("training_seed", -1))
    run_name = f"{config.experiment_id}-{arm}-s{seed}"
    checkpoint = repo_root / "runs" / config.experiment_id / "canonical" / run_name / "model.zip"
    output = repo_root / "results" / config.experiment_id / "test" / "raw" / run_name
    require(manifest.get("experiment_id") == run_name, f"run name mismatch: {run_name}")
    require(manifest.get("development") is False, f"development model in test grid: {run_name}")
    require(manifest.get("git_dirty") is False, f"dirty training provenance: {run_name}")
    require(
        manifest.get("actual_training_timesteps") == config.total_timesteps,
        f"training budget mismatch: {run_name}",
    )
    require(manifest.get("study_config_hash") == config.digest(), f"config drift: {run_name}")
    require(
        manifest.get("frozen_inputs_sha256") == _canonical_digest(frozen),
        f"frozen-input drift: {run_name}",
    )
    require(manifest.get("torch_num_threads") == 1, f"thread control drift: {run_name}")
    require(checkpoint.is_file(), f"missing checkpoint: {run_name}")
    require(
        manifest.get("checkpoint_sha256") == sha256_file(checkpoint),
        f"checkpoint hash mismatch: {run_name}",
    )
    require(metadata.get("training_manifest") == manifest, f"embedded manifest drift: {run_name}")
    require(
        metadata.get("phase") == "phase8_frozen_recurrent_comparison", f"phase drift: {run_name}"
    )
    require(metadata.get("split") == "test", f"evaluation split drift: {run_name}")
    require(metadata.get("gradient_updates") is False, f"evaluation updated policy: {run_name}")
    require(metadata.get("invalid_mask_selections") == 0, f"invalid action selected: {run_name}")
    require(
        metadata.get("checkpoint_sha256") == manifest.get("checkpoint_sha256"),
        f"evaluated checkpoint drift: {run_name}",
    )
    require(
        metadata.get("policy_sha256_before")
        == metadata.get("policy_sha256_after")
        == manifest.get("policy_sha256"),
        f"policy changed during evaluation: {run_name}",
    )
    if arm == "knowledge_masked_recurrent_ppo":
        require(metadata.get("state_reset_count") == 60, f"recurrent reset mismatch: {run_name}")
        require(
            int(metadata.get("recurrent_predictions", 0))
            == int(metadata.get("mask_transport_checks", -1))
            > 0,
            f"recurrent mask transport mismatch: {run_name}",
        )
    else:
        require(
            metadata.get("state_reset_count")
            == metadata.get("recurrent_predictions")
            == metadata.get("mask_transport_checks")
            == 0,
            f"baseline recurrent counters are non-zero: {run_name}",
        )

    rows = _csv(output / "episodes.csv")
    require(len(rows) == 60, f"held-out episode count mismatch: {run_name}")
    expected_keys = {
        (profile.value, topology_seed, evaluation_seed)
        for profile in config.train_profiles
        for topology_seed in config.topology_splits["test"]
        for evaluation_seed in config.evaluation_episode_seeds
    }
    observed_keys = {
        (row["profile"], int(row["topology_seed"]), int(row["evaluation_seed"])) for row in rows
    }
    require(observed_keys == expected_keys, f"held-out grid mismatch: {run_name}")
    distribution = manifest["distribution"]["test"]
    for row in rows:
        profile = row["profile"]
        topology_seed = row["topology_seed"]
        require(row["arm"] == arm, f"episode arm mismatch: {run_name}")
        require(int(row["training_seed"]) == seed, f"episode seed mismatch: {run_name}")
        require(row["goal_reached"] == "True", f"failed held-out episode: {run_name}")
        require(row["terminal_reason"] == "goal", f"non-goal terminal: {run_name}")
        require(int(row["invalid_mask_selections"]) == 0, f"invalid action row: {run_name}")
        require(int(row["failed_actions"]) == 0, f"failed action row: {run_name}")
        require(
            row["topology_hash"] == distribution[profile][topology_seed],
            f"topology hash mismatch: {run_name}",
        )

    trajectory_count = 0
    trajectory_keys: set[tuple[str, int, int]] = set()
    trajectory_events: dict[tuple[str, int, int], set[tuple]] = {}
    trajectories = output / "trajectories.jsonl"
    require(trajectories.is_file(), f"missing trajectories: {run_name}")
    for line_number, line in enumerate(trajectories.read_text().splitlines(), start=1):
        try:
            step = json.loads(line, parse_constant=_reject_constant)
        except (ValueError, json.JSONDecodeError) as exc:
            raise RecurrentCompletionError(
                f"invalid trajectory {run_name}:{line_number}: {exc}"
            ) from exc
        key = (step["profile"], int(step["topology_seed"]), int(step["evaluation_seed"]))
        require(key in expected_keys, f"trajectory grid mismatch: {run_name}")
        require(step["arm"] == arm, f"trajectory arm mismatch: {run_name}")
        require(int(step["training_seed"]) == seed, f"trajectory seed mismatch: {run_name}")
        require(
            step["topology_hash"] == distribution[key[0]][str(key[1])],
            f"trajectory topology hash mismatch: {run_name}",
        )
        trajectory_keys.add(key)
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
    require(trajectory_keys == expected_keys, f"trajectory episode coverage mismatch: {run_name}")
    require(
        trajectory_count == int(metadata.get("database_step_count", -1)),
        f"trajectory/database step mismatch: {run_name}",
    )

    attack_paths = _json(output / "attack_paths.json")
    require(isinstance(attack_paths, list), f"attack paths are not a list: {run_name}")
    require(len(attack_paths) == 60, f"attack-path count mismatch: {run_name}")
    path_keys = set()
    for path in attack_paths:
        key = (path["profile"], int(path["topology_seed"]), int(path["evaluation_seed"]))
        path_keys.add(key)
        steps = path.get("steps", [])
        require(bool(steps), f"empty successful attack path: {run_name}")
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
                f"attack path contains an unrecorded event: {run_name}",
            )
    require(path_keys == expected_keys, f"attack-path grid mismatch: {run_name}")

    summary = _json(output / "summary.json")
    require(summary.get("episode_count") == 60, f"summary episode mismatch: {run_name}")
    require(summary.get("topology_count") == 20, f"summary topology mismatch: {run_name}")
    require(summary.get("success_rate") == 1.0, f"summary success mismatch: {run_name}")
    require(summary.get("invalid_mask_selections") == 0, f"summary invalid actions: {run_name}")
    require(
        metadata.get("database_episode_count") == 60,
        f"database episode metadata mismatch: {run_name}",
    )
    return {"episodes": 60, "steps": trajectory_count, "attack_paths": 60}


def _verify_analysis(repo_root: Path, config: RecurrentResearchConfig) -> dict:
    analysis = _json(
        repo_root / "results" / config.experiment_id / "test" / "summaries" / "analysis.json"
    )
    require(analysis.get("complete") is True, "paired analysis is incomplete")
    require(analysis.get("primary_unit") == "training_seed", "paired unit drifted")
    require(
        analysis.get("expected_training_seeds") == list(config.training_seeds),
        "paired seed set drifted",
    )
    require(
        analysis.get("observed_training_seeds")
        == {arm: list(config.training_seeds) for arm in config.arms},
        "paired arm/seed grid is incomplete",
    )
    require(
        analysis.get("primary_metrics") == list(config.primary_metrics),
        "primary metric set drifted",
    )
    comparisons = analysis.get("comparisons", [])
    require(
        {item.get("metric") for item in comparisons}
        == set(config.primary_metrics + config.descriptive_metrics),
        "analysis metric set is incomplete",
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
    statistics = repo_root / "results" / config.experiment_id / "test" / "tables" / "statistics.csv"
    require(statistics.is_file(), "statistics table is absent")
    return {
        "pairs": len(config.training_seeds),
        "comparisons": len(comparisons),
        "significant_primary_results": sum(
            bool(item["significant"])
            for item in comparisons
            if item["metric"] in config.primary_metrics
        ),
    }


def _verify_postgres(evaluations: list[dict], config: RecurrentResearchConfig) -> dict[str, int]:
    import psycopg

    from rlredteam.storage.postgres_logger import connection_string

    try:
        connection = psycopg.connect(connection_string())
    except Exception as exc:
        raise RecurrentCompletionError(f"cannot connect to PostgreSQL: {exc}") from exc
    episodes = 0
    steps = 0
    experiment_ids: set[int] = set()
    run_ids: set[int] = set()
    with connection:
        for metadata in evaluations:
            experiment_id = int(metadata["database_experiment_id"])
            run_id = int(metadata["database_evaluation_run_id"])
            run = connection.execute(
                """
                SELECT r.status, r.designation, r.evaluation_seeds,
                       e.name, e.algorithm, e.git_sha
                FROM runs r JOIN experiments e ON e.id=r.experiment_id
                WHERE r.id=%s AND e.id=%s
                """,
                (run_id, experiment_id),
            ).fetchone()
            require(run is not None, f"PostgreSQL run is absent: {run_id}")
            require(
                run[0] == "complete" and run[1] == "evaluation",
                f"PostgreSQL run is incomplete: {run_id}",
            )
            manifest = metadata["training_manifest"]
            require(
                list(run[2]) == list(config.evaluation_episode_seeds),
                f"PostgreSQL evaluation seed drift: {run_id}",
            )
            require(
                run[3] == f"{manifest['experiment_id']}-test",
                f"PostgreSQL name drift: {run_id}",
            )
            require(run[4] == manifest["algorithm"], f"PostgreSQL algorithm drift: {run_id}")
            require(run[5] == manifest["git_commit"], f"PostgreSQL commit drift: {run_id}")
            episode = connection.execute(
                """
                SELECT count(*), bool_and(goal_reached),
                       coalesce(sum(invalid_mask_selections), 0),
                       coalesce(sum(failed_actions), 0), coalesce(sum(length), 0)
                FROM episodes WHERE run_id=%s
                """,
                (run_id,),
            ).fetchone()
            step_count = connection.execute(
                """
                SELECT count(*) FROM steps s
                JOIN episodes ep ON ep.id=s.episode_id WHERE ep.run_id=%s
                """,
                (run_id,),
            ).fetchone()[0]
            require(
                episode[0] == metadata["database_episode_count"] == 60,
                f"PostgreSQL episode mismatch: {run_id}",
            )
            require(episode[1] is True, f"PostgreSQL contains a failed goal: {run_id}")
            require(
                episode[2] == episode[3] == 0,
                f"PostgreSQL contains invalid/failed actions: {run_id}",
            )
            require(
                step_count == episode[4] == metadata["database_step_count"],
                f"PostgreSQL step mismatch: {run_id}",
            )
            episodes += episode[0]
            steps += step_count
            experiment_ids.add(experiment_id)
            run_ids.add(run_id)
        require(len(experiment_ids) == len(evaluations), "PostgreSQL experiment IDs are reused")
        require(len(run_ids) == len(evaluations), "PostgreSQL run IDs are reused")
        study_experiments = connection.execute(
            "SELECT count(*) FROM experiments WHERE name LIKE %s",
            (f"{config.experiment_id}-%",),
        ).fetchone()[0]
        require(
            study_experiments == len(evaluations),
            "PostgreSQL contains an unexpected Phase 8 experiment count",
        )
    return {
        "experiments": len(evaluations),
        "runs": len(evaluations),
        "episodes": episodes,
        "steps": steps,
    }


def verify_recurrent_completion(
    repo_root: Path, *, include_postgres: bool = False
) -> dict[str, Any]:
    """Verify all canonical Phase 8 evidence without mutating it."""
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

    totals = {"runs": 0, "episodes": 0, "steps": 0, "attack_paths": 0}
    for key in sorted(expected):
        counts = _verify_run(
            repo_root,
            config,
            frozen,
            manifests_by_key[key],
            evaluations_by_key[key],
        )
        totals["runs"] += 1
        for field in ("episodes", "steps", "attack_paths"):
            totals[field] += counts[field]
    analysis = _verify_analysis(repo_root, config)
    postgres = (
        _verify_postgres(evaluations, config) if include_postgres else {"status": "not-requested"}
    )
    return {
        "complete": True,
        "study_id": config.experiment_id,
        "checks": {
            "frozen_protocol": "pass",
            "matched_training_grid": "pass",
            "frozen_evaluation": "pass",
            "zero_invalid_actions": "pass",
            "recurrent_state_lifecycle": "pass",
            "policy_immutability": "pass",
            "attack_path_reconstruction": "pass",
            "paired_analysis": "pass",
            "postgres_reconstruction": "pass" if include_postgres else "not-requested",
        },
        "evidence": totals,
        "analysis": analysis,
        "postgres": postgres,
        "runtime": {
            "training_wall_minutes": study["training_wall_seconds"] / 60,
            "projected_minutes": study["projected_canonical_minutes"],
            "cap_minutes": config.runtime_cap_minutes,
        },
    }
