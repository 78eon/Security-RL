"""Training, frozen evaluation and evidence for the Phase 8 recurrent study."""

from __future__ import annotations

import csv
import json
import math
import statistics
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from rlredteam.analyse import paired_comparison
from rlredteam.enterprise.generalisation import (
    GeneralisationError,
    dependency_lock_hash,
    git_commit,
    git_dirty,
    persist_evaluation,
    policy_digest,
    sha256_file,
    write_evaluation_package,
)
from rlredteam.enterprise.infrastructure_generalisation import (
    infrastructure_distribution_manifest,
    infrastructure_vulnerability_manifest,
)
from rlredteam.enterprise.onprem import OnPremGeneralisationSplit
from rlredteam.enterprise.profiles import (
    EnterpriseProfileConfig,
    InfrastructureCurriculumEnv,
)
from rlredteam.enterprise.recurrent import (
    PHASE8_ARMS,
    KnowledgeActionGuard,
    KnowledgeMaskedRecurrentPolicy,
    KnowledgeMaskObservationWrapper,
    RecurrentResearchConfig,
    _canonical_digest,
)
from rlredteam.train import set_all_seeds

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_FROZEN_INPUTS = REPO_ROOT / "configs" / "frozen_recurrent_policy.json"
DEFAULT_RUN_ROOT = REPO_ROOT / "runs" / "advanced-rl-recurrent-v1"
DEFAULT_RESULT_ROOT = REPO_ROOT / "results" / "advanced-rl-recurrent-v1"

ALGORITHM_NAMES = {
    "maskable_ppo": "MaskablePPO",
    "knowledge_masked_recurrent_ppo": "KnowledgeMaskedRecurrentPPO",
}


class RecurrentStudyError(GeneralisationError):
    """Raised when a Phase 8 protocol or evidence invariant is violated."""


def arm_run_name(config: RecurrentResearchConfig, arm: str, seed: int) -> str:
    if arm not in config.arms:
        raise ValueError(f"unknown Phase 8 arm: {arm}")
    return f"{config.experiment_id}-{arm}-s{int(seed)}"


def current_input_manifest(config: RecurrentResearchConfig) -> dict[str, Any]:
    """Hash every input that may affect the canonical Phase 8 experiment."""
    profile_config = EnterpriseProfileConfig.from_yaml()
    split = OnPremGeneralisationSplit()
    distribution = infrastructure_distribution_manifest(
        split, profile_config, config.train_profiles
    )
    source_files = (
        REPO_ROOT / "src" / "rlredteam" / "enterprise" / "recurrent.py",
        REPO_ROOT / "src" / "rlredteam" / "enterprise" / "recurrent_study.py",
        REPO_ROOT
        / "src"
        / "rlredteam"
        / "enterprise"
        / "infrastructure_generalisation.py",
        REPO_ROOT / "src" / "rlredteam" / "enterprise" / "environment.py",
        REPO_ROOT / "src" / "rlredteam" / "enterprise" / "state.py",
        REPO_ROOT / "scripts" / "run_recurrent_study.py",
    )
    return {
        "schema_version": 1,
        "experiment_id": config.experiment_id,
        "experiment_config_sha256": config.digest(),
        "profile_config_sha256": profile_config.digest(),
        "dependency_lock_hash": dependency_lock_hash(),
        "distribution_sha256": _canonical_digest(distribution),
        "synthetic_vulnerability_manifest_sha256": infrastructure_vulnerability_manifest(
            split, profile_config, config.train_profiles
        ),
        "source_sha256": {
            str(path.relative_to(REPO_ROOT)): sha256_file(path) for path in source_files
        },
    }


def freeze_inputs(
    output: Path,
    config: RecurrentResearchConfig | None = None,
) -> dict[str, Any]:
    """Write the prospective input manifest; requires a clean tracked tree."""
    config = config or RecurrentResearchConfig.from_yaml()
    if git_dirty() is not False:
        raise RecurrentStudyError("refusing to freeze Phase 8 inputs from a dirty tree")
    frozen = current_input_manifest(config)
    frozen.update(
        {
            "frozen_at": datetime.now(UTC).isoformat(),
            "protocol_commit": git_commit(),
        }
    )
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(frozen, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return frozen


def load_frozen_inputs(path: Path = DEFAULT_FROZEN_INPUTS) -> dict[str, Any]:
    frozen = json.loads(Path(path).read_text())
    if not isinstance(frozen, dict) or frozen.get("schema_version") != 1:
        raise RecurrentStudyError("invalid Phase 8 frozen-input manifest")
    return frozen


def validate_frozen_inputs(
    frozen: dict[str, Any], config: RecurrentResearchConfig
) -> None:
    current = current_input_manifest(config)
    expected = {key: frozen.get(key) for key in current}
    if current != expected:
        differences = [
            key for key in current if current.get(key) != expected.get(key)
        ]
        raise RecurrentStudyError(
            f"Phase 8 frozen inputs differ: {', '.join(sorted(differences))}"
        )
    if git_dirty() is not False:
        raise RecurrentStudyError("canonical Phase 8 execution requires a clean tree")


def _make_training_env(config: RecurrentResearchConfig, arm: str):
    split = OnPremGeneralisationSplit()
    base = InfrastructureCurriculumEnv(
        split.train,
        config.train_profiles,
        config=EnterpriseProfileConfig.from_yaml(),
    )
    guarded = KnowledgeActionGuard(base)
    if arm == "knowledge_masked_recurrent_ppo":
        return KnowledgeMaskObservationWrapper(guarded)
    if arm == "maskable_ppo":
        return guarded
    base.close()
    raise ValueError(f"unknown Phase 8 arm: {arm}")


def _make_model(config: RecurrentResearchConfig, arm: str, seed: int, env):
    common = dict(config.common_ppo)
    if arm == "maskable_ppo":
        from sb3_contrib import MaskablePPO

        return MaskablePPO(
            "MlpPolicy",
            env,
            seed=seed,
            device="cpu",
            verbose=0,
            policy_kwargs={
                "net_arch": list(config.baseline_policy["policy_layers"])
            },
            **common,
        )
    if arm == "knowledge_masked_recurrent_ppo":
        from sb3_contrib import RecurrentPPO

        recurrent = config.recurrent_policy
        return RecurrentPPO(
            KnowledgeMaskedRecurrentPolicy,
            env,
            seed=seed,
            device="cpu",
            verbose=0,
            policy_kwargs={
                "feature_dim": int(recurrent["feature_dim"]),
                "lstm_hidden_size": int(recurrent["lstm_hidden_size"]),
                "n_lstm_layers": int(recurrent["n_lstm_layers"]),
                "net_arch": list(recurrent["policy_layers"]),
            },
            **common,
        )
    raise ValueError(f"unknown Phase 8 arm: {arm}")


def train_arm(
    output_dir: Path,
    *,
    arm: str,
    training_seed: int,
    config: RecurrentResearchConfig | None = None,
    frozen_inputs: dict[str, Any] | None = None,
    timesteps: int | None = None,
    development: bool = False,
    allow_dirty: bool = False,
) -> dict[str, Any]:
    """Train one matched arm and emit a provenance-complete local manifest."""
    config = config or RecurrentResearchConfig.from_yaml()
    if arm not in config.arms:
        raise ValueError(f"unknown Phase 8 arm: {arm}")
    seed = int(training_seed)
    budget = int(timesteps if timesteps is not None else config.total_timesteps)
    if development:
        if seed != config.development_seed:
            raise RecurrentStudyError("development execution must use the excluded seed")
    elif seed not in config.training_seeds:
        raise RecurrentStudyError("canonical training seed is outside the protocol")
    if budget <= 0 or budget % int(config.common_ppo["n_steps"]):
        raise RecurrentStudyError("training budget must contain complete PPO rollouts")
    dirty = git_dirty()
    if dirty is not False and not (development and allow_dirty):
        raise RecurrentStudyError("training requires a clean tracked tree")
    if not development:
        if frozen_inputs is None:
            raise RecurrentStudyError("canonical training requires frozen inputs")
        validate_frozen_inputs(frozen_inputs, config)

    set_all_seeds(seed)
    env = _make_training_env(config, arm)
    model = _make_model(config, arm, seed, env)
    started = time.monotonic()
    model.learn(total_timesteps=budget)
    elapsed = time.monotonic() - started
    if int(model.num_timesteps) != budget:
        raise RecurrentStudyError(
            f"actual training steps {model.num_timesteps} differ from {budget}"
        )
    if not all(
        bool(np.isfinite(parameter.detach().cpu().numpy()).all())
        for parameter in model.policy.parameters()
    ):
        raise RecurrentStudyError("trained policy contains non-finite parameters")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / "model"
    model.save(checkpoint)
    checkpoint = checkpoint.with_suffix(".zip")
    profile_config = EnterpriseProfileConfig.from_yaml()
    split = OnPremGeneralisationSplit()
    run_name = arm_run_name(config, arm, seed)
    manifest = {
        "schema_version": 1,
        "study_id": config.experiment_id,
        "experiment_id": run_name,
        "description": config.description,
        "arm": arm,
        "algorithm": ALGORITHM_NAMES[arm],
        "policy": (
            "MlpPolicy" if arm == "maskable_ppo" else "KnowledgeMaskedRecurrentPolicy"
        ),
        "action_mask_source": "AgentKnowledge",
        "action_mask_transport": (
            "MaskablePPO.action_masks"
            if arm == "maskable_ppo"
            else "observation_constraint_excluded_from_features"
        ),
        "recurrent_state_reset_contract": (
            "not_applicable"
            if arm == "maskable_ppo"
            else "state=None and episode_start=True for every evaluation episode"
        ),
        "development": bool(development),
        "training_seed": seed,
        "training_timesteps": budget,
        "actual_training_timesteps": int(model.num_timesteps),
        "training_elapsed_seconds": elapsed,
        "train_profiles": [profile.value for profile in config.train_profiles],
        "train_topology_seeds": list(split.train),
        "validation_topology_seeds": list(split.validation),
        "test_topology_seeds": list(split.test),
        "study_config_hash": config.digest(),
        "experiment_config_hash": config.digest(),
        "profile_config_hash": profile_config.digest(),
        "topology_config_hash": profile_config.digest(),
        "dependency_lock_hash": dependency_lock_hash(),
        "git_commit": git_commit(),
        "git_dirty": dirty,
        "ppo": config.common_ppo,
        "policy_config": (
            config.baseline_policy
            if arm == "maskable_ppo"
            else config.recurrent_policy
        ),
        "parameter_count": sum(
            parameter.numel() for parameter in model.policy.parameters()
        ),
        "distribution": infrastructure_distribution_manifest(
            split, profile_config, config.train_profiles
        ),
        "synthetic_vulnerability_manifest_sha256": infrastructure_vulnerability_manifest(
            split, profile_config, config.train_profiles
        ),
        "frozen_inputs_sha256": (
            _canonical_digest(frozen_inputs) if frozen_inputs is not None else None
        ),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "policy_sha256": policy_digest(model),
        "weights_release": "gated; runs/ is gitignored",
    }
    (output_dir / "training_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    env.close()
    return manifest


def validate_training_manifest(
    manifest: dict[str, Any],
    checkpoint: Path,
    *,
    arm: str,
    training_seed: int,
    config: RecurrentResearchConfig,
    frozen_inputs: dict[str, Any] | None,
    development: bool = False,
    allow_unverifiable: bool = False,
) -> None:
    split = OnPremGeneralisationSplit()
    profile_config = EnterpriseProfileConfig.from_yaml()
    checks = {
        "study": (manifest.get("study_id"), config.experiment_id),
        "run": (
            manifest.get("experiment_id"),
            arm_run_name(config, arm, training_seed),
        ),
        "arm": (manifest.get("arm"), arm),
        "algorithm": (manifest.get("algorithm"), ALGORITHM_NAMES[arm]),
        "training seed": (manifest.get("training_seed"), int(training_seed)),
        "study config": (manifest.get("study_config_hash"), config.digest()),
        "profile config": (manifest.get("profile_config_hash"), profile_config.digest()),
        "dependency lock": (manifest.get("dependency_lock_hash"), dependency_lock_hash()),
        "training split": (manifest.get("train_topology_seeds"), list(split.train)),
        "validation split": (
            manifest.get("validation_topology_seeds"),
            list(split.validation),
        ),
        "test split": (manifest.get("test_topology_seeds"), list(split.test)),
        "checkpoint": (manifest.get("checkpoint_sha256"), sha256_file(checkpoint)),
        "development": (manifest.get("development"), bool(development)),
    }
    if frozen_inputs is not None:
        checks["frozen inputs"] = (
            manifest.get("frozen_inputs_sha256"),
            _canonical_digest(frozen_inputs),
        )
    failures = [
        f"{name}: manifest={actual!r}, runtime={expected!r}"
        for name, (actual, expected) in checks.items()
        if actual != expected
    ]
    if manifest.get("git_dirty") is not False and not allow_unverifiable:
        failures.append("checkpoint was produced from a dirty/unknown tree")
    if not development and git_dirty() is not False and not allow_unverifiable:
        failures.append("canonical evaluation tree is dirty/unknown")
    if failures:
        raise RecurrentStudyError(
            "Phase 8 checkpoint provenance mismatch:\n  " + "\n  ".join(failures)
        )


def load_arm_model(arm: str, checkpoint: Path):
    if arm == "maskable_ppo":
        from sb3_contrib import MaskablePPO

        return MaskablePPO.load(checkpoint, device="cpu")
    if arm == "knowledge_masked_recurrent_ppo":
        from sb3_contrib import RecurrentPPO

        return RecurrentPPO.load(checkpoint, device="cpu")
    raise ValueError(f"unknown Phase 8 arm: {arm}")


def evaluate_arm(
    model,
    *,
    arm: str,
    training_seed: int,
    profiles,
    topology_seeds: tuple[int, ...],
    evaluation_episode_seeds: tuple[int, ...],
    deterministic: bool = True,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Evaluate one frozen arm with explicit recurrent state lifecycle."""
    if arm not in PHASE8_ARMS:
        raise ValueError(f"unknown Phase 8 arm: {arm}")
    split = OnPremGeneralisationSplit()
    if not profiles or not topology_seeds or not evaluation_episode_seeds:
        raise RecurrentStudyError("evaluation profiles/topologies/seeds are required")
    if set(topology_seeds) & set(split.train):
        raise RecurrentStudyError("evaluation topology seeds overlap the training split")

    profile_config = EnterpriseProfileConfig.from_yaml()
    episodes: list[dict[str, Any]] = []
    steps: list[dict[str, Any]] = []
    before = policy_digest(model)
    state_reset_count = 0
    recurrent_predictions = 0
    mask_transport_checks = 0
    for profile in profiles:
        for topology_seed in topology_seeds:
            base = InfrastructureCurriculumEnv(
                (topology_seed,), (profile,), config=profile_config
            )
            guard = KnowledgeActionGuard(base)
            env = (
                KnowledgeMaskObservationWrapper(guard)
                if arm == "knowledge_masked_recurrent_ppo"
                else guard
            )
            for evaluation_seed in evaluation_episode_seeds:
                set_all_seeds(evaluation_seed)
                model.set_random_seed(evaluation_seed)
                observation, reset_info = env.reset(
                    seed=evaluation_seed,
                    options={
                        "profile": profile.value,
                        "topology_seed": topology_seed,
                    },
                )
                recurrent_state = None
                episode_start = np.ones((1,), dtype=bool)
                if arm == "knowledge_masked_recurrent_ppo":
                    state_reset_count += 1
                terminated = truncated = False
                total_reward = 0.0
                failed_actions = 0
                episode_steps = 0
                while not (terminated or truncated):
                    mask = guard.action_masks().copy()
                    if not mask.any():
                        raise RecurrentStudyError(
                            "no valid AgentKnowledge action before termination"
                        )
                    if arm == "maskable_ppo":
                        action, _ = model.predict(
                            observation,
                            action_masks=mask,
                            deterministic=deterministic,
                        )
                    else:
                        _, transported = env.split_observation(observation)
                        if not np.array_equal(transported.astype(bool), mask):
                            raise RecurrentStudyError(
                                "transported recurrent mask differs from AgentKnowledge"
                            )
                        mask_transport_checks += 1
                        action, recurrent_state = model.predict(
                            observation,
                            state=recurrent_state,
                            episode_start=episode_start,
                            deterministic=deterministic,
                        )
                        recurrent_predictions += 1
                    selected = int(np.asarray(action).item())
                    if not mask[selected]:
                        raise RecurrentStudyError("policy selected a masked action")
                    observation, reward, terminated, truncated, info = env.step(selected)
                    episode_start = np.asarray([terminated or truncated], dtype=bool)
                    event = info["event"]
                    vulnerability = base.true_topology.vulnerabilities.get(
                        event.action.target
                    )
                    total_reward += float(reward)
                    failed_actions += int(not event.success)
                    steps.append(
                        {
                            "arm": arm,
                            "training_seed": int(training_seed),
                            "profile": profile.value,
                            "topology_seed": int(topology_seed),
                            "topology_hash": reset_info["topology_hash"],
                            "evaluation_seed": int(evaluation_seed),
                            "step": event.step,
                            "action": event.action.name,
                            "action_kind": event.action.type.value,
                            "target_entity": event.action.target,
                            "success": event.success,
                            "state_changed": event.state_changed,
                            "reward": event.reward,
                            "prerequisites": list(event.prerequisites),
                            "outcomes": list(event.outcomes),
                            "reason": event.reason,
                            "goal_reached": event.goal_reached,
                            "cve_id": vulnerability.id if vulnerability else None,
                            "cvss_base": vulnerability.cvss if vulnerability else None,
                        }
                    )
                    episode_steps += 1
                known_nodes = len(base.knowledge.discovered)
                episodes.append(
                    {
                        "arm": arm,
                        "training_seed": int(training_seed),
                        "profile": profile.value,
                        "topology_seed": int(topology_seed),
                        "topology_hash": reset_info["topology_hash"],
                        "evaluation_seed": int(evaluation_seed),
                        "goal_reached": bool(terminated and not truncated),
                        "terminal_reason": "goal" if terminated else "step_limit",
                        "steps_to_goal": episode_steps if terminated else None,
                        "episode_length": episode_steps,
                        "total_reward": total_reward,
                        "known_nodes": known_nodes,
                        "true_nodes": len(base.true_topology.nodes),
                        "discovery_coverage": known_nodes / len(base.true_topology.nodes),
                        "invalid_mask_selections": guard.invalid_action_selections,
                        "failed_actions": failed_actions,
                        "hosts_compromised": len(base.knowledge.access),
                        "path_events": len(base.attack_path()),
                    }
                )
            env.close()
    after = policy_digest(model)
    if before != after:
        raise RecurrentStudyError("policy parameters changed during frozen evaluation")
    if any(row["invalid_mask_selections"] for row in episodes):
        raise RecurrentStudyError("evaluation recorded an invalid masked selection")
    integrity = {
        "gradient_updates": False,
        "policy_sha256_before": before,
        "policy_sha256_after": after,
        "state_reset_count": state_reset_count,
        "recurrent_predictions": recurrent_predictions,
        "mask_transport_checks": mask_transport_checks,
        "invalid_mask_selections": 0,
    }
    return episodes, steps, integrity


def persist_and_write_run_evaluation(
    output_dir: Path,
    *,
    episodes: list[dict[str, Any]],
    steps: list[dict[str, Any]],
    integrity: dict[str, Any],
    training_manifest: dict[str, Any],
    checkpoint: Path,
    split_name: str,
    postgres: bool,
) -> dict[str, Any]:
    metadata = {
        "phase": "phase8_frozen_recurrent_comparison",
        "split": split_name,
        "gradient_updates": False,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "training_manifest": training_manifest,
        **integrity,
    }
    if postgres:
        metadata.update(
            persist_evaluation(
                episodes=episodes,
                steps=steps,
                training_manifest=training_manifest,
                checkpoint=checkpoint,
                split_name=split_name,
            )
        )
    write_evaluation_package(
        output_dir,
        episodes=episodes,
        steps=steps,
        metadata=metadata,
    )
    return metadata


def aggregate_seed_metrics(
    episodes: list[dict[str, Any]],
    config: RecurrentResearchConfig,
    *,
    expected_topology_seeds: tuple[int, ...],
) -> list[dict[str, Any]]:
    expected_count = (
        len(config.train_profiles)
        * len(expected_topology_seeds)
        * len(config.evaluation_episode_seeds)
    )
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for episode in episodes:
        key = (str(episode["arm"]), int(episode["training_seed"]))
        grouped.setdefault(key, []).append(episode)

    rows: list[dict[str, Any]] = []
    for (arm, seed), values in sorted(grouped.items()):
        if len(values) != expected_count:
            raise RecurrentStudyError(
                f"{arm} seed {seed} has {len(values)} episodes; expected {expected_count}"
            )
        episode_keys = {
            (
                row["profile"],
                int(row["topology_seed"]),
                int(row["evaluation_seed"]),
            )
            for row in values
        }
        if len(episode_keys) != expected_count:
            raise RecurrentStudyError(f"{arm} seed {seed} has duplicate evaluation cases")
        penalized = [
            int(row["steps_to_goal"])
            if row["goal_reached"]
            else config.failure_step_penalty
            for row in values
        ]
        rows.append(
            {
                "arm": arm,
                "training_seed": seed,
                "episode_count": len(values),
                "success_rate": statistics.fmean(
                    float(row["goal_reached"]) for row in values
                ),
                "penalized_steps": statistics.fmean(penalized),
                "total_reward": statistics.fmean(
                    float(row["total_reward"]) for row in values
                ),
                "discovery_coverage": statistics.fmean(
                    float(row["discovery_coverage"]) for row in values
                ),
                "failed_actions": statistics.fmean(
                    float(row["failed_actions"]) for row in values
                ),
            }
        )
    return rows


def analyse_seed_metrics(
    rows: list[dict[str, Any]],
    config: RecurrentResearchConfig,
    *,
    expected_seeds: tuple[int, ...] | None = None,
) -> dict[str, Any]:
    expected_seeds = expected_seeds or config.training_seeds
    by_arm = {
        arm: {
            int(row["training_seed"]): row
            for row in rows
            if row["arm"] == arm
        }
        for arm in config.arms
    }
    complete = all(set(by_arm[arm]) == set(expected_seeds) for arm in config.arms)
    protocol = {
        "primary_metrics": list(config.primary_metrics),
        "statistics": config.statistics,
    }
    metrics = (*config.primary_metrics, *config.descriptive_metrics)
    comparisons = []
    for metric in metrics:
        baseline = {
            seed: float(row[metric]) for seed, row in by_arm["maskable_ppo"].items()
        }
        recurrent = {
            seed: float(row[metric])
            for seed, row in by_arm["knowledge_masked_recurrent_ppo"].items()
        }
        comparisons.append(
            paired_comparison(metric, baseline, recurrent, protocol).to_dict()
        )
    report = {
        "schema_version": 1,
        "study_id": config.experiment_id,
        "complete": complete,
        "primary_unit": "training_seed",
        "arm_a": "maskable_ppo",
        "arm_b": "knowledge_masked_recurrent_ppo",
        "expected_training_seeds": list(expected_seeds),
        "observed_training_seeds": {
            arm: sorted(by_arm[arm]) for arm in config.arms
        },
        "primary_metrics": list(config.primary_metrics),
        "comparisons": comparisons,
    }
    json.dumps(report, allow_nan=False)
    return report


def write_study_summary(
    output_dir: Path,
    *,
    seed_metrics: list[dict[str, Any]],
    report: dict[str, Any],
    metadata: dict[str, Any],
) -> None:
    output_dir = Path(output_dir)
    for folder in ("metadata", "summaries", "tables"):
        (output_dir / folder).mkdir(parents=True, exist_ok=True)
    (output_dir / "metadata" / "study.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    (output_dir / "summaries" / "analysis.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    if seed_metrics:
        with (output_dir / "summaries" / "seed_metrics.csv").open(
            "w", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(seed_metrics[0]))
            writer.writeheader()
            writer.writerows(seed_metrics)
    comparisons = report["comparisons"]
    if comparisons:
        with (output_dir / "tables" / "statistics.csv").open(
            "w", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(comparisons[0]))
            writer.writeheader()
            writer.writerows(comparisons)


def validate_finite_evidence(
    episodes: list[dict[str, Any]], steps: list[dict[str, Any]]
) -> None:
    for collection in (episodes, steps):
        for row in collection:
            for key, value in row.items():
                if isinstance(value, float) and not math.isfinite(value):
                    raise RecurrentStudyError(f"non-finite evidence value at {key}")
