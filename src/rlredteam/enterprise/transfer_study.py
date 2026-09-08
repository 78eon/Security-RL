"""Training, frozen evaluation and evidence for the Phase 11 transfer study."""

from __future__ import annotations

import json
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np

from rlredteam.enterprise.curriculum import canonical_digest
from rlredteam.enterprise.curriculum_study import (
    aggregate_seed_metrics,
    analyse_seed_metrics,
    validate_finite_evidence,
    write_study_summary,
)
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
from rlredteam.enterprise.graph_policy import AgentKnowledgeGraphExtractor
from rlredteam.enterprise.profiles import EnterpriseProfileConfig, InfrastructureCurriculumEnv
from rlredteam.enterprise.recurrent import KnowledgeActionGuard
from rlredteam.enterprise.transfer_learning import (
    PHASE11_ARMS,
    TransferResearchConfig,
    observation_schema,
    transfer_distribution_manifest,
    transfer_vulnerability_manifest,
)
from rlredteam.train import set_all_seeds

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_FROZEN_INPUTS = REPO_ROOT / "configs/frozen_transfer_learning.json"
DEFAULT_RUN_ROOT = REPO_ROOT / "runs/advanced-rl-transfer-v1"
DEFAULT_RESULT_ROOT = REPO_ROOT / "results/advanced-rl-transfer-v1"


class TransferStudyError(GeneralisationError):
    """Raised when a Phase 11 protocol or evidence invariant is violated."""


def arm_run_name(config: TransferResearchConfig, arm: str, seed: int) -> str:
    if arm not in config.arms:
        raise ValueError(f"unknown Phase 11 arm: {arm}")
    return f"{config.experiment_id}-{arm}-s{int(seed)}"


def _graph_policy_kwargs(config: TransferResearchConfig) -> dict[str, Any]:
    profile = EnterpriseProfileConfig.from_yaml()
    graph = config.graph_policy
    return {
        "features_extractor_class": AgentKnowledgeGraphExtractor,
        "features_extractor_kwargs": {
            "max_nodes": profile.max_nodes,
            "node_hidden_dim": int(graph["node_hidden_dim"]),
            "message_passing_steps": int(graph["message_passing_steps"]),
            "feature_dim": int(graph["feature_dim"]),
        },
        "net_arch": list(graph["policy_layers"]),
    }


def current_input_manifest(config: TransferResearchConfig) -> dict[str, Any]:
    base = EnterpriseProfileConfig.from_yaml()
    distribution = transfer_distribution_manifest(config, base)
    source_files = (
        REPO_ROOT / "src/rlredteam/analyse.py",
        REPO_ROOT / "src/rlredteam/train.py",
        REPO_ROOT / "src/rlredteam/enterprise/curriculum.py",
        REPO_ROOT / "src/rlredteam/enterprise/curriculum_study.py",
        REPO_ROOT / "src/rlredteam/enterprise/environment.py",
        REPO_ROOT / "src/rlredteam/enterprise/generalisation.py",
        REPO_ROOT / "src/rlredteam/enterprise/graph_policy.py",
        REPO_ROOT / "src/rlredteam/enterprise/model.py",
        REPO_ROOT / "src/rlredteam/enterprise/onprem.py",
        REPO_ROOT / "src/rlredteam/enterprise/profiles.py",
        REPO_ROOT / "src/rlredteam/enterprise/recurrent.py",
        REPO_ROOT / "src/rlredteam/enterprise/state.py",
        REPO_ROOT / "src/rlredteam/enterprise/transfer_learning.py",
        REPO_ROOT / "src/rlredteam/enterprise/transfer_study.py",
        REPO_ROOT / "src/rlredteam/storage/postgres_logger.py",
        REPO_ROOT / "src/rlredteam/storage/schema.sql",
        REPO_ROOT / "scripts/run_transfer_study.py",
    )
    return {
        "schema_version": 1,
        "experiment_id": config.experiment_id,
        "experiment_config_sha256": config.digest(),
        "base_profile_config_sha256": base.digest(),
        "dependency_lock_hash": dependency_lock_hash(),
        "distribution_sha256": canonical_digest(distribution),
        "vulnerability_snapshot_sha256": transfer_vulnerability_manifest(config, base),
        "observation_schema": observation_schema(),
        "source_sha256": {
            str(path.relative_to(REPO_ROOT)): sha256_file(path) for path in source_files
        },
    }


def freeze_inputs(output: Path, config: TransferResearchConfig | None = None) -> dict[str, Any]:
    config = config or TransferResearchConfig.from_yaml()
    if git_dirty() is not False:
        raise TransferStudyError("refusing to freeze Phase 11 inputs from a dirty tree")
    frozen = current_input_manifest(config)
    frozen.update({"frozen_at": datetime.now(UTC).isoformat(), "protocol_commit": git_commit()})
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(frozen, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return frozen


def load_frozen_inputs(path: Path = DEFAULT_FROZEN_INPUTS) -> dict[str, Any]:
    try:
        frozen = json.loads(Path(path).read_text())
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise TransferStudyError(f"cannot load frozen Phase 11 inputs: {exc}") from exc
    if not isinstance(frozen, dict) or frozen.get("schema_version") != 1:
        raise TransferStudyError("frozen Phase 11 input manifest is invalid")
    return frozen


def validate_frozen_inputs(frozen: dict[str, Any], config: TransferResearchConfig) -> None:
    current = current_input_manifest(config)
    mismatches = [key for key, value in current.items() if frozen.get(key) != value]
    if mismatches:
        raise TransferStudyError("frozen Phase 11 inputs drifted: " + ", ".join(mismatches))
    if not frozen.get("protocol_commit"):
        raise TransferStudyError("frozen Phase 11 protocol has no Git commit")


class TrainingDistributionAudit(gym.Wrapper):
    """Record actual source or target cases selected at each reset."""

    def __init__(self, env: InfrastructureCurriculumEnv) -> None:
        super().__init__(env)
        self.records: list[dict[str, Any]] = []

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        observation, info = self.env.reset(seed=seed, options=options)
        self.records.append(
            {
                "profile": str(info["profile"]),
                "topology_seed": int(info["topology_seed"]),
                "topology_hash": str(info["topology_hash"]),
                "profile_config_hash": str(info["profile_config_hash"]),
            }
        )
        return observation, info

    def action_masks(self):
        return self.env.action_masks()


def _make_environment(
    topology_seeds: tuple[int, ...], profiles: tuple, base: EnterpriseProfileConfig
) -> tuple[TrainingDistributionAudit, KnowledgeActionGuard]:
    audit = TrainingDistributionAudit(
        InfrastructureCurriculumEnv(topology_seeds, profiles, config=base)
    )
    return audit, KnowledgeActionGuard(audit)


def _make_model(config: TransferResearchConfig, seed: int, env):
    from sb3_contrib import MaskablePPO

    return MaskablePPO(
        "MlpPolicy",
        env,
        seed=seed,
        device="cpu",
        verbose=0,
        policy_kwargs=_graph_policy_kwargs(config),
        **config.common_ppo,
    )


def _verify_training_records(
    records: list[dict[str, Any]],
    expected: dict[str, dict[str, str]],
    profile_hash: str,
    *,
    label: str,
) -> None:
    if not records:
        raise TransferStudyError(f"training contains no observed resets: {label}")
    for record in records:
        profile = record["profile"]
        topology_seed = str(record["topology_seed"])
        if profile not in expected or topology_seed not in expected[profile]:
            raise TransferStudyError(f"training selected an undeclared case: {label}")
        if record["topology_hash"] != expected[profile][topology_seed]:
            raise TransferStudyError(f"training topology hash drift: {label}")
        if record["profile_config_hash"] != profile_hash:
            raise TransferStudyError(f"training profile hash drift: {label}")


def _audit_payload(records: list[dict[str, Any]], environment_seed: int) -> dict[str, Any]:
    exposures = Counter(f"{item['profile']}:{item['topology_seed']}" for item in records)
    profiles = Counter(item["profile"] for item in records)
    return {
        "environment_seed": environment_seed,
        "reset_count": len(records),
        "observed_profile_counts": dict(sorted(profiles.items())),
        "exposure_counts": dict(sorted(exposures.items())),
        "reset_trace_sha256": canonical_digest(records),
    }


def _learn_exact(model, budget: int) -> float:
    started = time.monotonic()
    model.learn(total_timesteps=budget, reset_num_timesteps=True)
    elapsed = time.monotonic() - started
    if int(model.num_timesteps) != budget:
        raise TransferStudyError(
            f"actual adaptation steps {model.num_timesteps} differ from {budget}"
        )
    if not all(
        bool(np.isfinite(parameter.detach().cpu().numpy()).all())
        for parameter in model.policy.parameters()
    ):
        raise TransferStudyError("trained policy contains non-finite parameters")
    return elapsed


def train_arm(
    output_dir: Path,
    *,
    arm: str,
    training_seed: int,
    config: TransferResearchConfig | None = None,
    frozen_inputs: dict[str, Any] | None = None,
    source_timesteps: int | None = None,
    target_timesteps: int | None = None,
    development: bool = False,
    allow_dirty: bool = False,
) -> dict[str, Any]:
    """Train one architecture-matched target policy from scratch or source weights."""
    config = config or TransferResearchConfig.from_yaml()
    if arm not in config.arms:
        raise ValueError(f"unknown Phase 11 arm: {arm}")
    seed = int(training_seed)
    source_budget = int(source_timesteps or config.source_pretrain_timesteps)
    target_budget = int(target_timesteps or config.target_adaptation_timesteps)
    n_steps = int(config.common_ppo["n_steps"])
    if target_budget <= 0 or target_budget % n_steps:
        raise TransferStudyError("target budget must contain complete PPO rollouts")
    if source_budget <= 0 or source_budget % n_steps:
        raise TransferStudyError("source budget must contain complete PPO rollouts")
    if development:
        if seed != config.development_seed:
            raise TransferStudyError("development execution must use the excluded seed")
    elif seed not in config.training_seeds:
        raise TransferStudyError("canonical training seed is outside the protocol")
    dirty = git_dirty()
    if dirty is not False and not (development and allow_dirty):
        raise TransferStudyError("training requires a clean tracked tree")
    if not development:
        if frozen_inputs is None:
            raise TransferStudyError("canonical training requires frozen inputs")
        validate_frozen_inputs(frozen_inputs, config)

    import torch

    torch.set_num_threads(1)
    set_all_seeds(seed)
    base = EnterpriseProfileConfig.from_yaml()
    distribution = transfer_distribution_manifest(config, base)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    source_evidence: dict[str, Any] | None = None

    if arm == "transfer_legacy_cloud_to_hybrid":
        source_audit, source_env = _make_environment(
            config.topology_splits["source_train"], config.source_profiles, base
        )
        source_environment_seed = seed * 100 + 1
        source_env.reset(seed=source_environment_seed)
        source_audit.records.clear()
        model = _make_model(config, seed, source_env)
        source_initial_policy_sha256 = policy_digest(model)
        source_elapsed = _learn_exact(model, source_budget)
        if source_env.invalid_action_selections:
            raise TransferStudyError(f"invalid source action selected: {arm}/{seed}")
        _verify_training_records(
            source_audit.records,
            distribution["source_train"],
            base.digest(),
            label=f"{arm}/{seed}/source",
        )
        source_checkpoint = output_dir / "source_model"
        model.save(source_checkpoint)
        source_checkpoint = source_checkpoint.with_suffix(".zip")
        source_policy_sha256 = policy_digest(model)
        if source_policy_sha256 == source_initial_policy_sha256:
            raise TransferStudyError("source policy did not change during pretraining")
        source_evidence = {
            "profiles": [item.value for item in config.source_profiles],
            "topology_seeds": list(config.topology_splits["source_train"]),
            "training_timesteps": source_budget,
            "actual_training_timesteps": int(model.num_timesteps),
            "training_elapsed_seconds": source_elapsed,
            "initial_policy_sha256": source_initial_policy_sha256,
            "checkpoint": str(source_checkpoint),
            "checkpoint_sha256": sha256_file(source_checkpoint),
            "policy_sha256": source_policy_sha256,
            **_audit_payload(source_audit.records, source_environment_seed),
        }
        source_env.close()
    else:
        source_checkpoint = None
        source_policy_sha256 = None

    target_audit, target_env = _make_environment(
        config.topology_splits["target_train"], config.target_profiles, base
    )
    target_environment_seed = seed * 100 + 2
    target_env.reset(seed=target_environment_seed)
    target_audit.records.clear()
    set_all_seeds(seed)
    if source_checkpoint is None:
        model = _make_model(config, seed, target_env)
        initialization_type = "random"
    else:
        from sb3_contrib import MaskablePPO

        model = MaskablePPO.load(source_checkpoint, env=target_env, device="cpu")
        model.set_random_seed(seed)
        initialization_type = "legacy_cloud_source_checkpoint"
    pre_adaptation_policy_sha256 = policy_digest(model)
    if source_policy_sha256 is not None and pre_adaptation_policy_sha256 != source_policy_sha256:
        raise TransferStudyError("loaded transfer policy differs from its source checkpoint")
    target_elapsed = _learn_exact(model, target_budget)
    if target_env.invalid_action_selections:
        raise TransferStudyError(f"invalid target action selected: {arm}/{seed}")
    _verify_training_records(
        target_audit.records,
        distribution["target_train"],
        base.digest(),
        label=f"{arm}/{seed}/target",
    )
    checkpoint = output_dir / "model"
    final_policy_sha256 = policy_digest(model)
    if final_policy_sha256 == pre_adaptation_policy_sha256:
        raise TransferStudyError("target policy did not change during adaptation")
    model.save(checkpoint)
    checkpoint = checkpoint.with_suffix(".zip")
    target_audit_payload = _audit_payload(target_audit.records, target_environment_seed)
    manifest = {
        "schema_version": 1,
        "study_id": config.experiment_id,
        "experiment_id": arm_run_name(config, arm, seed),
        "description": config.description,
        "arm": arm,
        "algorithm": "MaskablePPO",
        "policy": "MlpPolicy",
        "representation": "agent_knowledge_message_passing_graph",
        "action_mask_source": "AgentKnowledge",
        "observation_source": "AgentKnowledge",
        "observation_schema": observation_schema(),
        "development": bool(development),
        "training_seed": seed,
        "initialization_type": initialization_type,
        "pre_adaptation_policy_sha256": pre_adaptation_policy_sha256,
        "source_pretraining": source_evidence,
        "target_adaptation_timesteps": target_budget,
        "actual_target_adaptation_timesteps": int(model.num_timesteps),
        "target_training_elapsed_seconds": target_elapsed,
        "target_profiles": [item.value for item in config.target_profiles],
        "source_topology_seeds": list(config.topology_splits["source_train"]),
        "target_train_topology_seeds": list(config.topology_splits["target_train"]),
        "validation_topology_seeds": list(config.topology_splits["validation"]),
        "test_topology_seeds": list(config.topology_splits["test"]),
        "study_config_hash": config.digest(),
        "scientific_config_hash": config.scientific_digest(),
        "base_profile_config_hash": base.digest(),
        "dependency_lock_hash": dependency_lock_hash(),
        "git_commit": git_commit(),
        "git_dirty": dirty,
        "ppo": config.common_ppo,
        "policy_config": config.graph_policy,
        "parameter_count": sum(parameter.numel() for parameter in model.policy.parameters()),
        "torch_num_threads": torch.get_num_threads(),
        "target_training_audit": target_audit_payload,
        "base_distribution": distribution,
        "base_vulnerability_snapshot_sha256": transfer_vulnerability_manifest(config, base),
        "frozen_inputs_sha256": (
            canonical_digest(frozen_inputs) if frozen_inputs is not None else None
        ),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "policy_sha256": final_policy_sha256,
        "weights_release": "gated; runs/ is gitignored",
    }
    (output_dir / "training_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    target_env.close()
    return manifest


def validate_training_manifest(
    manifest: dict[str, Any],
    checkpoint: Path,
    *,
    arm: str,
    training_seed: int,
    config: TransferResearchConfig,
    frozen_inputs: dict[str, Any] | None,
    development: bool = False,
    allow_unverifiable: bool = False,
) -> None:
    expected = {
        "study": (manifest.get("study_id"), config.experiment_id),
        "run": (manifest.get("experiment_id"), arm_run_name(config, arm, training_seed)),
        "arm": (manifest.get("arm"), arm),
        "algorithm": (manifest.get("algorithm"), "MaskablePPO"),
        "observation source": (manifest.get("observation_source"), "AgentKnowledge"),
        "action mask source": (manifest.get("action_mask_source"), "AgentKnowledge"),
        "training seed": (manifest.get("training_seed"), int(training_seed)),
        "target budget": (
            manifest.get("actual_target_adaptation_timesteps"),
            manifest.get("target_adaptation_timesteps"),
        ),
        "study config": (manifest.get("study_config_hash"), config.digest()),
        "development": (manifest.get("development"), bool(development)),
        "checkpoint": (manifest.get("checkpoint_sha256"), sha256_file(checkpoint)),
        "observation schema": (manifest.get("observation_schema"), observation_schema()),
        "policy config": (manifest.get("policy_config"), config.graph_policy),
    }
    if frozen_inputs is not None:
        expected["frozen inputs"] = (
            manifest.get("frozen_inputs_sha256"),
            canonical_digest(frozen_inputs),
        )
    failures = [
        f"{name}: manifest={actual!r}, runtime={required!r}"
        for name, (actual, required) in expected.items()
        if actual != required
    ]
    if manifest.get("git_dirty") is not False and not allow_unverifiable:
        failures.append("checkpoint was produced from a dirty/unknown tree")
    if not development and git_dirty() is not False and not allow_unverifiable:
        failures.append("canonical evaluation tree is dirty/unknown")
    if failures:
        raise TransferStudyError(
            "Phase 11 checkpoint provenance mismatch:\n  " + "\n  ".join(failures)
        )


def validate_paired_target_isolation(scratch: dict[str, Any], transfer: dict[str, Any]) -> None:
    """Prove that source initialization is the only designed target-phase difference."""
    controlled_fields = (
        "study_id",
        "description",
        "algorithm",
        "policy",
        "representation",
        "action_mask_source",
        "observation_source",
        "observation_schema",
        "development",
        "training_seed",
        "target_adaptation_timesteps",
        "actual_target_adaptation_timesteps",
        "target_profiles",
        "source_topology_seeds",
        "target_train_topology_seeds",
        "validation_topology_seeds",
        "test_topology_seeds",
        "study_config_hash",
        "scientific_config_hash",
        "base_profile_config_hash",
        "dependency_lock_hash",
        "git_commit",
        "git_dirty",
        "ppo",
        "policy_config",
        "parameter_count",
        "torch_num_threads",
        "base_distribution",
        "base_vulnerability_snapshot_sha256",
        "frozen_inputs_sha256",
    )
    differences = [
        field for field in controlled_fields if scratch.get(field) != transfer.get(field)
    ]
    if differences:
        raise TransferStudyError(
            "paired target arms differ outside initialization: " + ", ".join(differences)
        )
    if (
        scratch.get("initialization_type") != "random"
        or scratch.get("source_pretraining") is not None
    ):
        raise TransferStudyError("scratch arm contains source initialization")
    source = transfer.get("source_pretraining")
    if transfer.get("initialization_type") != "legacy_cloud_source_checkpoint" or not source:
        raise TransferStudyError("transfer arm lacks source initialization")
    if transfer.get("pre_adaptation_policy_sha256") != source.get("policy_sha256"):
        raise TransferStudyError("transfer arm did not load its declared source policy")


def load_model(checkpoint: Path):
    from sb3_contrib import MaskablePPO

    return MaskablePPO.load(checkpoint, device="cpu")


def evaluate_arm(
    model,
    *,
    arm: str,
    training_seed: int,
    config: TransferResearchConfig,
    topology_seeds: tuple[int, ...],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Evaluate a frozen adapted policy on explicit unseen hybrid topologies."""
    if arm not in PHASE11_ARMS:
        raise ValueError(f"unknown Phase 11 arm: {arm}")
    train_seeds = set(config.topology_splits["source_train"]) | set(
        config.topology_splits["target_train"]
    )
    if not topology_seeds or set(topology_seeds) & train_seeds:
        raise TransferStudyError("evaluation topology seeds are absent or overlap training")
    base_config = EnterpriseProfileConfig.from_yaml()
    episodes: list[dict[str, Any]] = []
    steps: list[dict[str, Any]] = []
    before = policy_digest(model)
    mask_checks = reset_count = 0
    for profile in config.target_profiles:
        for topology_seed in topology_seeds:
            base = InfrastructureCurriculumEnv((topology_seed,), (profile,), config=base_config)
            env = KnowledgeActionGuard(base)
            for evaluation_seed in config.evaluation_episode_seeds:
                set_all_seeds(evaluation_seed)
                model.set_random_seed(evaluation_seed)
                observation, reset_info = env.reset(
                    seed=evaluation_seed,
                    options={"profile": profile.value, "topology_seed": topology_seed},
                )
                reset_count += 1
                terminated = truncated = False
                total_reward = 0.0
                failed_actions = episode_steps = 0
                while not (terminated or truncated):
                    mask = env.action_masks().copy()
                    if not mask.any():
                        raise TransferStudyError(
                            "no valid AgentKnowledge action before termination"
                        )
                    action, _ = model.predict(
                        observation,
                        action_masks=mask,
                        deterministic=config.deterministic_evaluation,
                    )
                    selected = int(np.asarray(action).item())
                    mask_checks += 1
                    if not mask[selected]:
                        raise TransferStudyError("policy selected a masked action")
                    observation, reward, terminated, truncated, info = env.step(selected)
                    event = info["event"]
                    vulnerability = base.true_topology.vulnerabilities.get(event.action.target)
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
                        "invalid_mask_selections": env.invalid_action_selections,
                        "failed_actions": failed_actions,
                        "hosts_compromised": len(base.knowledge.access),
                        "path_events": len(base.attack_path()),
                    }
                )
            env.close()
    after = policy_digest(model)
    if before != after:
        raise TransferStudyError("policy parameters changed during frozen evaluation")
    if any(row["invalid_mask_selections"] for row in episodes):
        raise TransferStudyError("evaluation recorded an invalid masked selection")
    return (
        episodes,
        steps,
        {
            "gradient_updates": False,
            "policy_sha256_before": before,
            "policy_sha256_after": after,
            "evaluation_reset_count": reset_count,
            "knowledge_mask_checks": mask_checks,
            "invalid_mask_selections": 0,
        },
    )


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
        "phase": "phase11_frozen_transfer_comparison",
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


__all__ = [
    "DEFAULT_FROZEN_INPUTS",
    "DEFAULT_RESULT_ROOT",
    "DEFAULT_RUN_ROOT",
    "TransferStudyError",
    "aggregate_seed_metrics",
    "analyse_seed_metrics",
    "arm_run_name",
    "current_input_manifest",
    "evaluate_arm",
    "freeze_inputs",
    "git_commit",
    "load_frozen_inputs",
    "load_model",
    "persist_and_write_run_evaluation",
    "train_arm",
    "validate_finite_evidence",
    "validate_frozen_inputs",
    "validate_paired_target_isolation",
    "validate_training_manifest",
    "write_study_summary",
]
