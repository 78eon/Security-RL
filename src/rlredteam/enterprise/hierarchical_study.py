"""Training, frozen evaluation and evidence for the Phase 12 hierarchical study."""

from __future__ import annotations

import json
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

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
from rlredteam.enterprise.graph_policy_study import TrainingDistributionAudit, observation_schema
from rlredteam.enterprise.hierarchical_policy import (
    PHASE12_ARMS,
    HierarchicalResearchConfig,
    action_to_objective,
    fixed_action_catalogue,
    phase12_policy,
    phase12_policy_kwargs,
)
from rlredteam.enterprise.infrastructure_generalisation import (
    infrastructure_distribution_manifest,
    infrastructure_vulnerability_manifest,
)
from rlredteam.enterprise.onprem import OnPremGeneralisationSplit
from rlredteam.enterprise.profiles import EnterpriseProfileConfig, InfrastructureCurriculumEnv
from rlredteam.enterprise.recurrent import KnowledgeActionGuard
from rlredteam.frameworks import event_framework_fields
from rlredteam.train import set_all_seeds

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_FROZEN_INPUTS = REPO_ROOT / "configs/frozen_hierarchical_policy.json"
DEFAULT_RUN_ROOT = REPO_ROOT / "runs/advanced-rl-hierarchical-policy-v1"
DEFAULT_RESULT_ROOT = REPO_ROOT / "results/advanced-rl-hierarchical-policy-v1"


class HierarchicalStudyError(GeneralisationError):
    """Raised when a Phase 12 protocol or evidence invariant is violated."""


def split_from_config(config: HierarchicalResearchConfig) -> OnPremGeneralisationSplit:
    return OnPremGeneralisationSplit(
        train=config.topology_splits["train"],
        validation=config.topology_splits["validation"],
        test=config.topology_splits["test"],
    )


def arm_run_name(config: HierarchicalResearchConfig, arm: str, seed: int) -> str:
    if arm not in config.arms:
        raise ValueError(f"unknown Phase 12 arm: {arm}")
    return f"{config.experiment_id}-{arm}-s{int(seed)}"


def current_input_manifest(config: HierarchicalResearchConfig) -> dict[str, Any]:
    base = EnterpriseProfileConfig.from_yaml()
    split = split_from_config(config)
    source_files = (
        REPO_ROOT / "src/rlredteam/analyse.py",
        REPO_ROOT / "src/rlredteam/train.py",
        REPO_ROOT / "src/rlredteam/enterprise/curriculum.py",
        REPO_ROOT / "src/rlredteam/enterprise/curriculum_study.py",
        REPO_ROOT / "src/rlredteam/enterprise/environment.py",
        REPO_ROOT / "src/rlredteam/enterprise/generalisation.py",
        REPO_ROOT / "src/rlredteam/enterprise/graph_policy.py",
        REPO_ROOT / "src/rlredteam/enterprise/hierarchical_policy.py",
        REPO_ROOT / "src/rlredteam/enterprise/hierarchical_study.py",
        REPO_ROOT / "src/rlredteam/enterprise/infrastructure_generalisation.py",
        REPO_ROOT / "src/rlredteam/enterprise/model.py",
        REPO_ROOT / "src/rlredteam/enterprise/onprem.py",
        REPO_ROOT / "src/rlredteam/enterprise/profiles.py",
        REPO_ROOT / "src/rlredteam/enterprise/recurrent.py",
        REPO_ROOT / "src/rlredteam/enterprise/state.py",
        REPO_ROOT / "src/rlredteam/storage/postgres_logger.py",
        REPO_ROOT / "src/rlredteam/storage/schema.sql",
        REPO_ROOT / "scripts/run_hierarchical_study.py",
    )
    distribution = infrastructure_distribution_manifest(split, base, config.train_profiles)
    return {
        "schema_version": 1,
        "experiment_id": config.experiment_id,
        "experiment_config_sha256": config.digest(),
        "base_profile_config_sha256": base.digest(),
        "dependency_lock_hash": dependency_lock_hash(),
        "distribution_sha256": canonical_digest(distribution),
        "vulnerability_snapshot_sha256": infrastructure_vulnerability_manifest(
            split, base, config.train_profiles
        ),
        "observation_schema": observation_schema(),
        "hierarchy": {
            "action_to_objective": list(action_to_objective(config)),
            "action_catalogue_sha256": canonical_digest(fixed_action_catalogue()),
            "source": "static_public_slot_catalogue",
            "transition_semantics": "one_concrete_action_per_environment_step",
        },
        "source_sha256": {
            str(path.relative_to(REPO_ROOT)): sha256_file(path) for path in source_files
        },
    }


def freeze_inputs(
    output: Path, config: HierarchicalResearchConfig | None = None
) -> dict[str, Any]:
    config = config or HierarchicalResearchConfig.from_yaml()
    if git_dirty() is not False:
        raise HierarchicalStudyError("refusing to freeze Phase 12 inputs from a dirty tree")
    frozen = current_input_manifest(config)
    frozen.update({"frozen_at": datetime.now(UTC).isoformat(), "protocol_commit": git_commit()})
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(frozen, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return frozen


def load_frozen_inputs(path: Path = DEFAULT_FROZEN_INPUTS) -> dict[str, Any]:
    frozen = json.loads(Path(path).read_text())
    if not isinstance(frozen, dict) or frozen.get("schema_version") != 1:
        raise HierarchicalStudyError("invalid Phase 12 frozen-input manifest")
    return frozen


def validate_frozen_inputs(
    frozen: dict[str, Any], config: HierarchicalResearchConfig
) -> None:
    current = current_input_manifest(config)
    expected = {key: frozen.get(key) for key in current}
    if current != expected:
        differences = [key for key in current if current.get(key) != expected.get(key)]
        raise HierarchicalStudyError(
            f"Phase 12 frozen inputs differ: {', '.join(sorted(differences))}"
        )
    if git_dirty() is not False:
        raise HierarchicalStudyError("canonical Phase 12 execution requires a clean tree")


def make_model(config: HierarchicalResearchConfig, arm: str, seed: int, env):
    from sb3_contrib import MaskablePPO

    return MaskablePPO(
        phase12_policy(config, arm),
        env,
        seed=seed,
        device="cpu",
        verbose=0,
        policy_kwargs=phase12_policy_kwargs(config, arm),
        **config.common_ppo,
    )


def _verify_training_records(
    records: list[dict[str, Any]],
    distribution: dict[str, Any],
    profile_hash: str,
    *,
    run_name: str,
) -> None:
    if not records:
        raise HierarchicalStudyError(f"training contains no observed resets: {run_name}")
    for record in records:
        profile = record["profile"]
        seed = str(record["topology_seed"])
        if profile not in distribution["train"] or seed not in distribution["train"][profile]:
            raise HierarchicalStudyError(f"training selected an undeclared case: {run_name}")
        if record["topology_hash"] != distribution["train"][profile][seed]:
            raise HierarchicalStudyError(f"training topology hash drift: {run_name}")
        if record["profile_config_hash"] != profile_hash:
            raise HierarchicalStudyError(f"training profile hash drift: {run_name}")


def train_arm(
    output_dir: Path,
    *,
    arm: str,
    training_seed: int,
    config: HierarchicalResearchConfig | None = None,
    frozen_inputs: dict[str, Any] | None = None,
    timesteps: int | None = None,
    development: bool = False,
    allow_dirty: bool = False,
) -> dict[str, Any]:
    """Train one fresh arm and prove exact concrete-transition accounting."""
    config = config or HierarchicalResearchConfig.from_yaml()
    if arm not in config.arms:
        raise ValueError(f"unknown Phase 12 arm: {arm}")
    seed = int(training_seed)
    budget = int(timesteps or config.total_timesteps)
    if budget <= 0 or budget % int(config.common_ppo["n_steps"]):
        raise HierarchicalStudyError("training budget must contain complete PPO rollouts")
    if development:
        if seed != config.development_seed:
            raise HierarchicalStudyError("development execution must use the excluded seed")
    elif seed not in config.training_seeds:
        raise HierarchicalStudyError("canonical training seed is outside the protocol")
    dirty = git_dirty()
    if dirty is not False and not (development and allow_dirty):
        raise HierarchicalStudyError("training requires a clean tracked tree")
    if not development:
        if frozen_inputs is None:
            raise HierarchicalStudyError("canonical training requires frozen inputs")
        validate_frozen_inputs(frozen_inputs, config)

    import torch

    torch.set_num_threads(1)
    set_all_seeds(seed)
    base = EnterpriseProfileConfig.from_yaml()
    split = split_from_config(config)
    distribution = infrastructure_distribution_manifest(split, base, config.train_profiles)
    training = InfrastructureCurriculumEnv(
        config.topology_splits["train"], config.train_profiles, config=base
    )
    audited = TrainingDistributionAudit(training)
    guarded = KnowledgeActionGuard(audited)
    environment_seed = seed * 100 + 1
    guarded.reset(seed=environment_seed)
    audited.records.clear()
    model = make_model(config, arm, seed, guarded)
    started = time.monotonic()
    model.learn(total_timesteps=budget)
    elapsed = time.monotonic() - started
    if int(model.num_timesteps) != budget:
        raise HierarchicalStudyError(
            f"actual training steps {model.num_timesteps} differ from {budget}"
        )
    if guarded.invalid_action_selections:
        raise HierarchicalStudyError(f"invalid training action selected: {arm}/{seed}")
    run_name = arm_run_name(config, arm, seed)
    _verify_training_records(audited.records, distribution, base.digest(), run_name=run_name)
    if not all(
        bool(np.isfinite(parameter.detach().cpu().numpy()).all())
        for parameter in model.policy.parameters()
    ):
        raise HierarchicalStudyError("trained policy contains non-finite parameters")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / "model"
    model.save(checkpoint)
    checkpoint = checkpoint.with_suffix(".zip")
    counts = Counter(f"{item['profile']}:{item['topology_seed']}" for item in audited.records)
    profile_counts = Counter(item["profile"] for item in audited.records)
    representation = (
        "agent_knowledge_graph_flat_action"
        if arm == "flat_graph"
        else "agent_knowledge_graph_objective_conditional_action"
    )
    manifest = {
        "schema_version": 1,
        "study_id": config.experiment_id,
        "experiment_id": run_name,
        "description": config.description,
        "arm": arm,
        "algorithm": "MaskablePPO",
        "policy": "AgentKnowledgeGraphPolicy",
        "representation": representation,
        "action_mask_source": "AgentKnowledge",
        "observation_source": "AgentKnowledge",
        "observation_schema": observation_schema(),
        "transition_semantics": "one_concrete_action_per_environment_step",
        "development": bool(development),
        "training_seed": seed,
        "training_timesteps": budget,
        "actual_training_timesteps": int(model.num_timesteps),
        "training_elapsed_seconds": elapsed,
        "train_profiles": [profile.value for profile in config.train_profiles],
        "train_topology_seeds": list(config.topology_splits["train"]),
        "validation_topology_seeds": list(config.topology_splits["validation"]),
        "test_topology_seeds": list(config.topology_splits["test"]),
        "study_config_hash": config.digest(),
        "scientific_config_hash": config.scientific_digest(),
        "base_profile_config_hash": base.digest(),
        "dependency_lock_hash": dependency_lock_hash(),
        "git_commit": git_commit(),
        "git_dirty": dirty,
        "ppo": config.common_ppo,
        "graph_policy_config": config.graph_policy,
        "hierarchy_config": config.hierarchy if arm == "hierarchical_graph" else None,
        "parameter_count": sum(parameter.numel() for parameter in model.policy.parameters()),
        "torch_num_threads": torch.get_num_threads(),
        "environment_seed": environment_seed,
        "training_reset_count": len(audited.records),
        "observed_profile_counts": dict(sorted(profile_counts.items())),
        "exposure_counts": dict(sorted(counts.items())),
        "reset_trace_sha256": canonical_digest(audited.records),
        "base_distribution": distribution,
        "base_vulnerability_snapshot_sha256": infrastructure_vulnerability_manifest(
            split, base, config.train_profiles
        ),
        "frozen_inputs_sha256": (
            canonical_digest(frozen_inputs) if frozen_inputs is not None else None
        ),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "policy_sha256": policy_digest(model),
        "weights_release": "gated; runs/ is gitignored",
    }
    (output_dir / "training_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    guarded.close()
    return manifest


def validate_training_manifest(
    manifest: dict[str, Any],
    checkpoint: Path,
    *,
    arm: str,
    training_seed: int,
    config: HierarchicalResearchConfig,
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
        "transition semantics": (
            manifest.get("transition_semantics"),
            "one_concrete_action_per_environment_step",
        ),
        "training seed": (manifest.get("training_seed"), int(training_seed)),
        "training budget": (
            manifest.get("actual_training_timesteps"),
            manifest.get("training_timesteps"),
        ),
        "study config": (manifest.get("study_config_hash"), config.digest()),
        "development": (manifest.get("development"), bool(development)),
        "checkpoint": (manifest.get("checkpoint_sha256"), sha256_file(checkpoint)),
        "observation schema": (manifest.get("observation_schema"), observation_schema()),
        "graph policy": (manifest.get("graph_policy_config"), config.graph_policy),
        "hierarchy": (
            manifest.get("hierarchy_config"),
            config.hierarchy if arm == "hierarchical_graph" else None,
        ),
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
        raise HierarchicalStudyError(
            "Phase 12 checkpoint provenance mismatch:\n  " + "\n  ".join(failures)
        )


def validate_paired_training_isolation(
    flat: dict[str, Any], hierarchical: dict[str, Any]
) -> None:
    controlled_fields = (
        "study_id",
        "description",
        "algorithm",
        "policy_family",
        "action_mask_source",
        "observation_source",
        "observation_schema",
        "transition_semantics",
        "development",
        "training_seed",
        "training_timesteps",
        "actual_training_timesteps",
        "train_profiles",
        "train_topology_seeds",
        "validation_topology_seeds",
        "test_topology_seeds",
        "study_config_hash",
        "scientific_config_hash",
        "base_profile_config_hash",
        "dependency_lock_hash",
        "git_commit",
        "git_dirty",
        "ppo",
        "graph_policy_config",
        "torch_num_threads",
        "environment_seed",
        "base_distribution",
        "base_vulnerability_snapshot_sha256",
        "frozen_inputs_sha256",
    )
    differences = [
        field for field in controlled_fields if flat.get(field) != hierarchical.get(field)
    ]
    if differences:
        raise HierarchicalStudyError(
            "paired arms differ outside hierarchy: " + ", ".join(differences)
        )


def load_model(checkpoint: Path):
    from sb3_contrib import MaskablePPO

    return MaskablePPO.load(checkpoint, device="cpu")


def evaluate_arm(
    model,
    *,
    arm: str,
    training_seed: int,
    config: HierarchicalResearchConfig,
    topology_seeds: tuple[int, ...],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Evaluate a frozen model on an explicit held-out topology set."""
    if arm not in PHASE12_ARMS:
        raise ValueError(f"unknown Phase 12 arm: {arm}")
    if not topology_seeds or set(topology_seeds) & set(config.topology_splits["train"]):
        raise HierarchicalStudyError("evaluation topology seeds are absent or overlap training")
    base_config = EnterpriseProfileConfig.from_yaml()
    episodes: list[dict[str, Any]] = []
    steps: list[dict[str, Any]] = []
    before = policy_digest(model)
    mask_checks = reset_count = 0
    for profile in config.train_profiles:
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
                        raise HierarchicalStudyError(
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
                        raise HierarchicalStudyError("policy selected a masked action")
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
                            **event_framework_fields(
                                rl_action_index=selected,
                                simulator_action=event.action.name,
                                action_kind=event.action.type.value,
                            ),
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
        raise HierarchicalStudyError("policy parameters changed during frozen evaluation")
    if any(row["invalid_mask_selections"] for row in episodes):
        raise HierarchicalStudyError("evaluation recorded an invalid masked selection")
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
            "environment_steps": len(steps),
            "option_decisions": len(steps) if arm == "hierarchical_graph" else 0,
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
        "phase": "phase12_frozen_hierarchical_policy_comparison",
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
    "HierarchicalStudyError",
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
    "validate_paired_training_isolation",
    "validate_training_manifest",
    "write_study_summary",
]
