"""Training, frozen evaluation and evidence for the Phase 13 red-blue study."""

from __future__ import annotations

import json
import math
import statistics
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from rlredteam.analyse import paired_comparison
from rlredteam.enterprise.curriculum import canonical_digest
from rlredteam.enterprise.curriculum_study import write_study_summary
from rlredteam.enterprise.generalisation import (
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
from rlredteam.enterprise.multiagent import (
    PHASE13_ARMS,
    DefenderTrainingEnv,
    MultiAgentResearchConfig,
    MultiAgentStudyError,
    RedAgainstDefenderEnv,
    StaticDefenderController,
    make_defender_model,
    make_red_model,
    phase13_policy_manifest,
)
from rlredteam.enterprise.onprem import OnPremGeneralisationSplit
from rlredteam.enterprise.profiles import EnterpriseProfileConfig
from rlredteam.enterprise.recurrent import KnowledgeActionGuard
from rlredteam.frameworks import event_framework_fields
from rlredteam.train import set_all_seeds

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_FROZEN_INPUTS = REPO_ROOT / "configs/frozen_multiagent_defense.json"
DEFAULT_RUN_ROOT = REPO_ROOT / "runs/advanced-rl-multiagent-defense-v1"
DEFAULT_RESULT_ROOT = REPO_ROOT / "results/advanced-rl-multiagent-defense-v1"
PROTOCOL_DOCUMENT = REPO_ROOT / "docs/PHASE_13_MULTI_AGENT_RL.md"
PROTOCOL_HASH_FILE = REPO_ROOT / "configs/phase13_protocol.sha256"
PHASE13_SCHEMA = REPO_ROOT / "src/rlredteam/storage/phase13_schema.sql"


def split_from_config(config: MultiAgentResearchConfig) -> OnPremGeneralisationSplit:
    return OnPremGeneralisationSplit(
        train=config.topology_splits["train"],
        validation=config.topology_splits["validation"],
        test=config.topology_splits["test"],
    )


def arm_run_name(config: MultiAgentResearchConfig, arm: str, seed: int) -> str:
    if arm not in config.arms:
        raise ValueError(f"unknown Phase 13 arm: {arm}")
    return f"{config.experiment_id}-{arm}-s{int(seed)}"


def _protocol_document_hash(*, verify_document: bool) -> str:
    try:
        line = PROTOCOL_HASH_FILE.read_text().strip()
        expected, path = line.split(maxsplit=1)
    except (OSError, ValueError) as exc:
        raise MultiAgentStudyError("invalid Phase 13 protocol hash file") from exc
    if path != "docs/PHASE_13_MULTI_AGENT_RL.md" or len(expected) != 64:
        raise MultiAgentStudyError("Phase 13 protocol hash declaration differs")
    if verify_document:
        if not PROTOCOL_DOCUMENT.is_file():
            raise MultiAgentStudyError("local ignored Phase 13 protocol document is absent")
        if sha256_file(PROTOCOL_DOCUMENT) != expected:
            raise MultiAgentStudyError("local Phase 13 protocol document hash differs")
    return expected


def current_input_manifest(
    config: MultiAgentResearchConfig,
    *,
    defender_checkpoint: Path | None = None,
    verify_protocol_document: bool = False,
) -> dict[str, Any]:
    base = EnterpriseProfileConfig.from_yaml()
    split = split_from_config(config)
    source_files = (
        REPO_ROOT / "src/rlredteam/analyse.py",
        REPO_ROOT / "src/rlredteam/train.py",
        REPO_ROOT / "src/rlredteam/enterprise/curriculum.py",
        REPO_ROOT / "src/rlredteam/enterprise/curriculum_study.py",
        REPO_ROOT / "src/rlredteam/enterprise/defender_state.py",
        REPO_ROOT / "src/rlredteam/enterprise/environment.py",
        REPO_ROOT / "src/rlredteam/enterprise/generalisation.py",
        REPO_ROOT / "src/rlredteam/enterprise/graph_policy.py",
        REPO_ROOT / "src/rlredteam/enterprise/hierarchical_policy.py",
        REPO_ROOT / "src/rlredteam/enterprise/infrastructure_generalisation.py",
        REPO_ROOT / "src/rlredteam/enterprise/model.py",
        REPO_ROOT / "src/rlredteam/enterprise/multiagent.py",
        REPO_ROOT / "src/rlredteam/enterprise/multiagent_completion.py",
        REPO_ROOT / "src/rlredteam/enterprise/multiagent_study.py",
        REPO_ROOT / "src/rlredteam/enterprise/onprem.py",
        REPO_ROOT / "src/rlredteam/enterprise/profiles.py",
        REPO_ROOT / "src/rlredteam/enterprise/recurrent.py",
        REPO_ROOT / "src/rlredteam/enterprise/state.py",
        REPO_ROOT / "src/rlredteam/storage/phase13_schema.sql",
        REPO_ROOT / "src/rlredteam/storage/postgres_logger.py",
        REPO_ROOT / "src/rlredteam/storage/schema.sql",
        REPO_ROOT / "scripts/run_multiagent_study.py",
        REPO_ROOT / "scripts/verify_multiagent_completion.py",
        REPO_ROOT / "tests/test_multiagent.py",
        REPO_ROOT / "tests/test_multiagent_completion.py",
        REPO_ROOT / "tests/test_multiagent_study.py",
    )
    distribution = infrastructure_distribution_manifest(split, base, config.train_profiles)
    result = {
        "schema_version": 1,
        "experiment_id": config.experiment_id,
        "experiment_config_sha256": config.digest(),
        "base_profile_config_sha256": base.digest(),
        "dependency_lock_hash": dependency_lock_hash(),
        "distribution_sha256": canonical_digest(distribution),
        "vulnerability_snapshot_sha256": infrastructure_vulnerability_manifest(
            split, base, config.train_profiles
        ),
        "protocol_document_sha256": _protocol_document_hash(
            verify_document=verify_protocol_document
        ),
        "policy_manifest": phase13_policy_manifest(config),
        "source_sha256": {
            str(path.relative_to(REPO_ROOT)): sha256_file(path) for path in source_files
        },
    }
    if defender_checkpoint is not None:
        checkpoint = Path(defender_checkpoint)
        result["defender_checkpoint_sha256"] = sha256_file(checkpoint)
        manifest_path = checkpoint.parent / "training_manifest.json"
        result["defender_training_manifest_sha256"] = sha256_file(manifest_path)
    return result


def freeze_inputs(
    output: Path,
    *,
    development_study: Path,
    defender_checkpoint: Path,
    config: MultiAgentResearchConfig | None = None,
) -> dict[str, Any]:
    config = config or MultiAgentResearchConfig.from_yaml()
    if git_dirty() is not False:
        raise MultiAgentStudyError("refusing to freeze Phase 13 inputs from a dirty tree")
    if not Path(development_study).is_file():
        raise MultiAgentStudyError("full-budget Phase 13 development evidence is absent")
    frozen = current_input_manifest(
        config,
        defender_checkpoint=defender_checkpoint,
        verify_protocol_document=True,
    )
    frozen.update(
        {
            "development_study_sha256": sha256_file(development_study),
            "frozen_at": datetime.now(UTC).isoformat(),
            "protocol_commit": git_commit(),
        }
    )
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(frozen, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return frozen


def load_frozen_inputs(path: Path = DEFAULT_FROZEN_INPUTS) -> dict[str, Any]:
    try:
        frozen = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise MultiAgentStudyError("invalid Phase 13 frozen-input manifest") from exc
    if not isinstance(frozen, dict) or frozen.get("schema_version") != 1:
        raise MultiAgentStudyError("invalid Phase 13 frozen-input manifest")
    return frozen


def validate_frozen_inputs(
    frozen: dict[str, Any],
    config: MultiAgentResearchConfig,
    *,
    defender_checkpoint: Path,
    allow_dirty: bool = False,
) -> None:
    current = current_input_manifest(config, defender_checkpoint=defender_checkpoint)
    expected = {key: frozen.get(key) for key in current}
    if current != expected:
        differences = [key for key in current if current.get(key) != expected.get(key)]
        raise MultiAgentStudyError(
            f"Phase 13 frozen inputs differ: {', '.join(sorted(differences))}"
        )
    if git_dirty() is not False and not allow_dirty:
        raise MultiAgentStudyError("canonical Phase 13 execution requires a clean tree")


def load_maskable_model(checkpoint: Path):
    from sb3_contrib import MaskablePPO

    return MaskablePPO.load(checkpoint, device="cpu")


def _finite_parameters(model, *, label: str) -> None:
    if not all(
        bool(np.isfinite(parameter.detach().cpu().numpy()).all())
        for parameter in model.policy.parameters()
    ):
        raise MultiAgentStudyError(f"{label} contains non-finite parameters")


def _training_reset_evidence(
    records: list[dict[str, Any]], config: MultiAgentResearchConfig
) -> tuple[dict[str, int], dict[str, int]]:
    if not records:
        raise MultiAgentStudyError("training contains no observed environment reset")
    distribution = infrastructure_distribution_manifest(
        split_from_config(config), EnterpriseProfileConfig.from_yaml(), config.train_profiles
    )["train"]
    for record in records:
        profile = str(record["profile"])
        seed = str(record["topology_seed"])
        if profile not in distribution or seed not in distribution[profile]:
            raise MultiAgentStudyError("training selected an undeclared topology case")
        if record["topology_hash"] != distribution[profile][seed]:
            raise MultiAgentStudyError("training topology hash drifted")
    exposure = Counter(f"{row['profile']}:{row['topology_seed']}" for row in records)
    profiles = Counter(str(row["profile"]) for row in records)
    if set(profiles) != {item.value for item in config.train_profiles}:
        raise MultiAgentStudyError("training did not cover every infrastructure profile")
    return dict(sorted(exposure.items())), dict(sorted(profiles.items()))


def train_red_arm(
    output_dir: Path,
    *,
    arm: str,
    training_seed: int,
    defender_checkpoint: Path,
    config: MultiAgentResearchConfig | None = None,
    frozen_inputs: dict[str, Any] | None = None,
    timesteps: int | None = None,
    development: bool = False,
    allow_dirty: bool = False,
) -> dict[str, Any]:
    """Train one red arm while keeping architecture, reward and budget matched."""
    config = config or MultiAgentResearchConfig.from_yaml()
    if arm not in PHASE13_ARMS:
        raise ValueError(f"unknown Phase 13 arm: {arm}")
    seed = int(training_seed)
    budget = int(timesteps or config.red_total_timesteps)
    if budget <= 0 or budget % int(config.red_ppo["n_steps"]):
        raise MultiAgentStudyError("red training budget must contain complete PPO rollouts")
    if development and seed != config.development_seed:
        raise MultiAgentStudyError("development red training requires the excluded seed")
    if not development and seed not in config.training_seeds:
        raise MultiAgentStudyError("canonical red training seed is outside the protocol")
    dirty = git_dirty()
    if dirty is not False and not (development and allow_dirty):
        raise MultiAgentStudyError("red training requires a clean tracked tree")
    if not development:
        if frozen_inputs is None:
            raise MultiAgentStudyError("canonical red training requires frozen inputs")
        validate_frozen_inputs(
            frozen_inputs,
            config,
            defender_checkpoint=defender_checkpoint,
        )

    import torch

    torch.set_num_threads(1)
    set_all_seeds(seed)
    if arm == "static_defender_training":
        defender = StaticDefenderController()
        defender_hash_before = defender_hash_after = None
        defender_mode = "balanced_ids_monitor_only"
    else:
        defender = load_maskable_model(defender_checkpoint)
        defender_hash_before = policy_digest(defender)
        defender_hash_after = None
        defender_mode = "frozen_learned_dynamic_ids_and_delayed_patch"
    red_environment = RedAgainstDefenderEnv(
        config.topology_splits["train"],
        config.train_profiles,
        defender=defender,
        config=config,
        deterministic_defender=False,
    )
    guarded = KnowledgeActionGuard(red_environment)
    environment_seed = seed * 100 + 1
    guarded.reset(seed=environment_seed)
    red_environment.interaction.reset_records.clear()
    model = make_red_model(config, seed, guarded)
    red_hash_before = policy_digest(model)
    started = time.monotonic()
    model.learn(total_timesteps=budget)
    elapsed = time.monotonic() - started
    if int(model.num_timesteps) != budget:
        raise MultiAgentStudyError("red model did not consume the exact transition budget")
    if guarded.invalid_action_selections or red_environment.interaction.invalid_red_actions:
        raise MultiAgentStudyError("red model selected an AgentKnowledge-masked action")
    if red_environment.interaction.invalid_defender_actions:
        raise MultiAgentStudyError("defender selected a DefenderKnowledge-masked action")
    _finite_parameters(model, label="red policy")
    red_hash_after = policy_digest(model)
    if red_hash_before == red_hash_after:
        raise MultiAgentStudyError("red optimisation did not update policy parameters")
    if arm == "adaptive_defender_training":
        defender_hash_after = policy_digest(defender)
        if defender_hash_before != defender_hash_after:
            raise MultiAgentStudyError("frozen blue policy changed during red optimisation")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / "model"
    model.save(checkpoint)
    checkpoint = checkpoint.with_suffix(".zip")
    records = red_environment.interaction.reset_records
    exposure, profile_counts = _training_reset_evidence(records, config)
    knowledge = red_environment.defender_knowledge
    distribution = infrastructure_distribution_manifest(
        split_from_config(config), EnterpriseProfileConfig.from_yaml(), config.train_profiles
    )
    manifest = {
        "schema_version": 1,
        "study_id": config.experiment_id,
        "experiment_id": arm_run_name(config, arm, seed),
        "description": config.description,
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
        "defender_mode": defender_mode,
        "development": bool(development),
        "training_seed": seed,
        "environment_seed": environment_seed,
        "red_training_timesteps": budget,
        "actual_red_training_timesteps": int(model.num_timesteps),
        "red_training_elapsed_seconds": elapsed,
        "train_profiles": [item.value for item in config.train_profiles],
        "train_topology_seeds": list(config.topology_splits["train"]),
        "validation_topology_seeds": list(config.topology_splits["validation"]),
        "test_topology_seeds": list(config.topology_splits["test"]),
        "study_config_hash": config.digest(),
        "scientific_config_hash": config.scientific_digest(),
        "base_profile_config_hash": EnterpriseProfileConfig.from_yaml().digest(),
        "dependency_lock_hash": dependency_lock_hash(),
        "git_commit": git_commit(),
        "git_dirty": dirty,
        "red_ppo": config.red_ppo,
        "graph_policy_config": config.graph_policy,
        "hierarchy_config": config.hierarchy,
        "defense_config": config.defense,
        "red_parameter_count": sum(parameter.numel() for parameter in model.policy.parameters()),
        "torch_num_threads": torch.get_num_threads(),
        "training_reset_count": len(records),
        "observed_profile_counts": profile_counts,
        "exposure_counts": exposure,
        "reset_trace_sha256": canonical_digest(records),
        "reset_records": records,
        "defender_decision_count": knowledge.decision_count,
        "defender_threshold_adjustments": knowledge.threshold_adjustments,
        "defender_patch_schedules": knowledge.patch_schedules,
        "defender_patch_activations": knowledge.patch_activations,
        "training_detection_count": red_environment.interaction.detection_count,
        "training_mitigation_block_count": (
            red_environment.interaction.mitigation_block_count
        ),
        "defender_checkpoint_sha256": (
            sha256_file(defender_checkpoint)
            if arm == "adaptive_defender_training"
            else None
        ),
        "defender_policy_sha256_before": defender_hash_before,
        "defender_policy_sha256_after": defender_hash_after,
        "base_distribution": distribution,
        "base_vulnerability_snapshot_sha256": infrastructure_vulnerability_manifest(
            split_from_config(config), EnterpriseProfileConfig.from_yaml(), config.train_profiles
        ),
        "frozen_inputs_sha256": (
            canonical_digest(frozen_inputs) if frozen_inputs is not None else None
        ),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "policy_sha256_before_training": red_hash_before,
        "policy_sha256": red_hash_after,
        "weights_release": "gated; runs/ is gitignored",
    }
    if manifest["defender_decision_count"] != budget:
        raise MultiAgentStudyError("blue/red turn accounting differs from the red budget")
    (output_dir / "training_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    guarded.close()
    return manifest


def train_defender(
    output_dir: Path,
    *,
    bootstrap_red_checkpoint: Path,
    config: MultiAgentResearchConfig | None = None,
    timesteps: int | None = None,
    allow_dirty: bool = False,
) -> dict[str, Any]:
    """Train blue on the excluded seed while proving frozen-red immutability."""
    config = config or MultiAgentResearchConfig.from_yaml()
    budget = int(timesteps or config.defender_total_timesteps)
    if budget <= 0 or budget % int(config.defender_ppo["n_steps"]):
        raise MultiAgentStudyError("defender budget must contain complete PPO rollouts")
    dirty = git_dirty()
    if dirty is not False and not allow_dirty:
        raise MultiAgentStudyError("defender training requires a clean tracked tree")
    bootstrap_red_checkpoint = Path(bootstrap_red_checkpoint)
    if not bootstrap_red_checkpoint.is_file():
        raise MultiAgentStudyError("bootstrap red checkpoint is absent")

    import torch

    torch.set_num_threads(1)
    seed = config.development_seed
    set_all_seeds(seed)
    red = load_maskable_model(bootstrap_red_checkpoint)
    red_before = policy_digest(red)
    environment = DefenderTrainingEnv(
        config.topology_splits["train"],
        config.train_profiles,
        red_policy=red,
        config=config,
    )
    environment_seed = seed * 100 + 2
    environment.reset(seed=environment_seed)
    environment.interaction.reset_records.clear()
    model = make_defender_model(config, seed, environment)
    blue_before = policy_digest(model)
    started = time.monotonic()
    model.learn(total_timesteps=budget)
    elapsed = time.monotonic() - started
    if int(model.num_timesteps) != budget:
        raise MultiAgentStudyError("blue model did not consume the exact transition budget")
    if policy_digest(red) != red_before:
        raise MultiAgentStudyError("bootstrap red policy changed during blue optimisation")
    blue_after = policy_digest(model)
    if blue_before == blue_after:
        raise MultiAgentStudyError("blue optimisation did not update policy parameters")
    if environment.interaction.invalid_defender_actions:
        raise MultiAgentStudyError("blue model selected a DefenderKnowledge-masked action")
    if environment.interaction.invalid_red_actions:
        raise MultiAgentStudyError("bootstrap red selected an AgentKnowledge-masked action")
    if environment.red_mask_checks != budget:
        raise MultiAgentStudyError("frozen-red decision accounting differs from blue budget")
    _finite_parameters(model, label="blue policy")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / "model"
    model.save(checkpoint)
    checkpoint = checkpoint.with_suffix(".zip")
    records = environment.interaction.reset_records
    exposure, profile_counts = _training_reset_evidence(records, config)
    knowledge = environment.interaction.defender_knowledge
    manifest = {
        "schema_version": 1,
        "study_id": config.experiment_id,
        "experiment_id": f"{config.experiment_id}-development-defender",
        "role": "defender",
        "algorithm": "MaskablePPO",
        "policy": "MlpPolicy",
        "observation_source": "DefenderKnowledge",
        "action_mask_source": "DefenderKnowledge",
        "red_observation_source": "AgentKnowledge",
        "policy_manifest": phase13_policy_manifest(config),
        "training_seed": seed,
        "environment_seed": environment_seed,
        "defender_training_timesteps": budget,
        "actual_defender_training_timesteps": int(model.num_timesteps),
        "training_elapsed_seconds": elapsed,
        "train_profiles": [item.value for item in config.train_profiles],
        "train_topology_seeds": list(config.topology_splits["train"]),
        "study_config_hash": config.digest(),
        "scientific_config_hash": config.scientific_digest(),
        "base_profile_config_hash": EnterpriseProfileConfig.from_yaml().digest(),
        "dependency_lock_hash": dependency_lock_hash(),
        "git_commit": git_commit(),
        "git_dirty": dirty,
        "defender_ppo": config.defender_ppo,
        "defense_config": config.defense,
        "parameter_count": sum(parameter.numel() for parameter in model.policy.parameters()),
        "torch_num_threads": torch.get_num_threads(),
        "training_reset_count": len(records),
        "observed_profile_counts": profile_counts,
        "exposure_counts": exposure,
        "reset_trace_sha256": canonical_digest(records),
        "reset_records": records,
        "decision_count": knowledge.decision_count,
        "threshold_adjustments": knowledge.threshold_adjustments,
        "patch_schedules": knowledge.patch_schedules,
        "patch_activations": knowledge.patch_activations,
        "detection_count": environment.interaction.detection_count,
        "mitigation_block_count": environment.interaction.mitigation_block_count,
        "red_mask_checks": environment.red_mask_checks,
        "bootstrap_red_checkpoint_sha256": sha256_file(bootstrap_red_checkpoint),
        "bootstrap_red_policy_sha256_before": red_before,
        "bootstrap_red_policy_sha256_after": policy_digest(red),
        "policy_sha256_before_training": blue_before,
        "policy_sha256": blue_after,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "weights_release": "gated; runs/ is gitignored",
    }
    (output_dir / "training_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    environment.close()
    return manifest


def validate_red_training_manifest(
    manifest: dict[str, Any],
    checkpoint: Path,
    *,
    arm: str,
    training_seed: int,
    defender_checkpoint: Path,
    config: MultiAgentResearchConfig,
    frozen_inputs: dict[str, Any] | None,
    development: bool,
    allow_unverifiable: bool = False,
) -> None:
    expected = {
        "study": (manifest.get("study_id"), config.experiment_id),
        "run": (manifest.get("experiment_id"), arm_run_name(config, arm, training_seed)),
        "arm": (manifest.get("arm"), arm),
        "algorithm": (manifest.get("algorithm"), "MaskablePPO"),
        "red observation": (manifest.get("red_observation_source"), "AgentKnowledge"),
        "red mask": (manifest.get("red_action_mask_source"), "AgentKnowledge"),
        "blue observation": (
            manifest.get("defender_observation_source"),
            "DefenderKnowledge",
        ),
        "blue mask": (
            manifest.get("defender_action_mask_source"),
            "DefenderKnowledge",
        ),
        "turn semantics": (
            manifest.get("turn_semantics"),
            "one_blue_decision_then_one_red_transition",
        ),
        "training seed": (manifest.get("training_seed"), int(training_seed)),
        "training budget": (
            manifest.get("actual_red_training_timesteps"),
            manifest.get("red_training_timesteps"),
        ),
        "study config": (manifest.get("study_config_hash"), config.digest()),
        "development": (manifest.get("development"), bool(development)),
        "checkpoint": (manifest.get("checkpoint_sha256"), sha256_file(checkpoint)),
        "policy manifest": (manifest.get("policy_manifest"), phase13_policy_manifest(config)),
        "red PPO": (manifest.get("red_ppo"), config.red_ppo),
        "graph policy": (manifest.get("graph_policy_config"), config.graph_policy),
        "hierarchy": (manifest.get("hierarchy_config"), config.hierarchy),
        "defense": (manifest.get("defense_config"), config.defense),
    }
    if not development:
        expected["required budget"] = (
            manifest.get("actual_red_training_timesteps"),
            config.red_total_timesteps,
        )
    expected_defender_hash = (
        sha256_file(defender_checkpoint) if arm == "adaptive_defender_training" else None
    )
    expected["defender checkpoint"] = (
        manifest.get("defender_checkpoint_sha256"),
        expected_defender_hash,
    )
    if frozen_inputs is not None:
        expected["frozen inputs"] = (
            manifest.get("frozen_inputs_sha256"),
            canonical_digest(frozen_inputs),
        )
    failures = [
        f"{name}: manifest={actual!r}, required={required!r}"
        for name, (actual, required) in expected.items()
        if actual != required
    ]
    if manifest.get("git_dirty") is not False and not allow_unverifiable:
        failures.append("checkpoint was produced from a dirty/unknown tree")
    if not development and git_dirty() is not False and not allow_unverifiable:
        failures.append("canonical evaluation tree is dirty/unknown")
    if failures:
        raise MultiAgentStudyError(
            "Phase 13 red checkpoint provenance mismatch:\n  " + "\n  ".join(failures)
        )


def validate_paired_red_training_isolation(
    static: dict[str, Any], adaptive: dict[str, Any]
) -> None:
    controlled = (
        "study_id",
        "description",
        "algorithm",
        "policy",
        "representation",
        "red_observation_source",
        "red_action_mask_source",
        "defender_observation_source",
        "defender_action_mask_source",
        "policy_manifest",
        "turn_semantics",
        "development",
        "training_seed",
        "red_training_timesteps",
        "actual_red_training_timesteps",
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
        "red_ppo",
        "graph_policy_config",
        "hierarchy_config",
        "defense_config",
        "red_parameter_count",
        "torch_num_threads",
        "environment_seed",
        "base_distribution",
        "base_vulnerability_snapshot_sha256",
        "frozen_inputs_sha256",
    )
    differences = [field for field in controlled if static.get(field) != adaptive.get(field)]
    if differences:
        raise MultiAgentStudyError(
            "paired red arms differ outside defender controller: "
            + ", ".join(differences)
        )
    left = static.get("reset_records", [])
    right = adaptive.get("reset_records", [])
    common = min(len(left), len(right))
    reset_fields = ("profile", "topology_seed", "topology_hash", "episode_seed")
    for index in range(common):
        if any(left[index].get(field) != right[index].get(field) for field in reset_fields):
            raise MultiAgentStudyError("paired topology reset sequence diverged")


def evaluate_red_policy(
    model,
    *,
    arm: str,
    training_seed: int,
    defender_checkpoint: Path,
    config: MultiAgentResearchConfig,
    topology_seeds: tuple[int, ...],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Evaluate frozen red and blue policies over one ordered held-out campaign."""
    if arm not in PHASE13_ARMS:
        raise ValueError(f"unknown Phase 13 arm: {arm}")
    if not topology_seeds or set(topology_seeds) & set(config.topology_splits["train"]):
        raise MultiAgentStudyError("evaluation topology seeds are absent or overlap training")
    defender = load_maskable_model(defender_checkpoint)
    red_before = policy_digest(model)
    blue_before = policy_digest(defender)
    environment = RedAgainstDefenderEnv(
        topology_seeds,
        config.train_profiles,
        defender=defender,
        config=config,
        deterministic_defender=config.deterministic_evaluation,
    )
    guarded = KnowledgeActionGuard(environment)
    episodes: list[dict[str, Any]] = []
    steps: list[dict[str, Any]] = []
    red_mask_checks = blue_mask_checks = reset_count = 0
    first_reset = True
    for evaluation_seed in config.evaluation_episode_seeds:
        set_all_seeds(evaluation_seed)
        model.set_random_seed(evaluation_seed)
        defender.set_random_seed(evaluation_seed + 1)
        for profile in config.train_profiles:
            for topology_seed in topology_seeds:
                red_observation, reset_info = guarded.reset(
                    seed=evaluation_seed if first_reset else None,
                    options={"profile": profile.value, "topology_seed": topology_seed},
                )
                first_reset = False
                reset_count += 1
                episode_index = int(reset_info["episode_index"])
                activated_at_reset = len(reset_info["activated_patches"])
                before_threshold = environment.defender_knowledge.threshold_adjustments
                before_schedules = environment.defender_knowledge.patch_schedules
                before_activations = environment.defender_knowledge.patch_activations
                terminated = truncated = False
                total_reward = native_total_reward = 0.0
                detected_actions = mitigation_blocks = exploit_attempts = failed_actions = 0
                episode_steps = 0
                while not (terminated or truncated):
                    red_mask = guarded.action_masks().copy()
                    if not red_mask.any():
                        raise MultiAgentStudyError("red has no valid AgentKnowledge action")
                    red_action, _ = model.predict(
                        red_observation,
                        action_masks=red_mask,
                        deterministic=config.deterministic_evaluation,
                    )
                    selected = int(np.asarray(red_action).item())
                    red_mask_checks += 1
                    blue_mask_checks += 1
                    if not red_mask[selected]:
                        raise MultiAgentStudyError("red selected a masked evaluation action")
                    red_observation, reward, terminated, truncated, info = guarded.step(
                        selected
                    )
                    event = info["event"]
                    vulnerability = environment.true_topology.vulnerabilities.get(
                        event.action.target
                    )
                    total_reward += float(reward)
                    native_total_reward += float(info["red_native_reward"])
                    detected_actions += int(info["detected"])
                    mitigation_blocks += int(info["mitigation_blocked"])
                    exploit_attempts += int(event.action.type.value == "exploit")
                    failed_actions += int(not event.success)
                    steps.append(
                        {
                            "arm": arm,
                            "training_seed": int(training_seed),
                            "profile": profile.value,
                            "topology_seed": int(topology_seed),
                            "topology_hash": reset_info["topology_hash"],
                            "evaluation_seed": int(evaluation_seed),
                            "episode_index": episode_index,
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
                            "reward": float(reward),
                            "native_reward": float(info["red_native_reward"]),
                            "prerequisites": list(event.prerequisites),
                            "outcomes": list(event.outcomes),
                            "reason": event.reason,
                            "goal_reached": event.goal_reached,
                            "cve_id": vulnerability.id if vulnerability else None,
                            "cvss_base": vulnerability.cvss if vulnerability else None,
                            "detected": info["detected"],
                            "detection_probability": info["detection_probability"],
                            "ids_level": info["ids_level"],
                            "ids_level_name": info["ids_level_name"],
                            "defender_action": info["defender_action"],
                            "defender_action_index": info["defender_action_index"],
                            "defender_action_state_changed": info[
                                "defender_action_state_changed"
                            ],
                            "scheduled_vulnerability": info["scheduled_vulnerability"],
                            "patch_activation_episode": info[
                                "patch_activation_episode"
                            ],
                            "pending_patch_count": info["pending_patch_count"],
                            "active_patch_count": info["active_patch_count"],
                            "mitigation_blocked": info["mitigation_blocked"],
                            "defender_reward": info["defender_reward"],
                        }
                    )
                    episode_steps += 1
                known_nodes = len(environment.knowledge.discovered)
                true_nodes = len(environment.true_topology.nodes)
                detection_rate = detected_actions / episode_steps
                episodes.append(
                    {
                        "arm": arm,
                        "training_seed": int(training_seed),
                        "profile": profile.value,
                        "topology_seed": int(topology_seed),
                        "topology_hash": reset_info["topology_hash"],
                        "evaluation_seed": int(evaluation_seed),
                        "campaign_episode_index": episode_index,
                        "goal_reached": bool(terminated and not truncated),
                        "terminal_reason": "goal" if terminated else "step_limit",
                        "steps_to_goal": episode_steps if terminated else None,
                        "episode_length": episode_steps,
                        "total_reward": total_reward,
                        "native_total_reward": native_total_reward,
                        "known_nodes": known_nodes,
                        "true_nodes": true_nodes,
                        "discovery_coverage": known_nodes / true_nodes,
                        "invalid_mask_selections": 0,
                        "invalid_defender_selections": 0,
                        "failed_actions": failed_actions,
                        "hosts_compromised": len(environment.knowledge.access),
                        "path_events": len(environment.attack_path()),
                        "detected_actions": detected_actions,
                        "detectable_actions": episode_steps,
                        "detection_rate": detection_rate,
                        "evasion_rate": 1.0 - detection_rate,
                        "exploit_attempts": exploit_attempts,
                        "mitigated_exploits": mitigation_blocks,
                        "mitigation_block_rate": mitigation_blocks
                        / max(1, exploit_attempts),
                        "defender_actions": episode_steps,
                        "threshold_adjustments": (
                            environment.defender_knowledge.threshold_adjustments
                            - before_threshold
                        ),
                        "patch_schedules": (
                            environment.defender_knowledge.patch_schedules
                            - before_schedules
                        ),
                        "patch_activations": (
                            environment.defender_knowledge.patch_activations
                            - before_activations
                            + activated_at_reset
                        ),
                    }
                )
    red_after = policy_digest(model)
    blue_after = policy_digest(defender)
    guarded.close()
    if red_before != red_after:
        raise MultiAgentStudyError("red policy changed during frozen evaluation")
    if blue_before != blue_after:
        raise MultiAgentStudyError("blue policy changed during frozen evaluation")
    expected_resets = (
        len(config.train_profiles)
        * len(topology_seeds)
        * len(config.evaluation_episode_seeds)
    )
    if reset_count != expected_resets:
        raise MultiAgentStudyError("evaluation did not cover the complete held-out grid")
    if guarded.invalid_action_selections or environment.interaction.invalid_red_actions:
        raise MultiAgentStudyError("evaluation recorded an invalid red action")
    if environment.interaction.invalid_defender_actions:
        raise MultiAgentStudyError("evaluation recorded an invalid defender action")
    if environment.defender_knowledge.decision_count != len(steps):
        raise MultiAgentStudyError("blue/red evaluation turn accounting differs")
    return (
        episodes,
        steps,
        {
            "gradient_updates": False,
            "red_policy_sha256_before": red_before,
            "red_policy_sha256_after": red_after,
            "defender_policy_sha256_before": blue_before,
            "defender_policy_sha256_after": blue_after,
            "evaluation_reset_count": reset_count,
            "red_knowledge_mask_checks": red_mask_checks,
            "defender_knowledge_mask_checks": blue_mask_checks,
            "invalid_mask_selections": 0,
            "invalid_defender_selections": 0,
            "environment_steps": len(steps),
            "defender_decisions": environment.defender_knowledge.decision_count,
            "turn_semantics": "one_blue_decision_then_one_red_transition",
            "detection_count": environment.interaction.detection_count,
            "mitigation_block_count": environment.interaction.mitigation_block_count,
            "defender_action_counts": dict(
                sorted(environment.interaction.defender_action_counts.items())
            ),
            "reset_trace_sha256": canonical_digest(environment.reset_records),
        },
    )


def _persist_defense_extension(
    *,
    database_metadata: dict[str, Any],
    episodes: list[dict[str, Any]],
    steps: list[dict[str, Any]],
) -> dict[str, int]:
    """Attach Phase 13 telemetry to the base reconstructable evaluation rows."""
    import psycopg

    from rlredteam.storage.postgres_logger import connection_string

    run_id = int(database_metadata["database_evaluation_run_id"])
    experiment_id = int(database_metadata["database_experiment_id"])
    grouped: dict[tuple[str, int, int], list[dict[str, Any]]] = {}
    for step in steps:
        key = (
            str(step["profile"]),
            int(step["topology_seed"]),
            int(step["evaluation_seed"]),
        )
        grouped.setdefault(key, []).append(step)
    try:
        with psycopg.connect(connection_string()) as connection:
            connection.execute(PHASE13_SCHEMA.read_text())
            database_episodes = connection.execute(
                "SELECT id FROM episodes WHERE run_id=%s ORDER BY episode_idx",
                (run_id,),
            ).fetchall()
            if len(database_episodes) != len(episodes):
                raise MultiAgentStudyError("Phase 13 database episode count differs")
            extension_steps = 0
            for (episode_id,), episode in zip(database_episodes, episodes, strict=True):
                connection.execute(
                    """
                    INSERT INTO phase13_defense_episodes (
                        episode_id, native_total_reward, detected_actions,
                        detectable_actions, detection_rate, evasion_rate,
                        exploit_attempts, mitigated_exploits, defender_actions,
                        threshold_adjustments, patch_schedules, patch_activations
                    ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                    """,
                    (
                        episode_id,
                        episode["native_total_reward"],
                        episode["detected_actions"],
                        episode["detectable_actions"],
                        episode["detection_rate"],
                        episode["evasion_rate"],
                        episode["exploit_attempts"],
                        episode["mitigated_exploits"],
                        episode["defender_actions"],
                        episode["threshold_adjustments"],
                        episode["patch_schedules"],
                        episode["patch_activations"],
                    ),
                )
                key = (
                    str(episode["profile"]),
                    int(episode["topology_seed"]),
                    int(episode["evaluation_seed"]),
                )
                source_steps = grouped.get(key, [])
                database_steps = connection.execute(
                    "SELECT id FROM steps WHERE episode_id=%s ORDER BY step_idx",
                    (episode_id,),
                ).fetchall()
                if len(database_steps) != len(source_steps):
                    raise MultiAgentStudyError("Phase 13 database step count differs")
                with connection.cursor() as cursor:
                    cursor.executemany(
                        """
                        INSERT INTO phase13_defense_steps (
                            step_id, native_reward, detected, detection_probability,
                            ids_level, ids_level_name, defender_action,
                            defender_action_index, defender_action_changed,
                            scheduled_vulnerability, patch_activation_episode,
                            pending_patch_count, active_patch_count,
                            mitigation_blocked, defender_reward
                        ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                        """,
                        [
                            (
                                step_id,
                                step["native_reward"],
                                step["detected"],
                                step["detection_probability"],
                                step["ids_level"],
                                step["ids_level_name"],
                                step["defender_action"],
                                step["defender_action_index"],
                                step["defender_action_state_changed"],
                                step["scheduled_vulnerability"],
                                step["patch_activation_episode"],
                                step["pending_patch_count"],
                                step["active_patch_count"],
                                step["mitigation_blocked"],
                                step["defender_reward"],
                            )
                            for (step_id,), step in zip(
                                database_steps, source_steps, strict=True
                            )
                        ],
                    )
                extension_steps += len(source_steps)
            stored_episodes = connection.execute(
                """
                SELECT count(*) FROM phase13_defense_episodes d
                JOIN episodes e ON e.id=d.episode_id WHERE e.run_id=%s
                """,
                (run_id,),
            ).fetchone()[0]
            stored_steps = connection.execute(
                """
                SELECT count(*) FROM phase13_defense_steps d
                JOIN steps s ON s.id=d.step_id
                JOIN episodes e ON e.id=s.episode_id WHERE e.run_id=%s
                """,
                (run_id,),
            ).fetchone()[0]
            if stored_episodes != len(episodes) or stored_steps != extension_steps:
                raise MultiAgentStudyError("Phase 13 database extension is incomplete")
        return {
            "database_defense_episode_count": int(stored_episodes),
            "database_defense_step_count": int(stored_steps),
        }
    except Exception:
        # The base logger commits before this extension starts. Remove only this
        # newly-created experiment so a failed extension never looks complete.
        try:
            with psycopg.connect(connection_string()) as cleanup:
                cleanup.execute("DELETE FROM experiments WHERE id=%s", (experiment_id,))
        except Exception:
            pass
        raise


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
        "phase": "phase13_frozen_adversarial_defender_comparison",
        "split": split_name,
        "gradient_updates": False,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "training_manifest": training_manifest,
        **integrity,
    }
    if postgres:
        compatible_manifest = {**training_manifest, "ppo": training_manifest["red_ppo"]}
        database = persist_evaluation(
            episodes=episodes,
            steps=steps,
            training_manifest=compatible_manifest,
            checkpoint=checkpoint,
            split_name=split_name,
        )
        metadata.update(database)
        metadata.update(
            _persist_defense_extension(
                database_metadata=database,
                episodes=episodes,
                steps=steps,
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
    config: MultiAgentResearchConfig,
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
    rows = []
    for (arm, seed), values in sorted(grouped.items()):
        if len(values) != expected_count:
            raise MultiAgentStudyError(
                f"{arm} seed {seed} has {len(values)} episodes; expected {expected_count}"
            )
        cases = {
            (row["profile"], row["topology_seed"], row["evaluation_seed"])
            for row in values
        }
        if len(cases) != expected_count:
            raise MultiAgentStudyError(f"{arm} seed {seed} has duplicate cases")
        detected = sum(int(row["detected_actions"]) for row in values)
        detectable = sum(int(row["detectable_actions"]) for row in values)
        attempts = sum(int(row["exploit_attempts"]) for row in values)
        mitigated = sum(int(row["mitigated_exploits"]) for row in values)
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
                "detection_rate": detected / detectable,
                "penalized_steps": statistics.fmean(penalized),
                "evasion_rate": 1.0 - detected / detectable,
                "total_reward": statistics.fmean(
                    float(row["total_reward"]) for row in values
                ),
                "native_reward": statistics.fmean(
                    float(row["native_total_reward"]) for row in values
                ),
                "mitigation_block_rate": mitigated / max(1, attempts),
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
    config: MultiAgentResearchConfig,
    *,
    expected_seeds: tuple[int, ...] | None = None,
) -> dict[str, Any]:
    expected_seeds = expected_seeds or config.training_seeds
    by_arm = {
        arm: {
            int(row["training_seed"]): row for row in rows if row["arm"] == arm
        }
        for arm in config.arms
    }
    complete = all(set(by_arm[arm]) == set(expected_seeds) for arm in config.arms)
    protocol = {
        "primary_metrics": list(config.primary_metrics),
        "statistics": config.statistics,
    }
    comparisons = []
    for metric in (*config.primary_metrics, *config.descriptive_metrics):
        arm_a = {
            seed: float(row[metric]) for seed, row in by_arm[config.arms[0]].items()
        }
        arm_b = {
            seed: float(row[metric]) for seed, row in by_arm[config.arms[1]].items()
        }
        comparisons.append(paired_comparison(metric, arm_a, arm_b, protocol).to_dict())
    report = {
        "schema_version": 1,
        "study_id": config.experiment_id,
        "complete": complete,
        "primary_unit": "training_seed",
        "arm_a": config.arms[0],
        "arm_b": config.arms[1],
        "expected_training_seeds": list(expected_seeds),
        "observed_training_seeds": {
            arm: sorted(by_arm[arm]) for arm in config.arms
        },
        "primary_metrics": list(config.primary_metrics),
        "comparisons": comparisons,
    }
    json.dumps(report, allow_nan=False)
    return report


def validate_finite_evidence(
    episodes: list[dict[str, Any]], steps: list[dict[str, Any]]
) -> None:
    for collection in (episodes, steps):
        for row in collection:
            for key, value in row.items():
                if isinstance(value, float) and not math.isfinite(value):
                    raise MultiAgentStudyError(f"non-finite evidence value at {key}")


__all__ = [
    "DEFAULT_FROZEN_INPUTS",
    "DEFAULT_RESULT_ROOT",
    "DEFAULT_RUN_ROOT",
    "MultiAgentStudyError",
    "aggregate_seed_metrics",
    "analyse_seed_metrics",
    "arm_run_name",
    "current_input_manifest",
    "evaluate_red_policy",
    "freeze_inputs",
    "load_frozen_inputs",
    "load_maskable_model",
    "persist_and_write_run_evaluation",
    "split_from_config",
    "train_defender",
    "train_red_arm",
    "validate_finite_evidence",
    "validate_frozen_inputs",
    "validate_paired_red_training_isolation",
    "validate_red_training_manifest",
    "write_study_summary",
]
