"""Training, frozen evaluation and evidence for the Phase 9 curriculum study."""

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
from rlredteam.enterprise.curriculum import (
    PHASE9_ARMS,
    CurriculumResearchConfig,
    StageAuditEnv,
    canonical_digest,
    resolve_schedule,
    stage_distribution_manifest,
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
from rlredteam.enterprise.infrastructure_generalisation import (
    infrastructure_distribution_manifest,
    infrastructure_vulnerability_manifest,
)
from rlredteam.enterprise.onprem import OnPremGeneralisationSplit
from rlredteam.enterprise.profiles import EnterpriseProfileConfig, InfrastructureCurriculumEnv
from rlredteam.enterprise.recurrent import KnowledgeActionGuard
from rlredteam.train import set_all_seeds

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_FROZEN_INPUTS = REPO_ROOT / "configs" / "frozen_curriculum_learning.json"
DEFAULT_RUN_ROOT = REPO_ROOT / "runs" / "advanced-rl-curriculum-v1"
DEFAULT_RESULT_ROOT = REPO_ROOT / "results" / "advanced-rl-curriculum-v1"


class CurriculumStudyError(GeneralisationError):
    """Raised when a Phase 9 protocol or evidence invariant is violated."""


def split_from_config(config: CurriculumResearchConfig) -> OnPremGeneralisationSplit:
    return OnPremGeneralisationSplit(
        train=config.topology_splits["train"],
        validation=config.topology_splits["validation"],
        test=config.topology_splits["test"],
    )


def arm_run_name(config: CurriculumResearchConfig, arm: str, seed: int) -> str:
    if arm not in config.arms:
        raise ValueError(f"unknown Phase 9 arm: {arm}")
    return f"{config.experiment_id}-{arm}-s{int(seed)}"


def current_input_manifest(config: CurriculumResearchConfig) -> dict[str, Any]:
    base = EnterpriseProfileConfig.from_yaml()
    split = split_from_config(config)
    schedule = {arm: stage_distribution_manifest(config, arm) for arm in config.arms}
    base_distribution = infrastructure_distribution_manifest(split, base, config.train_profiles)
    source_files = (
        REPO_ROOT / "src" / "rlredteam" / "enterprise" / "curriculum.py",
        REPO_ROOT / "src" / "rlredteam" / "enterprise" / "curriculum_study.py",
        REPO_ROOT / "src" / "rlredteam" / "enterprise" / "profiles.py",
        REPO_ROOT / "src" / "rlredteam" / "enterprise" / "environment.py",
        REPO_ROOT / "src" / "rlredteam" / "enterprise" / "state.py",
        REPO_ROOT / "scripts" / "run_curriculum_study.py",
    )
    return {
        "schema_version": 1,
        "experiment_id": config.experiment_id,
        "experiment_config_sha256": config.digest(),
        "base_profile_config_sha256": base.digest(),
        "dependency_lock_hash": dependency_lock_hash(),
        "base_distribution_sha256": canonical_digest(base_distribution),
        "schedule_distribution_sha256": canonical_digest(schedule),
        "base_vulnerability_snapshot_sha256": infrastructure_vulnerability_manifest(
            split, base, config.train_profiles
        ),
        "source_sha256": {
            str(path.relative_to(REPO_ROOT)): sha256_file(path) for path in source_files
        },
    }


def freeze_inputs(output: Path, config: CurriculumResearchConfig | None = None) -> dict[str, Any]:
    config = config or CurriculumResearchConfig.from_yaml()
    if git_dirty() is not False:
        raise CurriculumStudyError("refusing to freeze Phase 9 inputs from a dirty tree")
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
        raise CurriculumStudyError("invalid Phase 9 frozen-input manifest")
    return frozen


def validate_frozen_inputs(frozen: dict[str, Any], config: CurriculumResearchConfig) -> None:
    current = current_input_manifest(config)
    expected = {key: frozen.get(key) for key in current}
    if current != expected:
        differences = [key for key in current if current.get(key) != expected.get(key)]
        raise CurriculumStudyError(
            f"Phase 9 frozen inputs differ: {', '.join(sorted(differences))}"
        )
    if git_dirty() is not False:
        raise CurriculumStudyError("canonical Phase 9 execution requires a clean tree")


def _make_model(config: CurriculumResearchConfig, seed: int, env):
    from sb3_contrib import MaskablePPO

    return MaskablePPO(
        "MlpPolicy",
        env,
        seed=seed,
        device="cpu",
        verbose=0,
        policy_kwargs={"net_arch": list(config.policy["policy_layers"])},
        **config.common_ppo,
    )


def _audit_stage_resets(
    records: list[dict[str, Any]],
    possible: dict[str, Any],
    *,
    run_name: str,
) -> None:
    if not records:
        raise CurriculumStudyError(f"stage contains no observed resets: {run_name}")
    expected_profiles = set(possible["profiles"])
    observed_profiles = {item["profile"] for item in records}
    if not observed_profiles or not observed_profiles <= expected_profiles:
        raise CurriculumStudyError(f"stage profile exposure drift: {run_name}")
    for item in records:
        profile = item["profile"]
        topology_seed = str(item["topology_seed"])
        if profile not in expected_profiles or topology_seed not in possible["topologies"][profile]:
            raise CurriculumStudyError(f"stage selected an undeclared case: {run_name}")
        if item["topology_hash"] != possible["topologies"][profile][topology_seed]:
            raise CurriculumStudyError(f"stage topology hash drift: {run_name}")
        if item["profile_config_hash"] != possible["profile_config_sha256"]:
            raise CurriculumStudyError(f"stage profile config drift: {run_name}")


def train_arm(
    output_dir: Path,
    *,
    arm: str,
    training_seed: int,
    config: CurriculumResearchConfig | None = None,
    frozen_inputs: dict[str, Any] | None = None,
    timesteps: int | None = None,
    development: bool = False,
    allow_dirty: bool = False,
) -> dict[str, Any]:
    """Train one direct/curriculum arm with exact staged budget accounting."""
    config = config or CurriculumResearchConfig.from_yaml()
    if arm not in config.arms:
        raise ValueError(f"unknown Phase 9 arm: {arm}")
    seed = int(training_seed)
    budget = int(timesteps or config.total_timesteps)
    if development:
        if seed != config.development_seed:
            raise CurriculumStudyError("development execution must use the excluded seed")
    elif seed not in config.training_seeds:
        raise CurriculumStudyError("canonical training seed is outside the protocol")
    schedule = resolve_schedule(config, arm, total_timesteps=budget)
    dirty = git_dirty()
    if dirty is not False and not (development and allow_dirty):
        raise CurriculumStudyError("training requires a clean tracked tree")
    if not development:
        if frozen_inputs is None:
            raise CurriculumStudyError("canonical training requires frozen inputs")
        validate_frozen_inputs(frozen_inputs, config)

    import torch

    torch.set_num_threads(1)
    set_all_seeds(seed)
    possible_schedule = stage_distribution_manifest(config, arm)
    model = None
    stage_audit = []
    started = time.monotonic()
    for stage, possible in zip(schedule, possible_schedule, strict=True):
        base_env = InfrastructureCurriculumEnv(
            config.topology_splits["train"],
            stage.profiles,
            config=stage.profile_config,
        )
        audited = StageAuditEnv(base_env, stage_name=stage.name)
        guarded = KnowledgeActionGuard(audited)
        environment_seed = seed * 100 + stage.index
        guarded.reset(seed=environment_seed)
        audited.reset_records.clear()
        if model is None:
            model = _make_model(config, seed, guarded)
        else:
            model.set_env(guarded, force_reset=True)
        before = int(model.num_timesteps)
        model.learn(
            total_timesteps=stage.timesteps,
            reset_num_timesteps=stage.index == 1,
        )
        after = int(model.num_timesteps)
        if after - before != stage.timesteps:
            raise CurriculumStudyError(
                f"stage budget drift: {arm}/{stage.name} trained {after - before}"
            )
        if guarded.invalid_action_selections:
            raise CurriculumStudyError(f"invalid training action: {arm}/{stage.name}")
        _audit_stage_resets(
            audited.reset_records,
            possible,
            run_name=f"{arm}/{stage.name}",
        )
        stage_audit.append(
            {
                **{key: value for key, value in possible.items() if key != "topologies"},
                "environment_seed": environment_seed,
                "actual_stage_timesteps": after - before,
                "cumulative_timesteps": after,
                "reset_count": len(audited.reset_records),
                "observed_profile_counts": dict(
                    sorted(
                        {
                            profile: sum(
                                item["profile"] == profile for item in audited.reset_records
                            )
                            for profile in possible["profiles"]
                        }.items()
                    )
                ),
                "exposure_counts": audited.exposure_counts(),
                "reset_trace_sha256": audited.trace_digest(),
            }
        )
        guarded.close()
    assert model is not None
    elapsed = time.monotonic() - started
    if int(model.num_timesteps) != budget:
        raise CurriculumStudyError(
            f"actual training steps {model.num_timesteps} differ from {budget}"
        )
    if not all(
        bool(np.isfinite(parameter.detach().cpu().numpy()).all())
        for parameter in model.policy.parameters()
    ):
        raise CurriculumStudyError("trained policy contains non-finite parameters")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / "model"
    model.save(checkpoint)
    checkpoint = checkpoint.with_suffix(".zip")
    base = EnterpriseProfileConfig.from_yaml()
    split = split_from_config(config)
    run_name = arm_run_name(config, arm, seed)
    manifest = {
        "schema_version": 1,
        "study_id": config.experiment_id,
        "experiment_id": run_name,
        "description": config.description,
        "arm": arm,
        "algorithm": "MaskablePPO",
        "policy": "MlpPolicy",
        "action_mask_source": "AgentKnowledge",
        "schedule_kind": (
            "full_distribution_every_stage"
            if arm == "direct_full_distribution"
            else "fixed_simple_to_complex_curriculum"
        ),
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
        "policy_config": config.policy,
        "parameter_count": sum(parameter.numel() for parameter in model.policy.parameters()),
        "torch_num_threads": torch.get_num_threads(),
        "segment_count": len(schedule),
        "stage_possible_distributions": possible_schedule,
        "stage_audit": stage_audit,
        "base_distribution": infrastructure_distribution_manifest(
            split, base, config.train_profiles
        ),
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
    return manifest


def validate_training_manifest(
    manifest: dict[str, Any],
    checkpoint: Path,
    *,
    arm: str,
    training_seed: int,
    config: CurriculumResearchConfig,
    frozen_inputs: dict[str, Any] | None,
    development: bool = False,
    allow_unverifiable: bool = False,
) -> None:
    base = EnterpriseProfileConfig.from_yaml()
    checks = {
        "study": (manifest.get("study_id"), config.experiment_id),
        "run": (
            manifest.get("experiment_id"),
            arm_run_name(config, arm, training_seed),
        ),
        "arm": (manifest.get("arm"), arm),
        "algorithm": (manifest.get("algorithm"), "MaskablePPO"),
        "training seed": (manifest.get("training_seed"), int(training_seed)),
        "training budget": (
            manifest.get("actual_training_timesteps"),
            manifest.get("training_timesteps"),
        ),
        "study config": (manifest.get("study_config_hash"), config.digest()),
        "base profile config": (
            manifest.get("base_profile_config_hash"),
            base.digest(),
        ),
        "dependency lock": (
            manifest.get("dependency_lock_hash"),
            dependency_lock_hash(),
        ),
        "training split": (
            manifest.get("train_topology_seeds"),
            list(config.topology_splits["train"]),
        ),
        "validation split": (
            manifest.get("validation_topology_seeds"),
            list(config.topology_splits["validation"]),
        ),
        "test split": (
            manifest.get("test_topology_seeds"),
            list(config.topology_splits["test"]),
        ),
        "checkpoint": (manifest.get("checkpoint_sha256"), sha256_file(checkpoint)),
        "development": (manifest.get("development"), bool(development)),
        "stage schedule": (
            manifest.get("stage_possible_distributions"),
            stage_distribution_manifest(config, arm),
        ),
    }
    if frozen_inputs is not None:
        checks["frozen inputs"] = (
            manifest.get("frozen_inputs_sha256"),
            canonical_digest(frozen_inputs),
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
        raise CurriculumStudyError(
            "Phase 9 checkpoint provenance mismatch:\n  " + "\n  ".join(failures)
        )


def validate_paired_training_isolation(direct: dict[str, Any], curriculum: dict[str, Any]) -> None:
    controlled_fields = (
        "study_id",
        "description",
        "algorithm",
        "policy",
        "action_mask_source",
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
        "policy_config",
        "parameter_count",
        "torch_num_threads",
        "segment_count",
        "base_distribution",
        "base_vulnerability_snapshot_sha256",
        "frozen_inputs_sha256",
    )
    differences = [
        field for field in controlled_fields if direct.get(field) != curriculum.get(field)
    ]
    if differences:
        raise CurriculumStudyError(
            f"paired arms differ outside curriculum schedule: {', '.join(differences)}"
        )
    for left, right in zip(
        direct.get("stage_audit", []),
        curriculum.get("stage_audit", []),
        strict=True,
    ):
        for field in (
            "index",
            "name",
            "timesteps",
            "environment_seed",
            "actual_stage_timesteps",
            "cumulative_timesteps",
        ):
            if left.get(field) != right.get(field):
                raise CurriculumStudyError(f"paired stage lifecycle differs at {field}")


def load_model(checkpoint: Path):
    from sb3_contrib import MaskablePPO

    return MaskablePPO.load(checkpoint, device="cpu")


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
    if arm not in PHASE9_ARMS:
        raise ValueError(f"unknown Phase 9 arm: {arm}")
    if not profiles or not topology_seeds or not evaluation_episode_seeds:
        raise CurriculumStudyError("evaluation profiles/topologies/seeds are required")
    if set(topology_seeds) & set(OnPremGeneralisationSplit().train):
        raise CurriculumStudyError("evaluation topology seeds overlap training")
    base_config = EnterpriseProfileConfig.from_yaml()
    episodes: list[dict[str, Any]] = []
    steps: list[dict[str, Any]] = []
    before = policy_digest(model)
    mask_checks = 0
    reset_count = 0
    for profile in profiles:
        for topology_seed in topology_seeds:
            base = InfrastructureCurriculumEnv((topology_seed,), (profile,), config=base_config)
            env = KnowledgeActionGuard(base)
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
                reset_count += 1
                terminated = truncated = False
                total_reward = 0.0
                failed_actions = 0
                episode_steps = 0
                while not (terminated or truncated):
                    mask = env.action_masks().copy()
                    if not mask.any():
                        raise CurriculumStudyError(
                            "no valid AgentKnowledge action before termination"
                        )
                    action, _ = model.predict(
                        observation,
                        action_masks=mask,
                        deterministic=deterministic,
                    )
                    selected = int(np.asarray(action).item())
                    mask_checks += 1
                    if not mask[selected]:
                        raise CurriculumStudyError("policy selected a masked action")
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
        raise CurriculumStudyError("policy parameters changed during frozen evaluation")
    if any(row["invalid_mask_selections"] for row in episodes):
        raise CurriculumStudyError("evaluation recorded an invalid masked selection")
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
        "phase": "phase9_frozen_curriculum_comparison",
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
    config: CurriculumResearchConfig,
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
            raise CurriculumStudyError(
                f"{arm} seed {seed} has {len(values)} episodes; expected {expected_count}"
            )
        keys = {(row["profile"], row["topology_seed"], row["evaluation_seed"]) for row in values}
        if len(keys) != expected_count:
            raise CurriculumStudyError(f"{arm} seed {seed} has duplicate cases")
        penalized = [
            int(row["steps_to_goal"]) if row["goal_reached"] else config.failure_step_penalty
            for row in values
        ]
        rows.append(
            {
                "arm": arm,
                "training_seed": seed,
                "episode_count": len(values),
                "success_rate": statistics.fmean(float(row["goal_reached"]) for row in values),
                "penalized_steps": statistics.fmean(penalized),
                "total_reward": statistics.fmean(float(row["total_reward"]) for row in values),
                "discovery_coverage": statistics.fmean(
                    float(row["discovery_coverage"]) for row in values
                ),
                "failed_actions": statistics.fmean(float(row["failed_actions"]) for row in values),
            }
        )
    return rows


def analyse_seed_metrics(
    rows: list[dict[str, Any]],
    config: CurriculumResearchConfig,
    *,
    expected_seeds: tuple[int, ...] | None = None,
) -> dict[str, Any]:
    expected_seeds = expected_seeds or config.training_seeds
    by_arm = {
        arm: {int(row["training_seed"]): row for row in rows if row["arm"] == arm}
        for arm in config.arms
    }
    complete = all(set(by_arm[arm]) == set(expected_seeds) for arm in config.arms)
    protocol = {
        "primary_metrics": list(config.primary_metrics),
        "statistics": config.statistics,
    }
    comparisons = []
    for metric in (*config.primary_metrics, *config.descriptive_metrics):
        direct = {seed: float(row[metric]) for seed, row in by_arm[config.arms[0]].items()}
        curriculum = {seed: float(row[metric]) for seed, row in by_arm[config.arms[1]].items()}
        comparisons.append(paired_comparison(metric, direct, curriculum, protocol).to_dict())
    report = {
        "schema_version": 1,
        "study_id": config.experiment_id,
        "complete": complete,
        "primary_unit": "training_seed",
        "arm_a": config.arms[0],
        "arm_b": config.arms[1],
        "expected_training_seeds": list(expected_seeds),
        "observed_training_seeds": {arm: sorted(by_arm[arm]) for arm in config.arms},
        "primary_metrics": list(config.primary_metrics),
        "comparisons": comparisons,
    }
    json.dumps(report, allow_nan=False)
    return report


def validate_finite_evidence(episodes: list[dict[str, Any]], steps: list[dict[str, Any]]) -> None:
    for collection in (episodes, steps):
        for row in collection:
            for key, value in row.items():
                if isinstance(value, float) and not math.isfinite(value):
                    raise CurriculumStudyError(f"non-finite evidence value at {key}")


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
        with (output_dir / "summaries" / "seed_metrics.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(seed_metrics[0]))
            writer.writeheader()
            writer.writerows(seed_metrics)
    if report["comparisons"]:
        with (output_dir / "tables" / "statistics.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(report["comparisons"][0]))
            writer.writeheader()
            writer.writerows(report["comparisons"])
