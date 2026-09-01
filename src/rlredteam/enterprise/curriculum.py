"""Prospectively fixed curriculum definitions and audit wrappers for Phase 9."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import gymnasium as gym
import yaml

from rlredteam.enterprise.onprem import topology_digest
from rlredteam.enterprise.profiles import (
    DeploymentProfile,
    EnterpriseProfileConfig,
    generate_profile_topology,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CURRICULUM_CONFIG = REPO_ROOT / "configs" / "experiments" / "curriculum_learning.yaml"
PHASE9_ARMS = ("direct_full_distribution", "staged_curriculum")
BOUND_FIELDS = (
    "min_segments",
    "max_segments",
    "min_hosts",
    "max_hosts",
    "min_pivots",
    "max_pivots",
    "min_decoy_services",
    "max_decoy_services",
)


def canonical_digest(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True, slots=True)
class CurriculumStage:
    name: str
    rollouts: int
    profiles: tuple[DeploymentProfile, ...]
    bounds: dict[str, int]

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> CurriculumStage:
        stage = cls(
            name=str(raw["name"]),
            rollouts=int(raw["rollouts"]),
            profiles=tuple(DeploymentProfile(item) for item in raw["profiles"]),
            bounds={name: int(value) for name, value in raw["bounds"].items()},
        )
        stage.validate()
        return stage

    def validate(self) -> None:
        if not self.name or self.rollouts <= 0 or not self.profiles:
            raise ValueError("curriculum stage name, rollouts and profiles are required")
        if len(set(self.profiles)) != len(self.profiles):
            raise ValueError(f"curriculum stage profiles repeat: {self.name}")
        if set(self.bounds) != set(BOUND_FIELDS):
            raise ValueError(f"curriculum stage bounds differ: {self.name}")
        pairs = (
            ("segments", self.bounds["min_segments"], self.bounds["max_segments"], 1, 5),
            ("hosts", self.bounds["min_hosts"], self.bounds["max_hosts"], 10, 60),
            ("pivots", self.bounds["min_pivots"], self.bounds["max_pivots"], 0, 3),
            (
                "decoy services",
                self.bounds["min_decoy_services"],
                self.bounds["max_decoy_services"],
                1,
                10,
            ),
        )
        for label, lower, upper, allowed_lower, allowed_upper in pairs:
            if not allowed_lower <= lower <= upper <= allowed_upper:
                raise ValueError(f"invalid {label} bounds in stage {self.name}")

    def profile_config(self, base: EnterpriseProfileConfig) -> EnterpriseProfileConfig:
        result = replace(base, **self.bounds)
        result.validate()
        return result


@dataclass(frozen=True, slots=True)
class CurriculumResearchConfig:
    experiment_id: str
    description: str
    protocol_status: str
    arms: tuple[str, ...]
    development_seed: int
    training_seeds: tuple[int, ...]
    train_profiles: tuple[DeploymentProfile, ...]
    topology_splits: dict[str, tuple[int, ...]]
    total_timesteps: int
    evaluation_episode_seeds: tuple[int, ...]
    deterministic_evaluation: bool
    runtime_cap_minutes: int
    parallel_training_workers: int
    common_ppo: dict[str, Any]
    policy: dict[str, Any]
    stages: tuple[CurriculumStage, ...]
    primary_metrics: tuple[str, ...]
    descriptive_metrics: tuple[str, ...]
    failure_step_penalty: int
    statistics: dict[str, Any]

    @classmethod
    def from_yaml(cls, path: Path | None = None) -> CurriculumResearchConfig:
        raw = yaml.safe_load((path or DEFAULT_CURRICULUM_CONFIG).read_text())["experiment"]
        outcomes = raw["outcomes"]
        config = cls(
            experiment_id=str(raw["id"]),
            description=str(raw["description"]),
            protocol_status=str(raw["protocol_status"]),
            arms=tuple(map(str, raw["arms"])),
            development_seed=int(raw["development_seed"]),
            training_seeds=tuple(map(int, raw["training_seeds"])),
            train_profiles=tuple(DeploymentProfile(item) for item in raw["train_profiles"]),
            topology_splits={
                name: tuple(map(int, values)) for name, values in raw["topology_splits"].items()
            },
            total_timesteps=int(raw["total_timesteps"]),
            evaluation_episode_seeds=tuple(map(int, raw["evaluation_episode_seeds"])),
            deterministic_evaluation=bool(raw["deterministic_evaluation"]),
            runtime_cap_minutes=int(raw["runtime_cap_minutes"]),
            parallel_training_workers=int(raw["parallel_training_workers"]),
            common_ppo=dict(raw["common_ppo"]),
            policy=dict(raw["policy"]),
            stages=tuple(CurriculumStage.from_dict(item) for item in raw["stages"]),
            primary_metrics=tuple(map(str, outcomes["primary_metrics"])),
            descriptive_metrics=tuple(map(str, outcomes["descriptive_metrics"])),
            failure_step_penalty=int(outcomes["failure_step_penalty"]),
            statistics=dict(raw["statistics"]),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.arms != PHASE9_ARMS:
            raise ValueError("Phase 9 arms or their declared comparison order changed")
        if self.protocol_status != "development_then_freeze":
            raise ValueError("Phase 9 must use development-then-freeze")
        if len(self.training_seeds) != 10 or len(set(self.training_seeds)) != 10:
            raise ValueError("Phase 9 requires ten unique canonical training seeds")
        if self.development_seed in self.training_seeds:
            raise ValueError("development seed must be excluded from canonical training")
        expected_profiles = {
            DeploymentProfile.LEGACY,
            DeploymentProfile.CLOUD,
            DeploymentProfile.HYBRID,
        }
        if set(self.train_profiles) != expected_profiles or len(self.train_profiles) != 3:
            raise ValueError("Phase 9 requires legacy, cloud and hybrid exactly once")
        if set(self.topology_splits) != {"train", "validation", "test"}:
            raise ValueError("Phase 9 topology splits differ")
        split_sets = [set(self.topology_splits[name]) for name in self.topology_splits]
        if any(not values for values in split_sets):
            raise ValueError("Phase 9 topology splits must be non-empty")
        if any(
            split_sets[left] & split_sets[right]
            for left in range(3)
            for right in range(left + 1, 3)
        ):
            raise ValueError("Phase 9 topology splits overlap")
        required_ppo = {
            "learning_rate",
            "n_steps",
            "batch_size",
            "n_epochs",
            "gamma",
            "gae_lambda",
            "clip_range",
            "ent_coef",
            "vf_coef",
            "max_grad_norm",
        }
        if set(self.common_ppo) != required_ppo:
            raise ValueError("Phase 9 PPO configuration fields differ")
        n_steps = int(self.common_ppo["n_steps"])
        if n_steps <= 0 or n_steps % int(self.common_ppo["batch_size"]):
            raise ValueError("Phase 9 PPO rollout/batch sizes are incompatible")
        if set(self.policy) != {"policy_layers"} or not self.policy["policy_layers"]:
            raise ValueError("Phase 9 policy architecture differs")
        if len(self.stages) != 4 or len({stage.name for stage in self.stages}) != 4:
            raise ValueError("Phase 9 requires four uniquely named stages")
        if len({stage.rollouts for stage in self.stages}) != 1:
            raise ValueError("Phase 9 stages must use equal PPO rollout counts")
        if sum(stage.rollouts * n_steps for stage in self.stages) != self.total_timesteps:
            raise ValueError("Phase 9 stage budgets do not equal the total budget")
        if self.total_timesteps <= 0 or self.failure_step_penalty <= 0:
            raise ValueError("Phase 9 budget and failure penalty must be positive")
        if not 1 <= self.parallel_training_workers <= 6:
            raise ValueError("parallel training workers must be between 1 and 6")
        if self.runtime_cap_minutes <= 0:
            raise ValueError("runtime cap must be positive")
        if not self.evaluation_episode_seeds or len(set(self.evaluation_episode_seeds)) != len(
            self.evaluation_episode_seeds
        ):
            raise ValueError("evaluation seeds must be non-empty and unique")
        if len(self.primary_metrics) != int(self.statistics.get("family_size", -1)):
            raise ValueError("statistical family size differs from primary outcomes")
        base = EnterpriseProfileConfig.from_yaml()
        final = self.stages[-1]
        if final.profiles != self.train_profiles:
            raise ValueError("final curriculum stage must use the full profile distribution")
        if any(final.bounds[name] != getattr(base, name) for name in BOUND_FIELDS):
            raise ValueError("final curriculum stage must use the full base bounds")

    @property
    def stage_timesteps(self) -> int:
        return self.stages[0].rollouts * int(self.common_ppo["n_steps"])

    def digest(self) -> str:
        return canonical_digest(asdict(self))

    def scientific_digest(self) -> str:
        payload = asdict(self)
        payload.pop("parallel_training_workers")
        return canonical_digest(payload)


@dataclass(frozen=True, slots=True)
class ResolvedStage:
    index: int
    name: str
    timesteps: int
    profiles: tuple[DeploymentProfile, ...]
    profile_config: EnterpriseProfileConfig


def resolve_schedule(
    config: CurriculumResearchConfig,
    arm: str,
    *,
    total_timesteps: int | None = None,
) -> tuple[ResolvedStage, ...]:
    if arm not in config.arms:
        raise ValueError(f"unknown Phase 9 arm: {arm}")
    budget = int(total_timesteps or config.total_timesteps)
    n_steps = int(config.common_ppo["n_steps"])
    if budget <= 0 or budget % (len(config.stages) * n_steps):
        raise ValueError("Phase 9 budget must contain equal complete stage rollouts")
    stage_budget = budget // len(config.stages)
    base = EnterpriseProfileConfig.from_yaml()
    result = []
    for index, stage in enumerate(config.stages, start=1):
        if arm == "direct_full_distribution":
            profiles = config.train_profiles
            profile_config = base
        else:
            profiles = stage.profiles
            profile_config = stage.profile_config(base)
        result.append(
            ResolvedStage(
                index=index,
                name=stage.name,
                timesteps=stage_budget,
                profiles=profiles,
                profile_config=profile_config,
            )
        )
    return tuple(result)


def stage_distribution_manifest(
    config: CurriculumResearchConfig,
    arm: str,
) -> list[dict[str, Any]]:
    train_seeds = config.topology_splits["train"]
    records = []
    for stage in resolve_schedule(config, arm):
        topologies = {
            profile.value: {
                str(seed): topology_digest(
                    generate_profile_topology(profile, seed, stage.profile_config)
                )
                for seed in train_seeds
            }
            for profile in stage.profiles
        }
        vulnerabilities = {
            profile.value: {
                str(seed): generate_profile_topology(profile, seed, stage.profile_config).to_dict()[
                    "vulnerabilities"
                ]
                for seed in train_seeds
            }
            for profile in stage.profiles
        }
        records.append(
            {
                "index": stage.index,
                "name": stage.name,
                "timesteps": stage.timesteps,
                "profiles": [profile.value for profile in stage.profiles],
                "profile_config_sha256": stage.profile_config.digest(),
                "topologies": topologies,
                "topology_distribution_sha256": canonical_digest(topologies),
                "vulnerability_snapshot_sha256": canonical_digest(vulnerabilities),
            }
        )
    return records


class StageAuditEnv(gym.Wrapper):
    """Record the actual seeded training distribution selected at resets."""

    def __init__(self, env: gym.Env, *, stage_name: str) -> None:
        super().__init__(env)
        self.stage_name = stage_name
        self.reset_records: list[dict[str, Any]] = []

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        observation, info = self.env.reset(seed=seed, options=options)
        self.reset_records.append(
            {
                "stage": self.stage_name,
                "profile": str(info["profile"]),
                "topology_seed": int(info["topology_seed"]),
                "topology_hash": str(info["topology_hash"]),
                "profile_config_hash": str(info["profile_config_hash"]),
            }
        )
        return observation, info

    def action_masks(self):
        return self.env.action_masks()

    def exposure_counts(self) -> dict[str, int]:
        counts = Counter(
            f"{item['profile']}:{item['topology_seed']}" for item in self.reset_records
        )
        return dict(sorted(counts.items()))

    def trace_digest(self) -> str:
        return canonical_digest(self.reset_records)
