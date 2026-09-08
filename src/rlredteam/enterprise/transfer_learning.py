"""Prospective controls for Phase 11 legacy/cloud-to-hybrid transfer."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from rlredteam.enterprise.curriculum import canonical_digest
from rlredteam.enterprise.model import NodeType
from rlredteam.enterprise.onprem import topology_digest
from rlredteam.enterprise.profiles import (
    DeploymentProfile,
    EnterpriseProfileConfig,
    generate_profile_topology,
)
from rlredteam.enterprise.state import Observation

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_TRANSFER_CONFIG = REPO_ROOT / "configs/experiments/transfer_learning.yaml"
PHASE11_ARMS = ("scratch_hybrid", "transfer_legacy_cloud_to_hybrid")


@dataclass(frozen=True, slots=True)
class TransferResearchConfig:
    """Controlled source-pretraining versus target-scratch experiment."""

    experiment_id: str
    description: str
    protocol_status: str
    arms: tuple[str, ...]
    development_seed: int
    training_seeds: tuple[int, ...]
    source_profiles: tuple[DeploymentProfile, ...]
    target_profiles: tuple[DeploymentProfile, ...]
    topology_splits: dict[str, tuple[int, ...]]
    source_pretrain_timesteps: int
    target_adaptation_timesteps: int
    evaluation_episode_seeds: tuple[int, ...]
    deterministic_evaluation: bool
    runtime_cap_minutes: int
    parallel_training_workers: int
    common_ppo: dict[str, Any]
    graph_policy: dict[str, Any]
    primary_metrics: tuple[str, ...]
    descriptive_metrics: tuple[str, ...]
    failure_step_penalty: int
    statistics: dict[str, Any]

    @property
    def train_profiles(self) -> tuple[DeploymentProfile, ...]:
        """Compatibility alias: evaluation aggregation uses the target profile set."""
        return self.target_profiles

    @classmethod
    def from_yaml(cls, path: Path | None = None) -> TransferResearchConfig:
        raw = yaml.safe_load((path or DEFAULT_TRANSFER_CONFIG).read_text())["experiment"]
        outcomes = raw["outcomes"]
        config = cls(
            experiment_id=str(raw["id"]),
            description=str(raw["description"]),
            protocol_status=str(raw["protocol_status"]),
            arms=tuple(map(str, raw["arms"])),
            development_seed=int(raw["development_seed"]),
            training_seeds=tuple(map(int, raw["training_seeds"])),
            source_profiles=tuple(DeploymentProfile(item) for item in raw["source_profiles"]),
            target_profiles=tuple(DeploymentProfile(item) for item in raw["target_profiles"]),
            topology_splits={
                name: tuple(map(int, values)) for name, values in raw["topology_splits"].items()
            },
            source_pretrain_timesteps=int(raw["source_pretrain_timesteps"]),
            target_adaptation_timesteps=int(raw["target_adaptation_timesteps"]),
            evaluation_episode_seeds=tuple(map(int, raw["evaluation_episode_seeds"])),
            deterministic_evaluation=bool(raw["deterministic_evaluation"]),
            runtime_cap_minutes=int(raw["runtime_cap_minutes"]),
            parallel_training_workers=int(raw["parallel_training_workers"]),
            common_ppo=dict(raw["common_ppo"]),
            graph_policy=dict(raw["graph_policy"]),
            primary_metrics=tuple(map(str, outcomes["primary_metrics"])),
            descriptive_metrics=tuple(map(str, outcomes["descriptive_metrics"])),
            failure_step_penalty=int(outcomes["failure_step_penalty"]),
            statistics=dict(raw["statistics"]),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if not self.experiment_id or self.protocol_status != "development_then_freeze":
            raise ValueError("Phase 11 experiment identity/status is invalid")
        if self.arms != PHASE11_ARMS:
            raise ValueError(f"Phase 11 arms must be {PHASE11_ARMS}")
        if self.source_profiles != (DeploymentProfile.LEGACY, DeploymentProfile.CLOUD):
            raise ValueError("Phase 11 source profiles must be legacy then cloud")
        if self.target_profiles != (DeploymentProfile.HYBRID,):
            raise ValueError("Phase 11 target profile must be hybrid only")
        if len(self.training_seeds) != 10 or len(set(self.training_seeds)) != 10:
            raise ValueError("Phase 11 requires ten unique canonical training seeds")
        if self.development_seed in self.training_seeds:
            raise ValueError("development seed must be excluded from canonical training")
        expected_splits = {"source_train", "target_train", "validation", "test"}
        if set(self.topology_splits) != expected_splits:
            raise ValueError("Phase 11 topology split names differ")
        split_sets = [set(self.topology_splits[name]) for name in expected_splits]
        if any(not values for values in split_sets):
            raise ValueError("Phase 11 topology splits must be non-empty")
        if any(
            split_sets[left] & split_sets[right]
            for left in range(4)
            for right in range(left + 1, 4)
        ):
            raise ValueError("Phase 11 topology splits overlap")
        previously_observed = (
            set(range(1, 61))
            | set(range(1001, 1021))
            | set(range(2001, 2021))
            | set(range(3001, 3061))
            | set(range(4001, 4021))
            | set(range(5001, 5021))
        )
        if any(values & previously_observed for values in split_sets):
            raise ValueError("Phase 11 reuses a topology seed observed in an earlier phase")
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
            raise ValueError("Phase 11 PPO configuration fields differ")
        n_steps = int(self.common_ppo["n_steps"])
        for label, budget in (
            ("source", self.source_pretrain_timesteps),
            ("target", self.target_adaptation_timesteps),
        ):
            if budget <= 0 or budget % n_steps:
                raise ValueError(f"Phase 11 {label} budget must contain complete PPO rollouts")
        if n_steps % int(self.common_ppo["batch_size"]):
            raise ValueError("Phase 11 PPO rollout/batch sizes are incompatible")
        required_graph = {
            "node_hidden_dim",
            "message_passing_steps",
            "feature_dim",
            "policy_layers",
            "pooling",
            "adjacency",
        }
        if set(self.graph_policy) != required_graph:
            raise ValueError("Phase 11 graph policy fields differ")
        if self.graph_policy["pooling"] != "mean_max":
            raise ValueError("Phase 11 graph pooling differs")
        if self.graph_policy["adjacency"] != "directed_agent_known_binary":
            raise ValueError("Phase 11 adjacency definition differs")
        if not 1 <= self.parallel_training_workers <= 6 or self.runtime_cap_minutes <= 0:
            raise ValueError("Phase 11 execution controls are invalid")
        if self.primary_metrics != ("success_rate", "penalized_steps", "total_reward"):
            raise ValueError("Phase 11 primary metric family differs")
        if int(self.statistics.get("family_size", -1)) != len(self.primary_metrics):
            raise ValueError("Phase 11 statistical family size differs")
        if self.failure_step_penalty != EnterpriseProfileConfig.from_yaml().max_steps + 1:
            raise ValueError("failure step penalty must be max_steps + 1")

    def digest(self) -> str:
        return canonical_digest(asdict(self))

    def scientific_digest(self) -> str:
        payload = asdict(self)
        payload.pop("parallel_training_workers")
        return canonical_digest(payload)


def observation_schema() -> dict[str, Any]:
    """Declare the only tensor fields visible to either Phase 11 policy."""
    profile = EnterpriseProfileConfig.from_yaml()
    feature_count = Observation.feature_count(tuple(NodeType))
    return {
        "source": "AgentKnowledge",
        "max_nodes": profile.max_nodes,
        "node_feature_count": feature_count,
        "node_values": profile.max_nodes * feature_count,
        "adjacency_values": profile.max_nodes * profile.max_nodes,
        "step_values": 1,
        "observation_values": profile.max_nodes * feature_count
        + profile.max_nodes * profile.max_nodes
        + 1,
        "node_types": "one_hot_discovered",
        "adjacency": "directed_agent_known_binary",
        "hidden_topology_fields": 0,
    }


def transfer_distribution_manifest(
    config: TransferResearchConfig, profile_config: EnterpriseProfileConfig | None = None
) -> dict[str, Any]:
    """Hash every declared source or target case without exposing it to a policy."""
    profile_config = profile_config or EnterpriseProfileConfig.from_yaml()
    profiles_by_split = {
        "source_train": config.source_profiles,
        "target_train": config.target_profiles,
        "validation": config.target_profiles,
        "test": config.target_profiles,
    }
    return {
        split_name: {
            profile.value: {
                str(seed): topology_digest(generate_profile_topology(profile, seed, profile_config))
                for seed in config.topology_splits[split_name]
            }
            for profile in profiles_by_split[split_name]
        }
        for split_name in ("source_train", "target_train", "validation", "test")
    }


def transfer_vulnerability_manifest(
    config: TransferResearchConfig, profile_config: EnterpriseProfileConfig | None = None
) -> str:
    """Hash the synthetic vulnerability records for the full declared protocol."""
    profile_config = profile_config or EnterpriseProfileConfig.from_yaml()
    profiles_by_split = {
        "source_train": config.source_profiles,
        "target_train": config.target_profiles,
        "validation": config.target_profiles,
        "test": config.target_profiles,
    }
    records = {
        split_name: {
            profile.value: {
                str(seed): generate_profile_topology(profile, seed, profile_config).to_dict()[
                    "vulnerabilities"
                ]
                for seed in config.topology_splits[split_name]
            }
            for profile in profiles_by_split[split_name]
        }
        for split_name in profiles_by_split
    }
    return canonical_digest({"source": "synthetic-enterprise-profiles-v1", "records": records})
