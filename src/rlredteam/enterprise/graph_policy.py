"""AgentKnowledge-only graph representation for the prospective Phase 10 study."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch as th
import yaml
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from rlredteam.enterprise.curriculum import canonical_digest
from rlredteam.enterprise.model import NodeType
from rlredteam.enterprise.profiles import DeploymentProfile, EnterpriseProfileConfig
from rlredteam.enterprise.state import Observation

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_GRAPH_POLICY_CONFIG = REPO_ROOT / "configs/experiments/graph_policy.yaml"
PHASE10_ARMS = ("flat_mlp", "knowledge_graph_gnn")


@dataclass(frozen=True, slots=True)
class GraphPolicyResearchConfig:
    """Prospectively controlled MLP-versus-graph-policy experiment."""

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
    baseline_policy: dict[str, Any]
    graph_policy: dict[str, Any]
    primary_metrics: tuple[str, ...]
    descriptive_metrics: tuple[str, ...]
    failure_step_penalty: int
    statistics: dict[str, Any]

    @classmethod
    def from_yaml(cls, path: Path | None = None) -> GraphPolicyResearchConfig:
        raw = yaml.safe_load((path or DEFAULT_GRAPH_POLICY_CONFIG).read_text())["experiment"]
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
            baseline_policy=dict(raw["baseline_policy"]),
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
            raise ValueError("Phase 10 experiment identity/status is invalid")
        if self.arms != PHASE10_ARMS:
            raise ValueError(f"Phase 10 arms must be {PHASE10_ARMS}")
        if self.train_profiles != (
            DeploymentProfile.LEGACY,
            DeploymentProfile.CLOUD,
            DeploymentProfile.HYBRID,
        ):
            raise ValueError("Phase 10 must preserve legacy/cloud/hybrid order")
        if len(self.training_seeds) != 10 or len(set(self.training_seeds)) != 10:
            raise ValueError("Phase 10 requires ten unique canonical training seeds")
        if self.development_seed in self.training_seeds:
            raise ValueError("development seed must be excluded from canonical training")
        if set(self.topology_splits) != {"train", "validation", "test"}:
            raise ValueError("Phase 10 topology splits differ")
        split_sets = [set(self.topology_splits[name]) for name in self.topology_splits]
        if any(not values for values in split_sets):
            raise ValueError("Phase 10 topology splits must be non-empty")
        if any(
            split_sets[left] & split_sets[right]
            for left in range(3)
            for right in range(left + 1, 3)
        ):
            raise ValueError("Phase 10 topology splits overlap")
        previously_observed = set(range(1, 61)) | set(range(1001, 1021)) | set(range(2001, 2021))
        if any(values & previously_observed for values in split_sets):
            raise ValueError("Phase 10 must use topology seeds not observed in earlier phases")
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
            raise ValueError("Phase 10 PPO configuration fields differ")
        n_steps = int(self.common_ppo["n_steps"])
        if self.total_timesteps <= 0 or self.total_timesteps % n_steps:
            raise ValueError("Phase 10 budget must contain complete PPO rollouts")
        if n_steps % int(self.common_ppo["batch_size"]):
            raise ValueError("Phase 10 PPO rollout/batch sizes are incompatible")
        if set(self.baseline_policy) != {"policy_layers"}:
            raise ValueError("Phase 10 baseline policy fields differ")
        required_graph = {
            "node_hidden_dim",
            "message_passing_steps",
            "feature_dim",
            "policy_layers",
            "pooling",
            "adjacency",
        }
        if set(self.graph_policy) != required_graph:
            raise ValueError("Phase 10 graph policy fields differ")
        if any(
            int(self.graph_policy[field]) <= 0
            for field in ("node_hidden_dim", "message_passing_steps", "feature_dim")
        ):
            raise ValueError("Phase 10 graph dimensions must be positive")
        if self.graph_policy["pooling"] != "mean_max":
            raise ValueError("Phase 10 graph pooling differs")
        if self.graph_policy["adjacency"] != "directed_agent_known_binary":
            raise ValueError("Phase 10 adjacency definition differs")
        if not 1 <= self.parallel_training_workers <= 6 or self.runtime_cap_minutes <= 0:
            raise ValueError("Phase 10 execution controls are invalid")
        if self.primary_metrics != ("success_rate", "penalized_steps", "total_reward"):
            raise ValueError("Phase 10 primary metric family differs")
        if int(self.statistics.get("family_size", -1)) != len(self.primary_metrics):
            raise ValueError("Phase 10 statistical family size differs")
        if self.failure_step_penalty != EnterpriseProfileConfig.from_yaml().max_steps + 1:
            raise ValueError("failure step penalty must be max_steps + 1")

    def digest(self) -> str:
        return canonical_digest(asdict(self))

    def scientific_digest(self) -> str:
        payload = asdict(self)
        payload.pop("parallel_training_workers")
        return canonical_digest(payload)


class AgentKnowledgeGraphExtractor(BaseFeaturesExtractor):
    """Message-passing encoder over only the graph encoded in an observation.

    The fixed flat observation is parsed into node features, the directed
    binary adjacency matrix of discovered edges, and the normalized episode
    step. No environment reference or hidden topology object is accepted.
    """

    def __init__(
        self,
        observation_space: spaces.Box,
        *,
        max_nodes: int,
        node_hidden_dim: int = 64,
        message_passing_steps: int = 2,
        feature_dim: int = 128,
    ) -> None:
        if not isinstance(observation_space, spaces.Box) or len(observation_space.shape) != 1:
            raise TypeError("graph extractor requires a flat Box observation")
        if min(max_nodes, node_hidden_dim, message_passing_steps, feature_dim) <= 0:
            raise ValueError("graph extractor dimensions must be positive")
        observation_size = int(observation_space.shape[0])
        remaining = observation_size - max_nodes * max_nodes - 1
        if remaining <= 0 or remaining % max_nodes:
            raise ValueError("observation does not contain fixed node/adjacency/step sections")
        node_feature_dim = remaining // max_nodes
        expected_feature_dim = Observation.feature_count(tuple(NodeType))
        if node_feature_dim != expected_feature_dim:
            raise ValueError(
                f"node feature width {node_feature_dim} differs from {expected_feature_dim}"
            )
        super().__init__(observation_space, features_dim=feature_dim)
        self.max_nodes = int(max_nodes)
        self.node_feature_dim = int(node_feature_dim)
        self.message_passing_steps = int(message_passing_steps)
        self.node_projection = th.nn.Linear(node_feature_dim, node_hidden_dim)
        self.self_projection = th.nn.Linear(node_hidden_dim, node_hidden_dim)
        self.neighbor_projection = th.nn.Linear(node_hidden_dim, node_hidden_dim, bias=False)
        self.graph_projection = th.nn.Sequential(
            th.nn.Linear(node_hidden_dim * 2 + 1, feature_dim),
            th.nn.Tanh(),
        )

    def split_observation(self, observations: th.Tensor) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
        node_values = self.max_nodes * self.node_feature_dim
        adjacency_values = self.max_nodes * self.max_nodes
        nodes = observations[..., :node_values].reshape(
            *observations.shape[:-1], self.max_nodes, self.node_feature_dim
        )
        adjacency = observations[..., node_values : node_values + adjacency_values].reshape(
            *observations.shape[:-1], self.max_nodes, self.max_nodes
        )
        step = observations[..., -1:]
        return nodes, adjacency, step

    def forward(self, observations: th.Tensor) -> th.Tensor:
        nodes, adjacency, step = self.split_observation(observations)
        present = nodes[..., :1] > 0.5
        hidden = th.tanh(self.node_projection(nodes)) * present
        degree = adjacency.sum(dim=-1, keepdim=True).clamp_min(1.0)
        normalized = adjacency / degree
        for _ in range(self.message_passing_steps):
            messages = th.matmul(normalized, hidden)
            hidden = (
                th.tanh(self.self_projection(hidden) + self.neighbor_projection(messages)) * present
            )
        denominator = present.sum(dim=-2).clamp_min(1).to(hidden.dtype)
        mean_pool = hidden.sum(dim=-2) / denominator
        minimum = th.finfo(hidden.dtype).min
        max_pool = hidden.masked_fill(~present, minimum).max(dim=-2).values
        any_present = present.any(dim=-2)
        max_pool = th.where(any_present, max_pool, th.zeros_like(max_pool))
        return self.graph_projection(th.cat((mean_pool, max_pool, step), dim=-1))


def phase10_policy_kwargs(config: GraphPolicyResearchConfig, arm: str) -> dict[str, Any]:
    """Return the declared policy kwargs for one Phase 10 arm."""
    if arm == "flat_mlp":
        return {"net_arch": list(config.baseline_policy["policy_layers"])}
    if arm != "knowledge_graph_gnn":
        raise ValueError(f"unknown Phase 10 arm: {arm}")
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
