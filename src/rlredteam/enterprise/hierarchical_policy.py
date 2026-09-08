"""AgentKnowledge-only factorized policy for the prospective Phase 12 study."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Self

import numpy as np
import torch as th
import yaml
from gymnasium import spaces
from torch import nn

from rlredteam.enterprise.curriculum import canonical_digest
from rlredteam.enterprise.environment import EnterpriseActionType
from rlredteam.enterprise.graph_policy import AgentKnowledgeGraphExtractor
from rlredteam.enterprise.profiles import DeploymentProfile, EnterpriseProfileConfig

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_HIERARCHICAL_CONFIG = REPO_ROOT / "configs/experiments/hierarchical_policy.yaml"
PHASE12_ARMS = ("flat_graph", "hierarchical_graph")
OBJECTIVES = ("discover", "gain_access", "control", "reach_asset")


@dataclass(frozen=True, slots=True)
class HierarchicalResearchConfig:
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
    graph_policy: dict[str, Any]
    hierarchy: dict[str, Any]
    primary_metrics: tuple[str, ...]
    descriptive_metrics: tuple[str, ...]
    failure_step_penalty: int
    statistics: dict[str, Any]

    @classmethod
    def from_yaml(cls, path: Path | None = None) -> Self:
        raw = yaml.safe_load((path or DEFAULT_HIERARCHICAL_CONFIG).read_text())["experiment"]
        outcomes = raw["outcomes"]
        result = cls(
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
            graph_policy=dict(raw["graph_policy"]),
            hierarchy=dict(raw["hierarchy"]),
            primary_metrics=tuple(map(str, outcomes["primary_metrics"])),
            descriptive_metrics=tuple(map(str, outcomes["descriptive_metrics"])),
            failure_step_penalty=int(outcomes["failure_step_penalty"]),
            statistics=dict(raw["statistics"]),
        )
        result.validate()
        return result

    def validate(self) -> None:
        if not self.experiment_id or self.protocol_status != "development_then_freeze":
            raise ValueError("Phase 12 experiment identity/status is invalid")
        if self.arms != PHASE12_ARMS:
            raise ValueError(f"Phase 12 arms must be {PHASE12_ARMS}")
        if self.train_profiles != (
            DeploymentProfile.LEGACY,
            DeploymentProfile.CLOUD,
            DeploymentProfile.HYBRID,
        ):
            raise ValueError("Phase 12 must preserve legacy/cloud/hybrid order")
        if len(self.training_seeds) != 10 or len(set(self.training_seeds)) != 10:
            raise ValueError("Phase 12 requires ten unique canonical training seeds")
        if self.development_seed in self.training_seeds:
            raise ValueError("development seed must be excluded from canonical training")
        if set(self.topology_splits) != {"train", "validation", "test"}:
            raise ValueError("Phase 12 topology splits differ")
        splits = [set(self.topology_splits[name]) for name in self.topology_splits]
        if any(not split for split in splits) or any(
            splits[left] & splits[right]
            for left in range(3)
            for right in range(left + 1, 3)
        ):
            raise ValueError("Phase 12 topology splits are empty or overlap")
        previously_observed = set(range(1, 9021))
        if any(split & previously_observed for split in splits):
            raise ValueError("Phase 12 must use fresh topology seeds")
        required_ppo = {
            "learning_rate", "n_steps", "batch_size", "n_epochs", "gamma",
            "gae_lambda", "clip_range", "ent_coef", "vf_coef", "max_grad_norm",
        }
        if set(self.common_ppo) != required_ppo:
            raise ValueError("Phase 12 PPO configuration fields differ")
        if self.total_timesteps <= 0 or self.total_timesteps % int(self.common_ppo["n_steps"]):
            raise ValueError("Phase 12 budget must contain complete PPO rollouts")
        required_graph = {
            "node_hidden_dim", "message_passing_steps", "feature_dim", "policy_layers",
            "pooling", "adjacency",
        }
        if set(self.graph_policy) != required_graph:
            raise ValueError("Phase 12 graph-policy fields differ")
        if (
            self.graph_policy["pooling"] != "mean_max"
            or self.graph_policy["adjacency"] != "directed_agent_known_binary"
        ):
            raise ValueError("Phase 12 graph semantics differ")
        declared = tuple(map(str, self.hierarchy.get("objectives", ())))
        if declared != OBJECTIVES:
            raise ValueError(f"Phase 12 objectives must be {OBJECTIVES}")
        assigned = [item for objective in OBJECTIVES for item in self.hierarchy.get(objective, ())]
        if len(assigned) != len(set(assigned)) or set(assigned) != {
            item.value for item in EnterpriseActionType
        }:
            raise ValueError("Phase 12 objective mapping must partition all action types")
        if self.hierarchy.get("distribution") != "objective_then_conditional_action":
            raise ValueError("Phase 12 hierarchy distribution differs")
        if self.hierarchy.get("transition_semantics") != "one_concrete_action_per_environment_step":
            raise ValueError("Phase 12 transition semantics differ")
        if self.primary_metrics != ("success_rate", "penalized_steps", "total_reward"):
            raise ValueError("Phase 12 primary metric family differs")
        if int(self.statistics.get("family_size", -1)) != len(self.primary_metrics):
            raise ValueError("Phase 12 statistical family size differs")
        if self.failure_step_penalty != EnterpriseProfileConfig.from_yaml().max_steps + 1:
            raise ValueError("failure step penalty must be max_steps + 1")
        if not 1 <= self.parallel_training_workers <= 6 or self.runtime_cap_minutes <= 0:
            raise ValueError("Phase 12 execution controls are invalid")

    def digest(self) -> str:
        return canonical_digest(asdict(self))

    def scientific_digest(self) -> str:
        payload = asdict(self)
        payload.pop("parallel_training_workers")
        return canonical_digest(payload)


def action_to_objective(config: HierarchicalResearchConfig) -> tuple[int, ...]:
    """Map fixed public action slots to objectives without inspecting a topology."""
    profile = EnterpriseProfileConfig.from_yaml()
    type_to_objective = {
        EnterpriseActionType(action): option
        for option, objective in enumerate(OBJECTIVES)
        for action in config.hierarchy[objective]
    }
    mapping: list[int] = []
    for action_type in EnterpriseActionType:
        count = (
            profile.max_vulnerabilities
            if action_type == EnterpriseActionType.EXPLOIT
            else profile.max_nodes
        )
        mapping.extend([type_to_objective[action_type]] * count)
    expected = 10 * profile.max_nodes + profile.max_vulnerabilities
    if len(mapping) != expected:
        raise ValueError("hierarchical action mapping differs from fixed action catalogue")
    return tuple(mapping)


class HierarchicalLogitHead(nn.Module):
    """Independent learned objective and conditional concrete-action logits."""

    def __init__(self, latent_dim: int, action_dim: int, option_dim: int) -> None:
        super().__init__()
        self.option_head = nn.Linear(latent_dim, option_dim)
        self.action_head = nn.Linear(latent_dim, action_dim)

    def forward(self, latent: th.Tensor) -> th.Tensor:
        return th.cat((self.option_head(latent), self.action_head(latent)), dim=-1)


def _factorized_logits(
    option_logits: th.Tensor,
    action_logits: th.Tensor,
    action_to_option: th.Tensor,
    masks: np.ndarray | th.Tensor | None,
) -> tuple[th.Tensor, th.Tensor]:
    """Return normalized log P(option) + log P(action | option) and valid mask."""
    batch, action_dim = action_logits.shape
    if masks is None:
        valid = th.ones((batch, action_dim), dtype=th.bool, device=action_logits.device)
    else:
        valid = th.as_tensor(masks, dtype=th.bool, device=action_logits.device).reshape(
            batch, action_dim
        )
    if not bool(valid.any(dim=1).all()):
        raise ValueError("hierarchical distribution received a row with no valid action")
    option_dim = option_logits.shape[1]
    option_valid = th.stack(
        [valid[:, action_to_option == option].any(dim=1) for option in range(option_dim)], dim=1
    )
    negative = th.finfo(action_logits.dtype).min
    masked_options = option_logits.masked_fill(~option_valid, negative)
    option_log_probability = th.log_softmax(masked_options, dim=1)
    combined = th.full_like(action_logits, negative)
    for option in range(option_dim):
        members = action_to_option == option
        member_valid = valid[:, members]
        member_logits = action_logits[:, members].masked_fill(~member_valid, negative)
        conditional = th.log_softmax(member_logits, dim=1)
        combined[:, members] = option_log_probability[:, option : option + 1] + conditional
    return combined.masked_fill(~valid, negative), valid


class HierarchicalMaskableDistribution:
    """SB3-compatible categorical implementing P(o|K) P(a|o,K,mask(K))."""

    def __init__(self, action_dim: int, option_dim: int, action_to_option: tuple[int, ...]) -> None:
        from sb3_contrib.common.maskable.distributions import MaskableCategorical

        del MaskableCategorical
        if len(action_to_option) != action_dim or set(action_to_option) != set(range(option_dim)):
            raise ValueError("action-to-objective mapping is incomplete")
        self.action_dim = int(action_dim)
        self.option_dim = int(option_dim)
        self.action_to_option = th.as_tensor(action_to_option, dtype=th.long)
        self._option_logits: th.Tensor | None = None
        self._action_logits: th.Tensor | None = None
        self.distribution = None

    def proba_distribution_net(self, latent_dim: int) -> nn.Module:
        return HierarchicalLogitHead(latent_dim, self.action_dim, self.option_dim)

    def proba_distribution(self, action_logits: th.Tensor) -> Self:
        reshaped = action_logits.reshape(-1, self.option_dim + self.action_dim)
        self._option_logits = reshaped[:, : self.option_dim]
        self._action_logits = reshaped[:, self.option_dim :]
        self.apply_masking(None)
        return self

    def apply_masking(self, masks=None) -> None:
        from sb3_contrib.common.maskable.distributions import MaskableCategorical

        if self._option_logits is None or self._action_logits is None:
            raise RuntimeError("probability distribution has not received logits")
        mapping = self.action_to_option.to(self._action_logits.device)
        logits, valid = _factorized_logits(
            self._option_logits, self._action_logits, mapping, masks
        )
        self.distribution = MaskableCategorical(logits=logits, masks=valid)

    def log_prob(self, actions: th.Tensor) -> th.Tensor:
        return self.distribution.log_prob(actions)

    def entropy(self) -> th.Tensor:
        return self.distribution.entropy()

    def sample(self) -> th.Tensor:
        return self.distribution.sample()

    def mode(self) -> th.Tensor:
        return th.argmax(self.distribution.probs, dim=1)

    def get_actions(self, deterministic: bool = False) -> th.Tensor:
        return self.mode() if deterministic else self.sample()

    def actions_from_params(
        self, action_logits: th.Tensor, deterministic: bool = False
    ) -> th.Tensor:
        self.proba_distribution(action_logits)
        return self.get_actions(deterministic=deterministic)

    def log_prob_from_params(self, action_logits: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
        actions = self.actions_from_params(action_logits)
        return actions, self.log_prob(actions)


def hierarchical_policy_class():
    """Build lazily so importing configuration does not require sb3-contrib."""
    from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy

    class HierarchicalMaskableActorCriticPolicy(MaskableActorCriticPolicy):
        def __init__(
            self,
            observation_space,
            action_space,
            lr_schedule,
            *,
            action_to_option: tuple[int, ...],
            option_dim: int,
            **kwargs,
        ) -> None:
            object.__setattr__(self, "_phase12_action_to_option", tuple(action_to_option))
            object.__setattr__(self, "_phase12_option_dim", int(option_dim))
            super().__init__(observation_space, action_space, lr_schedule, **kwargs)

        def _build(self, lr_schedule) -> None:
            if not isinstance(self.action_space, spaces.Discrete):
                raise TypeError("hierarchical policy requires a Discrete action space")
            self.action_dist = HierarchicalMaskableDistribution(
                int(self.action_space.n),
                self._phase12_option_dim,
                self._phase12_action_to_option,
            )
            super()._build(lr_schedule)

        def _get_constructor_parameters(self) -> dict[str, Any]:
            data = super()._get_constructor_parameters()
            data.update(
                action_to_option=self._phase12_action_to_option,
                option_dim=self._phase12_option_dim,
            )
            return data

    HierarchicalMaskableActorCriticPolicy.__name__ = "HierarchicalMaskableActorCriticPolicy"
    HierarchicalMaskableActorCriticPolicy.__qualname__ = "HierarchicalMaskableActorCriticPolicy"
    HierarchicalMaskableActorCriticPolicy.__module__ = __name__
    return HierarchicalMaskableActorCriticPolicy


HierarchicalMaskableActorCriticPolicy = hierarchical_policy_class()


def phase12_policy(config: HierarchicalResearchConfig, arm: str):
    if arm not in PHASE12_ARMS:
        raise ValueError(f"unknown Phase 12 arm: {arm}")
    return "MlpPolicy" if arm == "flat_graph" else HierarchicalMaskableActorCriticPolicy


def phase12_policy_kwargs(config: HierarchicalResearchConfig, arm: str) -> dict[str, Any]:
    if arm not in PHASE12_ARMS:
        raise ValueError(f"unknown Phase 12 arm: {arm}")
    graph = config.graph_policy
    kwargs: dict[str, Any] = {
        "features_extractor_class": AgentKnowledgeGraphExtractor,
        "features_extractor_kwargs": {
            "max_nodes": EnterpriseProfileConfig.from_yaml().max_nodes,
            "node_hidden_dim": int(graph["node_hidden_dim"]),
            "message_passing_steps": int(graph["message_passing_steps"]),
            "feature_dim": int(graph["feature_dim"]),
        },
        "net_arch": list(graph["policy_layers"]),
    }
    if arm == "hierarchical_graph":
        kwargs.update(action_to_option=action_to_objective(config), option_dim=len(OBJECTIVES))
    return kwargs


def fixed_action_catalogue() -> tuple[str, ...]:
    """Evidence helper: build the catalogue without any enterprise topology."""
    profile = EnterpriseProfileConfig.from_yaml()
    non_exploit = [kind for kind in EnterpriseActionType if kind != EnterpriseActionType.EXPLOIT]
    return tuple(
        [
            f"{kind.value}:node_slot_{slot}"
            for kind in non_exploit
            for slot in range(profile.max_nodes)
        ]
        + [f"exploit:vulnerability_slot_{slot}" for slot in range(profile.max_vulnerabilities)]
    )


_profile = EnterpriseProfileConfig.from_yaml()
assert len(fixed_action_catalogue()) == 10 * _profile.max_nodes + _profile.max_vulnerabilities


__all__ = [
    "DEFAULT_HIERARCHICAL_CONFIG",
    "HierarchicalMaskableActorCriticPolicy",
    "HierarchicalMaskableDistribution",
    "HierarchicalResearchConfig",
    "OBJECTIVES",
    "PHASE12_ARMS",
    "action_to_objective",
    "fixed_action_catalogue",
    "phase12_policy",
    "phase12_policy_kwargs",
]
