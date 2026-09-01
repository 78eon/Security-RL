"""Knowledge-masked recurrent policy primitives for the Phase 8 study.

The pinned sb3-contrib RecurrentPPO has no action-mask interface.  This module
transports the existing AgentKnowledge-derived mask beside the observation,
keeps it out of learned features, and applies it only to categorical logits.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch as th
import yaml
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from rlredteam.enterprise.onprem import OnPremGeneralisationSplit
from rlredteam.enterprise.profiles import DeploymentProfile, EnterpriseProfileConfig

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RECURRENT_CONFIG = (
    REPO_ROOT / "configs" / "experiments" / "recurrent_policy.yaml"
)
PHASE8_ARMS = ("maskable_ppo", "knowledge_masked_recurrent_ppo")
PHASE8_PROFILES = (
    DeploymentProfile.LEGACY,
    DeploymentProfile.CLOUD,
    DeploymentProfile.HYBRID,
)


def _canonical_digest(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    import hashlib

    return hashlib.sha256(encoded.encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class RecurrentResearchConfig:
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
    common_ppo: dict[str, Any]
    baseline_policy: dict[str, Any]
    recurrent_policy: dict[str, Any]
    primary_metrics: tuple[str, ...]
    descriptive_metrics: tuple[str, ...]
    failure_step_penalty: int
    statistics: dict[str, Any]

    @classmethod
    def from_yaml(cls, path: Path | None = None) -> RecurrentResearchConfig:
        raw = yaml.safe_load((path or DEFAULT_RECURRENT_CONFIG).read_text())["experiment"]
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
                name: tuple(map(int, seeds))
                for name, seeds in raw["topology_splits"].items()
            },
            total_timesteps=int(raw["total_timesteps"]),
            evaluation_episode_seeds=tuple(map(int, raw["evaluation_episode_seeds"])),
            deterministic_evaluation=bool(raw["deterministic_evaluation"]),
            runtime_cap_minutes=int(raw["runtime_cap_minutes"]),
            common_ppo=dict(raw["common_ppo"]),
            baseline_policy=dict(raw["baseline_policy"]),
            recurrent_policy=dict(raw["recurrent_policy"]),
            primary_metrics=tuple(map(str, outcomes["primary_metrics"])),
            descriptive_metrics=tuple(map(str, outcomes["descriptive_metrics"])),
            failure_step_penalty=int(outcomes["failure_step_penalty"]),
            statistics=dict(raw["statistics"]),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if not self.experiment_id or self.protocol_status != "development_then_freeze":
            raise ValueError("Phase 8 experiment identity/status is invalid")
        if self.arms != PHASE8_ARMS:
            raise ValueError(f"Phase 8 arms must be {PHASE8_ARMS}")
        if self.train_profiles != PHASE8_PROFILES:
            raise ValueError("Phase 8 profiles must preserve legacy/cloud/hybrid order")
        if not self.training_seeds or len(set(self.training_seeds)) != len(
            self.training_seeds
        ):
            raise ValueError("training seeds must be non-empty and unique")
        if self.development_seed in self.training_seeds:
            raise ValueError("development seed must be excluded from canonical training")
        if self.total_timesteps <= 0 or self.runtime_cap_minutes <= 0:
            raise ValueError("training budget and runtime cap must be positive")
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
            raise ValueError(
                f"common PPO fields differ: {sorted(set(self.common_ppo) ^ required_ppo)}"
            )
        n_steps = int(self.common_ppo["n_steps"])
        if self.total_timesteps % n_steps or n_steps % int(self.common_ppo["batch_size"]):
            raise ValueError("training budget/batch size must divide complete PPO rollouts")
        expected_split = OnPremGeneralisationSplit()
        expected = {
            name: tuple(getattr(expected_split, name))
            for name in ("train", "validation", "test")
        }
        if self.topology_splits != expected:
            raise ValueError("Phase 8 topology splits differ from the frozen baseline")
        if not self.evaluation_episode_seeds or len(set(self.evaluation_episode_seeds)) != len(
            self.evaluation_episode_seeds
        ):
            raise ValueError("evaluation episode seeds must be non-empty and unique")
        if set(self.baseline_policy) != {"policy_layers"}:
            raise ValueError("baseline policy configuration is incomplete")
        if set(self.recurrent_policy) != {
            "feature_dim",
            "lstm_hidden_size",
            "n_lstm_layers",
            "policy_layers",
        }:
            raise ValueError("recurrent policy configuration is incomplete")
        if any(
            int(self.recurrent_policy[key]) <= 0
            for key in ("feature_dim", "lstm_hidden_size", "n_lstm_layers")
        ):
            raise ValueError("recurrent dimensions must be positive")
        if self.primary_metrics != ("success_rate", "penalized_steps", "total_reward"):
            raise ValueError("primary metric family differs from the preregistration")
        if int(self.statistics.get("family_size", -1)) != len(self.primary_metrics):
            raise ValueError("statistical family size must equal primary metric count")
        if self.failure_step_penalty != EnterpriseProfileConfig.from_yaml().max_steps + 1:
            raise ValueError("failure step penalty must be max_steps + 1")

    def digest(self) -> str:
        return _canonical_digest(asdict(self))


class KnowledgeActionGuard(gym.Wrapper):
    """Fail closed if a study policy ever selects a knowledge-invalid action."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        if not isinstance(env.action_space, spaces.Discrete):
            raise TypeError("knowledge action guard requires a Discrete action space")
        self.action_count = int(env.action_space.n)
        self.invalid_action_selections = 0

    def action_masks(self) -> np.ndarray:
        mask = np.asarray(self.env.action_masks(), dtype=bool)
        if mask.shape != (self.action_count,):
            raise RuntimeError("AgentKnowledge action mask shape changed")
        return mask

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        self.invalid_action_selections = 0
        return self.env.reset(seed=seed, options=options)

    def step(self, action: int):
        selected = int(np.asarray(action).item())
        if not 0 <= selected < self.action_count or not self.action_masks()[selected]:
            self.invalid_action_selections += 1
            raise RuntimeError("policy selected an action outside AgentKnowledge mask")
        return self.env.step(selected)


class KnowledgeMaskObservationWrapper(gym.Wrapper):
    """Carry the knowledge mask with an observation for recurrent rollouts.

    The appended mask is a constraint channel.  The policy feature extractor
    deliberately slices it off before learned representation or LSTM layers.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        if not isinstance(env.observation_space, spaces.Box) or len(
            env.observation_space.shape
        ) != 1:
            raise TypeError("knowledge-mask transport requires a flat Box observation")
        if not isinstance(env.action_space, spaces.Discrete):
            raise TypeError("knowledge-mask transport requires a Discrete action space")
        self.policy_observation_space = env.observation_space
        self.policy_observation_size = int(env.observation_space.shape[0])
        self.action_count = int(env.action_space.n)
        low = np.concatenate(
            [
                np.asarray(env.observation_space.low, dtype=np.float32),
                np.zeros(self.action_count, dtype=np.float32),
            ]
        )
        high = np.concatenate(
            [
                np.asarray(env.observation_space.high, dtype=np.float32),
                np.ones(self.action_count, dtype=np.float32),
            ]
        )
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

    def action_masks(self) -> np.ndarray:
        mask = np.asarray(self.env.action_masks(), dtype=bool)
        if mask.shape != (self.action_count,):
            raise RuntimeError("AgentKnowledge action mask shape changed")
        return mask

    def _augment(self, observation: np.ndarray) -> np.ndarray:
        policy_observation = np.asarray(observation, dtype=np.float32)
        if not self.policy_observation_space.contains(policy_observation):
            raise RuntimeError("base policy observation is outside its frozen space")
        mask = self.action_masks().astype(np.float32, copy=False)
        return np.concatenate([policy_observation, mask]).astype(np.float32, copy=False)

    def split_observation(self, observation: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        value = np.asarray(observation, dtype=np.float32)
        if value.shape != self.observation_space.shape:
            raise ValueError("augmented observation shape is invalid")
        return (
            value[: self.policy_observation_size].copy(),
            value[self.policy_observation_size :].copy(),
        )

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        observation, info = self.env.reset(seed=seed, options=options)
        return self._augment(observation), info

    def step(self, action: int):
        observation, reward, terminated, truncated, info = self.env.step(action)
        return self._augment(observation), reward, terminated, truncated, info


class KnowledgeFeatureExtractor(BaseFeaturesExtractor):
    """Learn features from policy observation values, never from mask values."""

    def __init__(
        self,
        observation_space: spaces.Box,
        *,
        policy_observation_size: int,
        feature_dim: int,
    ) -> None:
        if policy_observation_size <= 0 or feature_dim <= 0:
            raise ValueError("policy observation and feature dimensions must be positive")
        if policy_observation_size >= int(observation_space.shape[0]):
            raise ValueError("augmented observation must contain a non-empty mask")
        super().__init__(observation_space, features_dim=feature_dim)
        self.policy_observation_size = int(policy_observation_size)
        self.network = th.nn.Sequential(
            th.nn.Linear(self.policy_observation_size, feature_dim),
            th.nn.Tanh(),
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.network(observations[..., : self.policy_observation_size])


def _policy_base_class():
    from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy

    return RecurrentActorCriticPolicy


class KnowledgeMaskedRecurrentPolicy(_policy_base_class()):
    """Recurrent actor-critic policy with fail-closed knowledge-only masking."""

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule,
        *,
        feature_dim: int = 128,
        **kwargs,
    ) -> None:
        from sb3_contrib.common.maskable.distributions import (
            make_masked_proba_distribution,
        )

        if not isinstance(observation_space, spaces.Box) or len(observation_space.shape) != 1:
            raise TypeError("recurrent policy requires a flat augmented Box observation")
        if not isinstance(action_space, spaces.Discrete):
            raise TypeError("recurrent policy requires a Discrete action space")
        self.mask_dimension = int(action_space.n)
        self.policy_observation_size = int(observation_space.shape[0]) - self.mask_dimension
        if self.policy_observation_size <= 0:
            raise ValueError("augmented observation does not contain policy features")
        kwargs["features_extractor_class"] = KnowledgeFeatureExtractor
        kwargs["features_extractor_kwargs"] = {
            "policy_observation_size": self.policy_observation_size,
            "feature_dim": int(feature_dim),
        }
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)
        self.action_dist = make_masked_proba_distribution(action_space)

    def _action_masks(
        self, observations: th.Tensor, *, allow_sequence_padding: bool = False
    ) -> th.Tensor:
        masks = observations[..., -self.mask_dimension :] > 0.5
        rows = masks.reshape(-1, self.mask_dimension)
        empty = ~rows.any(dim=1)
        if th.any(empty) and allow_sequence_padding:
            # RecurrentRolloutBuffer pads variable-length sequences with zero
            # observations. RecurrentPPO excludes those rows from every loss;
            # a single placeholder merely keeps Categorical well-defined.
            rows = rows.clone()
            rows[empty, 0] = True
            return rows.reshape(masks.shape)
        if th.any(empty):
            raise RuntimeError("no valid AgentKnowledge action in recurrent observation")
        return masks

    def _get_action_dist_from_latent(self, latent_pi: th.Tensor):
        action_logits = self.action_net(latent_pi)
        return self.action_dist.proba_distribution(action_logits=action_logits)

    def forward(self, obs, lstm_states, episode_starts, deterministic: bool = False):
        from sb3_contrib.common.recurrent.type_aliases import RNNStates

        features = self.extract_features(obs)
        if self.share_features_extractor:
            pi_features = vf_features = features
        else:
            pi_features, vf_features = features
        latent_pi, lstm_states_pi = self._process_sequence(
            pi_features, lstm_states.pi, episode_starts, self.lstm_actor
        )
        if self.lstm_critic is not None:
            latent_vf, lstm_states_vf = self._process_sequence(
                vf_features, lstm_states.vf, episode_starts, self.lstm_critic
            )
        elif self.shared_lstm:
            latent_vf = latent_pi.detach()
            lstm_states_vf = (
                lstm_states_pi[0].detach(),
                lstm_states_pi[1].detach(),
            )
        else:
            latent_vf = self.critic(vf_features)
            lstm_states_vf = lstm_states_pi

        latent_pi = self.mlp_extractor.forward_actor(latent_pi)
        latent_vf = self.mlp_extractor.forward_critic(latent_vf)
        values = self.value_net(latent_vf)
        distribution = self._get_action_dist_from_latent(latent_pi)
        distribution.apply_masking(self._action_masks(obs))
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        actions = actions.reshape((-1, *self.action_space.shape))
        return actions, values, log_prob, RNNStates(lstm_states_pi, lstm_states_vf)

    def get_distribution(self, obs, lstm_states, episode_starts):
        distribution, next_states = super().get_distribution(
            obs, lstm_states, episode_starts
        )
        distribution.apply_masking(self._action_masks(obs))
        return distribution, next_states

    def evaluate_actions(self, obs, actions, lstm_states, episode_starts):
        features = self.extract_features(obs)
        if self.share_features_extractor:
            pi_features = vf_features = features
        else:
            pi_features, vf_features = features
        latent_pi, _ = self._process_sequence(
            pi_features, lstm_states.pi, episode_starts, self.lstm_actor
        )
        if self.lstm_critic is not None:
            latent_vf, _ = self._process_sequence(
                vf_features, lstm_states.vf, episode_starts, self.lstm_critic
            )
        elif self.shared_lstm:
            latent_vf = latent_pi.detach()
        else:
            latent_vf = self.critic(vf_features)

        latent_pi = self.mlp_extractor.forward_actor(latent_pi)
        latent_vf = self.mlp_extractor.forward_critic(latent_vf)
        distribution = self._get_action_dist_from_latent(latent_pi)
        distribution.apply_masking(
            self._action_masks(obs, allow_sequence_padding=True)
        )
        log_prob = distribution.log_prob(actions)
        values = self.value_net(latent_vf)
        return values, log_prob, distribution.entropy()
