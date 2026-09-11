"""Simulation-only red-blue environments for the prospective Phase 13 study."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol, Self

import gymnasium as gym
import numpy as np
import yaml
from gymnasium import spaces

from rlredteam.enterprise.curriculum import canonical_digest
from rlredteam.enterprise.defender_state import (
    LOWER_IDS_ACTION,
    MONITOR_ACTION,
    PATCH_ACTION_OFFSET,
    RAISE_IDS_ACTION,
    DefenderDecision,
    DefenderKnowledge,
    DefenderObservation,
    defender_action_count,
    defender_observation_schema,
)
from rlredteam.enterprise.environment import (
    EnterpriseAction,
    EnterpriseActionType,
    EnterpriseCyberEnv,
)
from rlredteam.enterprise.hierarchical_policy import (
    OBJECTIVES,
    action_to_objective,
    phase12_policy,
    phase12_policy_kwargs,
)
from rlredteam.enterprise.onprem import OnPremGeneralisationSplit, topology_digest
from rlredteam.enterprise.profiles import (
    DeploymentProfile,
    EnterpriseProfileConfig,
    generate_profile_topology,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MULTIAGENT_CONFIG = REPO_ROOT / "configs/experiments/multiagent_defense.yaml"
PHASE13_ARMS = ("static_defender_training", "adaptive_defender_training")


class MultiAgentStudyError(RuntimeError):
    """Raised when a Phase 13 interaction or protocol invariant fails."""


@dataclass(frozen=True, slots=True)
class MultiAgentResearchConfig:
    experiment_id: str
    description: str
    protocol_status: str
    arms: tuple[str, ...]
    development_seed: int
    training_seeds: tuple[int, ...]
    train_profiles: tuple[DeploymentProfile, ...]
    topology_splits: dict[str, tuple[int, ...]]
    red_total_timesteps: int
    defender_total_timesteps: int
    evaluation_episode_seeds: tuple[int, ...]
    deterministic_evaluation: bool
    runtime_cap_minutes: int
    parallel_training_workers: int
    red_ppo: dict[str, Any]
    defender_ppo: dict[str, Any]
    graph_policy: dict[str, Any]
    hierarchy: dict[str, Any]
    defense: dict[str, Any]
    primary_metrics: tuple[str, ...]
    descriptive_metrics: tuple[str, ...]
    failure_step_penalty: int
    statistics: dict[str, Any]

    @classmethod
    def from_yaml(cls, path: Path | None = None) -> Self:
        raw = yaml.safe_load((path or DEFAULT_MULTIAGENT_CONFIG).read_text())["experiment"]
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
            red_total_timesteps=int(raw["red_total_timesteps"]),
            defender_total_timesteps=int(raw["defender_total_timesteps"]),
            evaluation_episode_seeds=tuple(map(int, raw["evaluation_episode_seeds"])),
            deterministic_evaluation=bool(raw["deterministic_evaluation"]),
            runtime_cap_minutes=int(raw["runtime_cap_minutes"]),
            parallel_training_workers=int(raw["parallel_training_workers"]),
            red_ppo=dict(raw["red_ppo"]),
            defender_ppo=dict(raw["defender_ppo"]),
            graph_policy=dict(raw["graph_policy"]),
            hierarchy=dict(raw["hierarchy"]),
            defense=dict(raw["defense"]),
            primary_metrics=tuple(map(str, outcomes["primary_metrics"])),
            descriptive_metrics=tuple(map(str, outcomes["descriptive_metrics"])),
            failure_step_penalty=int(outcomes["failure_step_penalty"]),
            statistics=dict(raw["statistics"]),
        )
        result.validate()
        return result

    def validate(self) -> None:
        if not self.experiment_id or self.protocol_status != "development_then_freeze":
            raise ValueError("Phase 13 experiment identity/status is invalid")
        if self.arms != PHASE13_ARMS:
            raise ValueError(f"Phase 13 arms must be {PHASE13_ARMS}")
        if self.train_profiles != (
            DeploymentProfile.LEGACY,
            DeploymentProfile.CLOUD,
            DeploymentProfile.HYBRID,
        ):
            raise ValueError("Phase 13 must preserve legacy/cloud/hybrid order")
        if len(self.training_seeds) != 10 or len(set(self.training_seeds)) != 10:
            raise ValueError("Phase 13 requires ten unique canonical training seeds")
        if self.development_seed in self.training_seeds:
            raise ValueError("development seed must be excluded from canonical training")
        if set(self.topology_splits) != {"train", "validation", "test"}:
            raise ValueError("Phase 13 topology splits differ")
        splits = [set(self.topology_splits[name]) for name in self.topology_splits]
        if any(not split for split in splits) or any(
            splits[left] & splits[right]
            for left in range(3)
            for right in range(left + 1, 3)
        ):
            raise ValueError("Phase 13 topology splits are empty or overlap")
        if any(seed <= 12020 for split in splits for seed in split):
            raise ValueError("Phase 13 must use fresh topology seeds")
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
        if set(self.red_ppo) != required_ppo:
            raise ValueError("Phase 13 red PPO configuration fields differ")
        if set(self.defender_ppo) != required_ppo | {"policy_layers"}:
            raise ValueError("Phase 13 defender PPO configuration fields differ")
        for budget, ppo, label in (
            (self.red_total_timesteps, self.red_ppo, "red"),
            (self.defender_total_timesteps, self.defender_ppo, "defender"),
        ):
            if budget <= 0 or budget % int(ppo["n_steps"]):
                raise ValueError(f"Phase 13 {label} budget must contain complete rollouts")
            if int(ppo["n_steps"]) % int(ppo["batch_size"]):
                raise ValueError(f"Phase 13 {label} batch size must divide a rollout")
        required_graph = {
            "node_hidden_dim",
            "message_passing_steps",
            "feature_dim",
            "policy_layers",
            "pooling",
            "adjacency",
        }
        if set(self.graph_policy) != required_graph:
            raise ValueError("Phase 13 graph-policy fields differ")
        if (
            self.graph_policy["pooling"] != "mean_max"
            or self.graph_policy["adjacency"] != "directed_agent_known_binary"
        ):
            raise ValueError("Phase 13 graph semantics differ")
        if tuple(map(str, self.hierarchy.get("objectives", ()))) != OBJECTIVES:
            raise ValueError("Phase 13 objective order differs from Phase 12")
        assigned = [item for objective in OBJECTIVES for item in self.hierarchy.get(objective, ())]
        if len(assigned) != len(set(assigned)) or set(assigned) != {
            item.value for item in EnterpriseActionType
        }:
            raise ValueError("Phase 13 hierarchy must partition every red action type")
        if self.hierarchy.get("distribution") != "objective_then_conditional_action":
            raise ValueError("Phase 13 hierarchy distribution differs")
        if self.hierarchy.get("transition_semantics") != (
            "one_concrete_action_per_environment_step"
        ):
            raise ValueError("Phase 13 red transition semantics differ")
        self._validate_defense()
        if self.primary_metrics != ("success_rate", "detection_rate", "penalized_steps"):
            raise ValueError("Phase 13 primary metric family differs")
        if int(self.statistics.get("family_size", -1)) != len(self.primary_metrics):
            raise ValueError("Phase 13 statistical family size differs")
        if self.failure_step_penalty != EnterpriseProfileConfig.from_yaml().max_steps + 1:
            raise ValueError("failure step penalty must be max_steps + 1")
        if not 1 <= self.parallel_training_workers <= 6 or self.runtime_cap_minutes <= 0:
            raise ValueError("Phase 13 execution controls are invalid")

    def _validate_defense(self) -> None:
        required = {
            "ids_levels",
            "initial_ids_level",
            "ids_multipliers",
            "base_detection_probability",
            "patch_delay_episodes",
            "mitigated_exploit_probability",
            "red_detection_penalty",
            "blue_reward",
        }
        if set(self.defense) != required:
            raise ValueError("Phase 13 defense configuration fields differ")
        if list(self.defense["ids_levels"]) != ["sensitive", "balanced", "permissive"]:
            raise ValueError("Phase 13 IDS level order differs")
        if self.defense["initial_ids_level"] != "balanced":
            raise ValueError("Phase 13 IDS must begin balanced")
        multipliers = list(map(float, self.defense["ids_multipliers"]))
        if len(multipliers) != 3 or not all(value > 0 for value in multipliers):
            raise ValueError("Phase 13 IDS multipliers are invalid")
        probabilities = self.defense["base_detection_probability"]
        if set(probabilities) != {item.value for item in EnterpriseActionType} or not all(
            0 <= float(value) <= 1 for value in probabilities.values()
        ):
            raise ValueError("Phase 13 detection probabilities are incomplete")
        if int(self.defense["patch_delay_episodes"]) != 5:
            raise ValueError("Phase 13 patch delay must be exactly five episodes")
        if not 0 < float(self.defense["mitigated_exploit_probability"]) < 1:
            raise ValueError("mitigation must reduce rather than remove exploit feasibility")
        if float(self.defense["red_detection_penalty"]) >= 0:
            raise ValueError("red detection penalty must be negative")
        blue_required = {
            "detection",
            "mitigation_block",
            "red_progress",
            "red_goal",
            "timeout_without_goal",
            "threshold_change_cost",
            "patch_schedule_cost",
        }
        if set(self.defense["blue_reward"]) != blue_required:
            raise ValueError("Phase 13 blue reward fields differ")

    @property
    def initial_ids_level(self) -> int:
        return list(self.defense["ids_levels"]).index(self.defense["initial_ids_level"])

    def digest(self) -> str:
        return canonical_digest(asdict(self))

    def scientific_digest(self) -> str:
        payload = asdict(self)
        payload.pop("parallel_training_workers")
        return canonical_digest(payload)


class MaskedPolicy(Protocol):
    def predict(self, observation, *, action_masks, deterministic: bool): ...

    def set_random_seed(self, seed: int) -> None: ...


class StaticDefenderController:
    """The matched fixed-environment control: balanced IDS and monitor only."""

    def set_random_seed(self, seed: int) -> None:
        del seed

    def predict(self, observation, *, action_masks, deterministic: bool):
        del observation, deterministic
        if not np.asarray(action_masks, dtype=bool)[MONITOR_ACTION]:
            raise MultiAgentStudyError("static defender monitor action was masked")
        return MONITOR_ACTION, None


class DefendedEnterpriseCyberEnv(EnterpriseCyberEnv):
    """Enterprise simulator with delayed mitigation applied to exploit outcomes."""

    def __init__(self, *args, active_patches: set[str], mitigation_probability: float, **kwargs):
        super().__init__(*args, **kwargs)
        self.active_patches = active_patches
        self.mitigation_probability = float(mitigation_probability)

    def _execute(
        self, action: EnterpriseAction
    ) -> tuple[bool, bool, float, tuple[str, ...], tuple[str, ...], str]:
        if action.type == EnterpriseActionType.EXPLOIT and action.target in self.active_patches:
            if self.np_random.random() > self.mitigation_probability:
                return (
                    False,
                    False,
                    -5.0,
                    (f"known_vulnerability:{action.target}", f"patched:{action.target}"),
                    (),
                    "defender mitigation blocked exploit",
                )
        return super()._execute(action)


class InteractiveRedBlueEnv(gym.Env):
    """One blue decision and one concrete red transition per environment step."""

    metadata = {"render_modes": ["ansi"]}

    def __init__(
        self,
        topology_seeds: tuple[int, ...] | list[int],
        profiles: tuple[DeploymentProfile, ...] | list[DeploymentProfile],
        *,
        config: MultiAgentResearchConfig | None = None,
    ) -> None:
        super().__init__()
        if not topology_seeds or not profiles:
            raise ValueError("topology seeds and profiles are required")
        self.config = config or MultiAgentResearchConfig.from_yaml()
        self.profile_config = EnterpriseProfileConfig.from_yaml()
        self.topology_seeds = tuple(map(int, topology_seeds))
        self.profiles = tuple(DeploymentProfile(item) for item in profiles)
        if any(item not in self.config.train_profiles for item in self.profiles):
            raise ValueError("profile is outside the Phase 13 infrastructure distribution")
        self.defender_knowledge = DefenderKnowledge(
            max_vulnerabilities=self.profile_config.max_vulnerabilities,
            patch_delay_episodes=int(self.config.defense["patch_delay_episodes"]),
            initial_ids_level=self.config.initial_ids_level,
        )
        self.profile = self.profiles[0]
        self.topology_seed = self.topology_seeds[0]
        self._env = self._make(self.profile, self.topology_seed)
        self.action_space = self._env.action_space
        self.observation_space = self._env.observation_space
        self.defender_action_space = spaces.Discrete(
            defender_action_count(self.profile_config.max_vulnerabilities)
        )
        size = DefenderObservation.size(self.profile_config.max_vulnerabilities)
        self.defender_observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(size,), dtype=np.float32
        )
        self._pending_decision: DefenderDecision | None = None
        self._red_observation: np.ndarray | None = None
        self._campaign_rng = np.random.default_rng(0)
        self._detection_rng = np.random.default_rng(1)
        self.red_transition_count = 0
        self.invalid_red_actions = 0
        self.invalid_defender_actions = 0
        self.detection_count = 0
        self.mitigation_block_count = 0
        self.defender_action_counts: Counter[str] = Counter()
        self.reset_records: list[dict[str, Any]] = []

    def _make(self, profile: DeploymentProfile, seed: int) -> DefendedEnterpriseCyberEnv:
        return DefendedEnterpriseCyberEnv(
            generate_profile_topology(profile, seed, self.profile_config),
            max_steps=self.profile_config.max_steps,
            max_nodes=self.profile_config.max_nodes,
            max_vulnerabilities=self.profile_config.max_vulnerabilities,
            active_patches=self.defender_knowledge.active_patches,
            mitigation_probability=float(
                self.config.defense["mitigated_exploit_probability"]
            ),
        )

    @property
    def true_topology(self):
        return self._env.true_topology

    @property
    def knowledge(self):
        return self._env.knowledge

    def defender_observation(self) -> np.ndarray:
        observation = DefenderObservation.from_knowledge(
            self.defender_knowledge,
            max_steps=self.profile_config.max_steps,
        ).as_array()
        if not self.defender_observation_space.contains(observation):
            raise MultiAgentStudyError("DefenderKnowledge observation left its fixed space")
        return observation

    def defender_action_masks(self) -> np.ndarray:
        mask = self.defender_knowledge.action_mask()
        if mask.shape != (int(self.defender_action_space.n),) or not mask.any():
            raise MultiAgentStudyError("DefenderKnowledge action mask is invalid")
        return mask

    def action_masks(self) -> np.ndarray:
        return self._env.action_masks()

    def action_index(self, action_type: EnterpriseActionType, target: str) -> int:
        return self._env.action_index(action_type, target)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._campaign_rng = np.random.default_rng(int(seed))
            self._detection_rng = np.random.default_rng(int(seed) ^ 0x5EED13)
            self.defender_knowledge.reset_campaign()
            self.red_transition_count = 0
            self.invalid_red_actions = 0
            self.invalid_defender_actions = 0
            self.detection_count = 0
            self.mitigation_block_count = 0
            self.defender_action_counts.clear()
            self.reset_records.clear()
        options = options or {}
        raw_topology_seed = options.get("topology_seed")
        raw_profile = options.get("profile")
        topology_seed = (
            int(raw_topology_seed)
            if raw_topology_seed is not None
            else int(self._campaign_rng.choice(self.topology_seeds))
        )
        profile = (
            DeploymentProfile(raw_profile)
            if raw_profile is not None
            else DeploymentProfile(self._campaign_rng.choice(self.profiles))
        )
        if topology_seed not in self.topology_seeds or profile not in self.profiles:
            raise ValueError("requested seed/profile is outside this Phase 13 campaign")
        self.topology_seed, self.profile = topology_seed, profile
        activated = self.defender_knowledge.begin_episode()
        self._env = self._make(profile, topology_seed)
        if self._env.action_space != self.action_space:
            raise MultiAgentStudyError("red action space changed across enterprise profiles")
        if self._env.observation_space != self.observation_space:
            raise MultiAgentStudyError("red observation space changed across profiles")
        episode_seed = int(self._campaign_rng.integers(0, 2**31 - 1))
        observation, info = self._env.reset(seed=episode_seed)
        self._red_observation = observation
        self._pending_decision = None
        topology_hash = topology_digest(self.true_topology)
        record = {
            "episode_index": self.defender_knowledge.episode_index,
            "profile": profile.value,
            "topology_seed": topology_seed,
            "topology_hash": topology_hash,
            "profile_config_hash": self.profile_config.digest(),
            "episode_seed": episode_seed,
            "activated_patches": list(activated),
        }
        self.reset_records.append(record)
        info.update(
            {
                **record,
                "defender_observation_source": "DefenderKnowledge",
                "red_observation_source": "AgentKnowledge",
                "defender_action_mask": self.defender_action_masks(),
            }
        )
        return observation, info

    def apply_defender_action(self, action: int) -> DefenderDecision:
        selected = int(np.asarray(action).item())
        mask = self.defender_action_masks()
        if not 0 <= selected < len(mask) or not mask[selected]:
            self.invalid_defender_actions += 1
            raise MultiAgentStudyError("blue policy selected a DefenderKnowledge-masked action")
        if self._pending_decision is not None:
            raise MultiAgentStudyError("blue policy attempted two decisions before one red step")
        self._pending_decision = self.defender_knowledge.apply_action(selected)
        self.defender_action_counts[self._pending_decision.name] += 1
        return self._pending_decision

    def _detection_probability(self, action_type: EnterpriseActionType) -> float:
        base = float(self.config.defense["base_detection_probability"][action_type.value])
        multiplier = float(
            self.config.defense["ids_multipliers"][self.defender_knowledge.ids_level]
        )
        return min(1.0, max(0.0, base * multiplier))

    def _blue_reward(
        self,
        *,
        detected: bool,
        mitigation_blocked: bool,
        red_progress: bool,
        terminated: bool,
        truncated: bool,
        decision: DefenderDecision,
    ) -> float:
        reward = self.config.defense["blue_reward"]
        value = float(reward["detection"]) * int(detected)
        value += float(reward["mitigation_block"]) * int(mitigation_blocked)
        value += float(reward["red_progress"]) * int(red_progress)
        value += float(reward["red_goal"]) * int(terminated)
        value += float(reward["timeout_without_goal"]) * int(truncated and not terminated)
        if decision.action in {LOWER_IDS_ACTION, RAISE_IDS_ACTION}:
            value += float(reward["threshold_change_cost"])
        if decision.action >= PATCH_ACTION_OFFSET:
            value += float(reward["patch_schedule_cost"])
        return value

    def step_red(self, action: int):
        if self._pending_decision is None:
            raise MultiAgentStudyError(
                "red transition requires exactly one preceding blue decision"
            )
        selected = int(np.asarray(action).item())
        mask = self.action_masks()
        if not 0 <= selected < len(mask) or not mask[selected]:
            self.invalid_red_actions += 1
            raise MultiAgentStudyError("red policy selected an AgentKnowledge-masked action")
        resolved = self._env._resolve_action(self._env.actions[selected])
        probability = self._detection_probability(resolved.type)
        detected = bool(self._detection_rng.random() < probability)
        observation, native_reward, terminated, truncated, info = self._env.step(selected)
        event = info["event"]
        revealed = tuple(
            outcome.removeprefix("vulnerability:")
            for outcome in event.outcomes
            if outcome.startswith("vulnerability:")
        )
        self.defender_knowledge.observe(
            event.action.type,
            detected=detected,
            revealed_vulnerabilities=revealed,
        )
        mitigation_blocked = event.reason == "defender mitigation blocked exploit"
        red_reward = float(native_reward) + (
            float(self.config.defense["red_detection_penalty"]) if detected else 0.0
        )
        decision = self._pending_decision
        blue_reward = self._blue_reward(
            detected=detected,
            mitigation_blocked=mitigation_blocked,
            red_progress=bool(event.state_changed and not event.goal_reached),
            terminated=bool(terminated),
            truncated=bool(truncated),
            decision=decision,
        )
        self._pending_decision = None
        self._red_observation = observation
        self.red_transition_count += 1
        self.detection_count += int(detected)
        self.mitigation_block_count += int(mitigation_blocked)
        info.update(
            {
                "profile": self.profile.value,
                "topology_seed": self.topology_seed,
                "topology_hash": topology_digest(self.true_topology),
                "episode_index": self.defender_knowledge.episode_index,
                "red_native_reward": float(native_reward),
                "red_reward": red_reward,
                "detected": detected,
                "detection_probability": probability,
                "ids_level": self.defender_knowledge.ids_level,
                "ids_level_name": self.config.defense["ids_levels"][
                    self.defender_knowledge.ids_level
                ],
                "defender_action": decision.name,
                "defender_action_index": decision.action,
                "defender_action_state_changed": decision.state_changed,
                "scheduled_vulnerability": decision.scheduled_vulnerability,
                "patch_activation_episode": decision.activation_episode,
                "pending_patch_count": len(self.defender_knowledge.pending_patches),
                "active_patch_count": len(self.defender_knowledge.active_patches),
                "mitigation_blocked": mitigation_blocked,
                "defender_reward": blue_reward,
                "defender_observation_source": "DefenderKnowledge",
                "red_observation_source": "AgentKnowledge",
            }
        )
        return observation, red_reward, terminated, truncated, info

    def attack_path(self):
        return self._env.attack_path()


class RedAgainstDefenderEnv(gym.Env):
    """Red MaskablePPO view with a fixed static or learned blue controller."""

    metadata = {"render_modes": ["ansi"]}

    def __init__(
        self,
        topology_seeds: tuple[int, ...] | list[int],
        profiles: tuple[DeploymentProfile, ...] | list[DeploymentProfile],
        *,
        defender: MaskedPolicy | StaticDefenderController,
        config: MultiAgentResearchConfig | None = None,
        deterministic_defender: bool = False,
    ) -> None:
        super().__init__()
        self.interaction = InteractiveRedBlueEnv(topology_seeds, profiles, config=config)
        self.defender = defender
        self.deterministic_defender = bool(deterministic_defender)
        self.action_space = self.interaction.action_space
        self.observation_space = self.interaction.observation_space

    @property
    def true_topology(self):
        return self.interaction.true_topology

    @property
    def knowledge(self):
        return self.interaction.knowledge

    @property
    def defender_knowledge(self) -> DefenderKnowledge:
        return self.interaction.defender_knowledge

    @property
    def reset_records(self) -> list[dict[str, Any]]:
        return self.interaction.reset_records

    def action_masks(self) -> np.ndarray:
        return self.interaction.action_masks()

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self.defender.set_random_seed(int(seed) + 1)
        return self.interaction.reset(seed=seed, options=options)

    def step(self, action: int):
        blue_observation = self.interaction.defender_observation()
        blue_mask = self.interaction.defender_action_masks()
        blue_action, _ = self.defender.predict(
            blue_observation,
            action_masks=blue_mask,
            deterministic=self.deterministic_defender,
        )
        self.interaction.apply_defender_action(int(np.asarray(blue_action).item()))
        return self.interaction.step_red(action)

    def attack_path(self):
        return self.interaction.attack_path()


class DefenderTrainingEnv(gym.Env):
    """Blue MaskablePPO view against a frozen red policy."""

    metadata = {"render_modes": ["ansi"]}

    def __init__(
        self,
        topology_seeds: tuple[int, ...] | list[int],
        profiles: tuple[DeploymentProfile, ...] | list[DeploymentProfile],
        *,
        red_policy: MaskedPolicy,
        config: MultiAgentResearchConfig | None = None,
    ) -> None:
        super().__init__()
        self.interaction = InteractiveRedBlueEnv(topology_seeds, profiles, config=config)
        self.red_policy = red_policy
        self.action_space = self.interaction.defender_action_space
        self.observation_space = self.interaction.defender_observation_space
        self._red_observation: np.ndarray | None = None
        self.red_mask_checks = 0

    def action_masks(self) -> np.ndarray:
        return self.interaction.defender_action_masks()

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self.red_policy.set_random_seed(int(seed) + 2)
            self.red_mask_checks = 0
        self._red_observation, info = self.interaction.reset(seed=seed, options=options)
        return self.interaction.defender_observation(), info

    def step(self, action: int):
        if self._red_observation is None:
            raise MultiAgentStudyError("defender environment must be reset before stepping")
        self.interaction.apply_defender_action(action)
        red_mask = self.interaction.action_masks()
        red_action, _ = self.red_policy.predict(
            self._red_observation,
            action_masks=red_mask,
            deterministic=False,
        )
        selected = int(np.asarray(red_action).item())
        if not 0 <= selected < len(red_mask) or not red_mask[selected]:
            raise MultiAgentStudyError("frozen red policy selected an invalid action")
        self.red_mask_checks += 1
        self._red_observation, _, terminated, truncated, info = (
            self.interaction.step_red(selected)
        )
        return (
            self.interaction.defender_observation(),
            float(info["defender_reward"]),
            terminated,
            truncated,
            info,
        )


def make_red_model(config: MultiAgentResearchConfig, seed: int, env):
    from sb3_contrib import MaskablePPO

    return MaskablePPO(
        phase12_policy(config, "hierarchical_graph"),
        env,
        seed=int(seed),
        device="cpu",
        verbose=0,
        policy_kwargs=phase12_policy_kwargs(config, "hierarchical_graph"),
        **config.red_ppo,
    )


def make_defender_model(config: MultiAgentResearchConfig, seed: int, env):
    from sb3_contrib import MaskablePPO

    ppo = dict(config.defender_ppo)
    policy_layers = ppo.pop("policy_layers")
    return MaskablePPO(
        "MlpPolicy",
        env,
        seed=int(seed),
        device="cpu",
        verbose=0,
        policy_kwargs={"net_arch": list(policy_layers)},
        **ppo,
    )


def red_observation_schema() -> dict[str, Any]:
    profile = EnterpriseProfileConfig.from_yaml()
    feature_count = EnterpriseCyberEnv._FEATURES_PER_NODE
    return {
        "source": "AgentKnowledge",
        "max_nodes": profile.max_nodes,
        "node_feature_count": feature_count,
        "node_values": profile.max_nodes * feature_count,
        "adjacency_values": profile.max_nodes * profile.max_nodes,
        "step_values": 1,
        "observation_values": (
            profile.max_nodes * feature_count + profile.max_nodes * profile.max_nodes + 1
        ),
        "hidden_topology_fields": 0,
    }


def phase13_policy_manifest(config: MultiAgentResearchConfig) -> dict[str, Any]:
    profile = EnterpriseProfileConfig.from_yaml()
    return {
        "red_observation_schema": red_observation_schema(),
        "defender_observation_schema": defender_observation_schema(
            profile.max_vulnerabilities
        ),
        "red_action_to_objective": list(action_to_objective(config)),
        "defender_action_count": defender_action_count(profile.max_vulnerabilities),
        "red_transition_semantics": "one_concrete_action_per_environment_step",
        "turn_semantics": "one_blue_decision_then_one_red_transition",
        "policy_ground_truth_fields": 0,
    }


def prior_topology_seed_limit() -> int:
    """Evidence helper keeping Phase 13 disjoint from all earlier frozen phases."""
    prior = OnPremGeneralisationSplit()
    return max(12020, *prior.train, *prior.validation, *prior.test)


__all__ = [
    "DEFAULT_MULTIAGENT_CONFIG",
    "DefendedEnterpriseCyberEnv",
    "DefenderTrainingEnv",
    "InteractiveRedBlueEnv",
    "MultiAgentResearchConfig",
    "MultiAgentStudyError",
    "PHASE13_ARMS",
    "RedAgainstDefenderEnv",
    "StaticDefenderController",
    "make_defender_model",
    "make_red_model",
    "phase13_policy_manifest",
    "prior_topology_seed_limit",
    "red_observation_schema",
]
