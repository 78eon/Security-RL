from __future__ import annotations

import numpy as np
import pytest

from rlredteam.enterprise.defender_state import (
    LOWER_IDS_ACTION,
    MONITOR_ACTION,
    PATCH_ACTION_OFFSET,
    DefenderKnowledge,
    DefenderObservation,
    defender_observation_schema,
)
from rlredteam.enterprise.environment import EnterpriseActionType
from rlredteam.enterprise.multiagent import (
    DefendedEnterpriseCyberEnv,
    DefenderTrainingEnv,
    InteractiveRedBlueEnv,
    MultiAgentResearchConfig,
    MultiAgentStudyError,
    RedAgainstDefenderEnv,
    StaticDefenderController,
    phase13_policy_manifest,
)
from rlredteam.enterprise.profiles import DeploymentProfile, generate_profile_topology


class HighestValidRed:
    def set_random_seed(self, seed: int) -> None:
        self.seed = seed

    def predict(self, observation, *, action_masks, deterministic: bool):
        del observation, deterministic
        return int(np.flatnonzero(action_masks)[-1]), None


def test_phase13_protocol_is_fresh_fixed_and_partial_observation_only() -> None:
    config = MultiAgentResearchConfig.from_yaml()
    assert config.training_seeds == tuple(range(601, 611))
    assert set(config.topology_splits) == {"train", "validation", "test"}
    assert min(seed for split in config.topology_splits.values() for seed in split) == 13001
    assert config.defense["patch_delay_episodes"] == 5
    manifest = phase13_policy_manifest(config)
    assert manifest["red_observation_schema"]["source"] == "AgentKnowledge"
    assert manifest["defender_observation_schema"]["source"] == "DefenderKnowledge"
    assert manifest["red_observation_schema"]["hidden_topology_fields"] == 0
    assert manifest["defender_observation_schema"]["hidden_topology_fields"] == 0
    assert manifest["policy_ground_truth_fields"] == 0
    assert len(manifest["red_action_to_objective"]) == 968


def test_defender_observation_and_mask_use_only_defender_knowledge() -> None:
    knowledge = DefenderKnowledge(max_vulnerabilities=8, patch_delay_episodes=5)
    knowledge.begin_episode()
    initial = DefenderObservation.from_knowledge(knowledge, max_steps=200).as_array()
    schema = defender_observation_schema(8)
    assert initial.shape == (schema["observation_values"],)
    assert knowledge.action_mask().tolist() == [True, True, True, *([False] * 8)]

    knowledge.observe(
        EnterpriseActionType.ASSESS_VULNERABILITY,
        detected=True,
        revealed_vulnerabilities=("public-observed-vulnerability",),
    )
    observed = DefenderObservation.from_knowledge(knowledge, max_steps=200).as_array()
    assert not np.array_equal(initial, observed)
    assert knowledge.action_mask()[PATCH_ACTION_OFFSET]
    decision = knowledge.apply_action(PATCH_ACTION_OFFSET)
    assert decision.scheduled_vulnerability == "public-observed-vulnerability"
    assert not knowledge.action_mask()[PATCH_ACTION_OFFSET]


def test_patch_activates_after_exactly_five_completed_episode_boundaries() -> None:
    knowledge = DefenderKnowledge(max_vulnerabilities=8, patch_delay_episodes=5)
    assert knowledge.begin_episode() == ()  # episode 0
    knowledge.observe(
        EnterpriseActionType.ASSESS_VULNERABILITY,
        detected=False,
        revealed_vulnerabilities=("vulnerability-a",),
    )
    decision = knowledge.apply_action(PATCH_ACTION_OFFSET)
    assert decision.activation_episode == 5
    for expected_episode in range(1, 5):
        assert knowledge.begin_episode() == ()
        assert knowledge.episode_index == expected_episode
        assert "vulnerability-a" not in knowledge.active_patches
    assert knowledge.begin_episode() == ("vulnerability-a",)
    assert knowledge.episode_index == 5
    assert "vulnerability-a" in knowledge.active_patches


def test_threshold_boundary_and_duplicate_patch_actions_are_masked() -> None:
    knowledge = DefenderKnowledge(max_vulnerabilities=8, patch_delay_episodes=5)
    knowledge.begin_episode()
    knowledge.apply_action(LOWER_IDS_ACTION)
    assert not knowledge.action_mask()[LOWER_IDS_ACTION]
    with pytest.raises(ValueError, match="outside DefenderKnowledge mask"):
        knowledge.apply_action(LOWER_IDS_ACTION)
    assert knowledge.action_mask()[MONITOR_ACTION]


def test_hidden_topology_does_not_change_initial_red_or_blue_policy_inputs() -> None:
    config = MultiAgentResearchConfig.from_yaml()
    environments = [
        InteractiveRedBlueEnv((seed,), ("legacy",), config=config)
        for seed in (13001, 13002)
    ]
    snapshots = []
    for environment in environments:
        red, _ = environment.reset(seed=77)
        snapshots.append(
            (
                red,
                environment.action_masks(),
                environment.defender_observation(),
                environment.defender_action_masks(),
            )
        )
    for left, right in zip(snapshots[0], snapshots[1], strict=True):
        assert np.array_equal(left, right)
    assert environments[0].true_topology.to_dict() != environments[1].true_topology.to_dict()


def test_one_blue_decision_precedes_exactly_one_red_transition() -> None:
    config = MultiAgentResearchConfig.from_yaml()
    environment = InteractiveRedBlueEnv((13001,), ("legacy",), config=config)
    environment.reset(seed=91)
    red_action = int(np.flatnonzero(environment.action_masks())[0])
    with pytest.raises(MultiAgentStudyError, match="requires exactly one"):
        environment.step_red(red_action)
    environment.apply_defender_action(MONITOR_ACTION)
    with pytest.raises(MultiAgentStudyError, match="two decisions"):
        environment.apply_defender_action(MONITOR_ACTION)
    _, _, _, _, info = environment.step_red(red_action)
    assert environment.red_transition_count == 1
    assert environment.defender_knowledge.decision_count == 1
    assert info["red_observation_source"] == "AgentKnowledge"
    assert info["defender_observation_source"] == "DefenderKnowledge"


def test_static_red_wrapper_is_deterministic_for_an_identical_campaign_seed() -> None:
    config = MultiAgentResearchConfig.from_yaml()
    traces = []
    for _ in range(2):
        environment = RedAgainstDefenderEnv(
            (13001,),
            ("cloud",),
            defender=StaticDefenderController(),
            config=config,
            deterministic_defender=True,
        )
        observation, _ = environment.reset(seed=123)
        trace = []
        for _ in range(8):
            mask = environment.action_masks()
            selected = int(np.flatnonzero(mask)[-1])
            observation, reward, terminated, truncated, info = environment.step(selected)
            trace.append(
                (
                    selected,
                    reward,
                    info["detected"],
                    info["detection_probability"],
                    info["event"].action.name,
                )
            )
            if terminated or truncated:
                break
        assert environment.observation_space.contains(observation)
        traces.append(trace)
    assert traces[0] == traces[1]


def test_blue_training_view_preserves_fixed_spaces_across_profiles() -> None:
    config = MultiAgentResearchConfig.from_yaml()
    environment = DefenderTrainingEnv(
        (13001, 13002),
        config.train_profiles,
        red_policy=HighestValidRed(),
        config=config,
    )
    expected_observation_space = environment.observation_space
    expected_action_space = environment.action_space
    for profile in config.train_profiles:
        observation, _ = environment.reset(
            seed=51,
            options={"profile": profile.value, "topology_seed": 13001},
        )
        assert expected_observation_space.contains(observation)
        action = int(np.flatnonzero(environment.action_masks())[0])
        observation, reward, _, _, info = environment.step(action)
        assert expected_observation_space.contains(observation)
        assert np.isfinite(reward)
        assert info["profile"] == profile.value
        assert environment.action_space == expected_action_space


class FixedRandom:
    def random(self) -> float:
        return 0.5


def _reveal_profile_vulnerability(environment: InteractiveRedBlueEnv) -> str:
    sequence = (
        (EnterpriseActionType.DISCOVER_NETWORK, "network_1"),
        (EnterpriseActionType.ENUMERATE_HOST, "host_entry"),
        (EnterpriseActionType.ENUMERATE_SERVICE, "service_entry"),
        (EnterpriseActionType.ASSESS_VULNERABILITY, "service_entry"),
    )
    for kind, target in sequence:
        selected = environment.action_index(kind, target)
        assert environment.action_masks()[selected]
        environment.apply_defender_action(MONITOR_ACTION)
        _, _, terminated, truncated, info = environment.step_red(selected)
        assert info["event"].success
        assert not terminated and not truncated
    return next(iter(environment.knowledge.known_vulnerabilities))


def test_active_mitigation_changes_probability_without_masking_exploit() -> None:
    config = MultiAgentResearchConfig.from_yaml()
    environment = InteractiveRedBlueEnv((13001,), ("legacy",), config=config)
    environment.reset(seed=104)
    vulnerability = _reveal_profile_vulnerability(environment)
    exploit = environment.action_index(EnterpriseActionType.EXPLOIT, vulnerability)
    assert environment.action_masks()[exploit]
    environment.defender_knowledge.active_patches.add(vulnerability)
    environment._env._np_random = FixedRandom()
    assert environment.action_masks()[exploit]
    environment.apply_defender_action(MONITOR_ACTION)
    _, _, _, _, blocked = environment.step_red(exploit)
    assert blocked["mitigation_blocked"]
    assert not blocked["event"].success

    control_topology = generate_profile_topology(DeploymentProfile.LEGACY, 13001)
    control = DefendedEnterpriseCyberEnv(
        control_topology,
        max_steps=environment.profile_config.max_steps,
        max_nodes=environment.profile_config.max_nodes,
        max_vulnerabilities=environment.profile_config.max_vulnerabilities,
        active_patches=set(),
        mitigation_probability=0.25,
    )
    control.reset(seed=104)
    for kind, target in (
        (EnterpriseActionType.DISCOVER_NETWORK, "network_1"),
        (EnterpriseActionType.ENUMERATE_HOST, "host_entry"),
        (EnterpriseActionType.ENUMERATE_SERVICE, "service_entry"),
        (EnterpriseActionType.ASSESS_VULNERABILITY, "service_entry"),
    ):
        control.step(control.action_index(kind, target))
    control._np_random = FixedRandom()
    _, _, _, _, unblocked = control.step(
        control.action_index(EnterpriseActionType.EXPLOIT, vulnerability)
    )
    assert unblocked["event"].success
