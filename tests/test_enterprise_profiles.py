"""Phase 4 infrastructure-profile invariants and simulation safety tests."""

from __future__ import annotations

import socket
from dataclasses import replace

import numpy as np
import pytest

from rlredteam.enterprise.environment import EnterpriseActionType, EnterpriseCyberEnv
from rlredteam.enterprise.model import NodeType
from rlredteam.enterprise.onprem import (
    generate_onprem_topology,
    knowledge_policy_action,
    topology_digest,
)
from rlredteam.enterprise.profiles import (
    DeploymentProfile,
    EnterpriseProfileConfig,
    InfrastructureCurriculumEnv,
    generate_profile_topology,
)

RESEARCH_PROFILES = (
    DeploymentProfile.LEGACY,
    DeploymentProfile.CLOUD,
    DeploymentProfile.HYBRID,
)


def make_env(profile: DeploymentProfile, seed: int) -> EnterpriseCyberEnv:
    config = EnterpriseProfileConfig.from_yaml()
    return EnterpriseCyberEnv(
        generate_profile_topology(profile, seed, config),
        max_steps=config.max_steps,
        max_nodes=config.max_nodes,
        max_vulnerabilities=config.max_vulnerabilities,
    )


def complete(profile: DeploymentProfile, seed: int) -> EnterpriseCyberEnv:
    env = make_env(profile, seed)
    env.reset(seed=seed)
    terminated = truncated = False
    while not (terminated or truncated):
        _, _, terminated, truncated, _ = env.step(knowledge_policy_action(env))
    assert terminated and not truncated
    return env


def test_configuration_is_complete_stable_and_bounded() -> None:
    config = EnterpriseProfileConfig.from_yaml()
    assert set(config.profiles) == set(RESEARCH_PROFILES)
    assert config.digest() == EnterpriseProfileConfig.from_yaml().digest()
    assert 1 <= config.min_segments <= config.max_segments <= 5
    assert 10 <= config.min_hosts <= config.max_hosts <= 60

    with pytest.raises(ValueError, match="max_nodes cannot fit"):
        replace(config, max_nodes=10).validate()


@pytest.mark.parametrize("profile", RESEARCH_PROFILES)
def test_profile_generation_is_deterministic_and_structurally_varied(
    profile: DeploymentProfile,
) -> None:
    first = generate_profile_topology(profile, 42)
    repeated = generate_profile_topology(profile, 42)
    assert first.to_dict() == repeated.to_dict()
    assert topology_digest(first) == topology_digest(repeated)

    signatures = {
        (len(topology.nodes), len(topology.edges), topology_digest(topology))
        for topology in (
            generate_profile_topology(profile, seed) for seed in range(1, 21)
        )
    }
    assert len(signatures) == 20


def test_profiles_are_distinct_configurations_of_one_typed_graph() -> None:
    legacy = generate_profile_topology(DeploymentProfile.LEGACY, 7)
    cloud = generate_profile_topology(DeploymentProfile.CLOUD, 7)
    hybrid = generate_profile_topology(DeploymentProfile.HYBRID, 7)
    legacy_types = {node.type for node in legacy.nodes.values()}
    cloud_types = {node.type for node in cloud.nodes.values()}
    hybrid_types = {node.type for node in hybrid.nodes.values()}

    assert NodeType.LEGACY_HOST in legacy_types
    assert not legacy_types & {
        NodeType.CLOUD_NETWORK,
        NodeType.CLOUD_WORKLOAD,
        NodeType.IAM_ROLE,
        NodeType.CLOUD_RESOURCE,
        NodeType.STORAGE,
    }
    assert {NodeType.CLOUD_NETWORK, NodeType.CLOUD_WORKLOAD, NodeType.IAM_ROLE} <= cloud_types
    assert NodeType.LEGACY_HOST not in cloud_types
    assert {
        NodeType.NETWORK_SEGMENT,
        NodeType.CLOUD_NETWORK,
        NodeType.HOST,
        NodeType.LEGACY_HOST,
        NodeType.CLOUD_WORKLOAD,
        NodeType.IDENTITY,
        NodeType.IAM_ROLE,
    } <= hybrid_types


@pytest.mark.parametrize("profile", RESEARCH_PROFILES)
def test_generated_profiles_respect_capacities(profile: DeploymentProfile) -> None:
    config = EnterpriseProfileConfig.from_yaml()
    for seed in range(1, 101):
        topology = generate_profile_topology(profile, seed, config)
        hosts = sum(
            node.type in {NodeType.HOST, NodeType.LEGACY_HOST, NodeType.CLOUD_WORKLOAD}
            for node in topology.nodes.values()
        )
        networks = sum(
            node.type in {NodeType.NETWORK_SEGMENT, NodeType.CLOUD_NETWORK}
            for node in topology.nodes.values()
        )
        assert config.min_hosts <= hosts <= config.max_hosts
        assert config.min_segments <= networks <= config.max_segments
        assert len(topology.nodes) <= config.max_nodes
        assert len(topology.vulnerabilities) <= config.max_vulnerabilities


@pytest.mark.parametrize("profile", RESEARCH_PROFILES)
@pytest.mark.parametrize("seed", [1, 2, 7, 19, 42, 1001, 2001])
def test_every_profile_sample_has_a_discovered_causal_path(
    profile: DeploymentProfile, seed: int
) -> None:
    env = complete(profile, seed)
    actions = [event.action.type for event in env.attack_path()]
    assert env.attack_path()[-1].goal_reached
    assert EnterpriseActionType.EXPLOIT in actions
    assert EnterpriseActionType.OBTAIN_CREDENTIAL in actions
    assert EnterpriseActionType.ACCESS_ASSET in actions


def test_profile_generation_and_execution_never_use_real_network(monkeypatch) -> None:
    def refuse(*args, **kwargs):
        raise AssertionError("enterprise simulation attempted real network access")

    monkeypatch.setattr(socket, "socket", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    for profile in RESEARCH_PROFILES:
        complete(profile, 42)


def test_curriculum_hides_seed_and_profile_behind_fixed_spaces() -> None:
    curriculum = InfrastructureCurriculumEnv(
        [1, 2, 3],
        list(RESEARCH_PROFILES),
    )
    expected_action_space = curriculum.action_space
    expected_observation_space = curriculum.observation_space
    for profile in RESEARCH_PROFILES:
        observation, info = curriculum.reset(
            seed=99,
            options={"profile": profile.value, "topology_seed": 2},
        )
        assert curriculum.action_space == expected_action_space
        assert curriculum.observation_space == expected_observation_space
        assert curriculum.observation_space.contains(observation)
        assert info["profile"] == profile.value
        assert info["topology_seed"] == 2
        assert info["topology_hash"] == topology_digest(curriculum.true_topology)
        assert info["profile_config_hash"] == curriculum.config.digest()

    with pytest.raises(ValueError, match="outside this curriculum"):
        curriculum.reset(options={"topology_seed": 2001})
    with pytest.raises(ValueError, match="outside this curriculum"):
        curriculum.reset(options={"profile": DeploymentProfile.ON_PREMISES.value})


def test_action_mask_cannot_read_profile_ground_truth() -> None:
    curriculum = InfrastructureCurriculumEnv([1], [DeploymentProfile.HYBRID])
    curriculum.reset(
        seed=77,
        options={"profile": DeploymentProfile.HYBRID.value, "topology_seed": 1},
    )
    expected = curriculum.action_masks().copy()

    class ForbiddenTruth:
        def __getattribute__(self, name):
            raise AssertionError(f"action mask read ground truth: {name}")

    curriculum._env.true_topology = ForbiddenTruth()
    actual = curriculum.action_masks()
    assert actual.dtype == np.bool_
    assert np.array_equal(actual, expected)


def test_on_premises_profile_delegates_to_frozen_phase_two_generator() -> None:
    for seed in (1, 42, 1001, 2001):
        assert topology_digest(
            generate_profile_topology(DeploymentProfile.ON_PREMISES, seed)
        ) == topology_digest(generate_onprem_topology(seed))
