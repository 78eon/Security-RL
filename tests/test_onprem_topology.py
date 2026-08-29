"""Phase 2 on-prem distribution, secrecy and feasibility tests."""

from __future__ import annotations

import socket

import numpy as np
import pytest

from rlredteam.enterprise.environment import EnterpriseActionType, EnterpriseCyberEnv
from rlredteam.enterprise.model import NodeType
from rlredteam.enterprise.onprem import (
    ON_PREM_TYPES,
    OnPremCurriculumEnv,
    OnPremGeneralisationSplit,
    OnPremTopologyConfig,
    generate_onprem_topology,
    knowledge_policy_action,
    topology_digest,
)


def make_env(seed: int) -> EnterpriseCyberEnv:
    config = OnPremTopologyConfig.from_yaml()
    return EnterpriseCyberEnv(
        generate_onprem_topology(seed, config),
        max_steps=config.max_steps,
        max_nodes=config.max_nodes,
        max_vulnerabilities=config.max_vulnerabilities,
    )


def complete_with_knowledge_policy(seed: int) -> EnterpriseCyberEnv:
    env = make_env(seed)
    env.reset(seed=seed)
    terminated = truncated = False
    while not (terminated or truncated):
        _, _, terminated, truncated, _ = env.step(knowledge_policy_action(env))
    assert terminated and not truncated
    return env


def test_distribution_is_deterministic_hashed_and_structurally_varied() -> None:
    config = OnPremTopologyConfig.from_yaml()
    first = generate_onprem_topology(42, config)
    repeated = generate_onprem_topology(42, config)
    assert first.to_dict() == repeated.to_dict()
    assert topology_digest(first) == topology_digest(repeated)

    signatures = set()
    for seed in range(1, 31):
        topology = generate_onprem_topology(seed, config)
        signatures.add(
            (
                len(topology.nodes),
                len(topology.edges),
                sum(edge.type == "pivots_to" for edge in topology.edges),
                tuple(
                    sorted(
                        vulnerability.id
                        for vulnerability in topology.vulnerabilities.values()
                    )
                ),
            )
        )
    assert len(signatures) >= 12


def test_distribution_is_bounded_and_on_prem_only() -> None:
    config = OnPremTopologyConfig.from_yaml()
    forbidden = {
        NodeType.CLOUD_RESOURCE,
        NodeType.CLOUD_ACCOUNT,
        NodeType.CLOUD_NETWORK,
        NodeType.CLOUD_WORKLOAD,
        NodeType.IAM_ROLE,
        NodeType.STORAGE,
        NodeType.LEGACY_HOST,
    }
    for seed in (*range(1, 61), *range(1001, 1021), *range(2001, 2021)):
        topology = generate_onprem_topology(seed, config)
        types = {node.type for node in topology.nodes.values()}
        assert types <= ON_PREM_TYPES
        assert not types & forbidden
        host_count = sum(node.type == NodeType.HOST for node in topology.nodes.values())
        segment_count = sum(
            node.type == NodeType.NETWORK_SEGMENT for node in topology.nodes.values()
        )
        assert config.min_hosts <= host_count <= config.max_hosts
        assert config.min_segments <= segment_count <= config.max_segments
        assert len(topology.nodes) <= config.max_nodes
        assert len(topology.vulnerabilities) <= config.max_vulnerabilities


def test_invalid_capacity_fails_closed() -> None:
    with pytest.raises(ValueError, match="worst-case topology size"):
        OnPremTopologyConfig(max_nodes=10)


def test_generalisation_splits_are_nonempty_and_disjoint() -> None:
    split = OnPremGeneralisationSplit()
    assert split.train and split.validation and split.test
    assert not set(split.train) & set(split.validation)
    assert not set(split.train) & set(split.test)
    assert not set(split.validation) & set(split.test)


def test_spaces_and_initial_policy_inputs_do_not_reveal_hidden_topology() -> None:
    first = make_env(1)
    second = make_env(29)
    first_observation, first_info = first.reset(seed=77)
    second_observation, second_info = second.reset(seed=77)

    assert first.action_space == second.action_space
    assert first.observation_space == second.observation_space
    assert first.actions == second.actions
    assert np.array_equal(first_observation, second_observation)
    assert np.array_equal(first_info["action_mask"], second_info["action_mask"])


@pytest.mark.parametrize(
    "seed",
    [1, 2, 3, 7, 11, 19, 42, 60, 1001, 1020, 2001, 2020],
)
def test_every_sampled_topology_has_a_reconstructable_path(seed: int) -> None:
    env = complete_with_knowledge_policy(seed)
    events = env.attack_path()
    action_types = [event.action.type for event in events]
    assert events[-1].goal_reached
    assert EnterpriseActionType.DISCOVER_NETWORK in action_types
    assert EnterpriseActionType.ENUMERATE_HOST in action_types
    assert EnterpriseActionType.ASSESS_VULNERABILITY in action_types
    assert EnterpriseActionType.EXPLOIT in action_types
    assert EnterpriseActionType.OBTAIN_CREDENTIAL in action_types
    assert EnterpriseActionType.ACCESS_ASSET in action_types


def test_every_preregistered_topology_has_a_feasible_path() -> None:
    split = OnPremGeneralisationSplit()
    for seed in (*split.train, *split.validation, *split.test):
        env = complete_with_knowledge_policy(seed)
        assert env.knowledge.accessed_assets == {"asset_crown"}


def test_discovery_and_attack_are_interleaved_across_new_segments() -> None:
    env = complete_with_knowledge_policy(6)
    action_types = [event.action.type for event in env.attack_path()]
    pivot_index = action_types.index(EnterpriseActionType.PIVOT)
    assert action_types[pivot_index + 1] == EnterpriseActionType.DISCOVER_NETWORK


def test_generation_and_episode_do_not_use_real_network(monkeypatch) -> None:
    def refuse(*args, **kwargs):
        raise AssertionError("on-prem simulation attempted real network access")

    monkeypatch.setattr(socket, "socket", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    complete_with_knowledge_policy(42)


def test_curriculum_honours_seed_allowlist_and_reports_provenance() -> None:
    env = OnPremCurriculumEnv([1, 2, 3])
    observation, info = env.reset(seed=8, options={"topology_seed": 2})
    assert env.observation_space.contains(observation)
    assert info["topology_seed"] == 2
    assert info["topology_hash"] == topology_digest(env.true_topology)
    assert info["topology_config_hash"] == env.config.digest()
    with pytest.raises(ValueError, match="outside this split"):
        env.reset(options={"topology_seed": 2001})
