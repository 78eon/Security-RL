"""Typed enterprise graph and partial-observation simulator tests."""

from __future__ import annotations

import socket

import numpy as np
import pytest

from rlredteam.enterprise import EnterpriseCyberEnv, generate_enterprise
from rlredteam.enterprise.environment import EnterpriseActionType
from rlredteam.enterprise.model import NodeType


def act(env: EnterpriseCyberEnv, kind: EnterpriseActionType, target: str):
    return env.step(env.action_index(kind, target))


def complete_attack_path(env: EnterpriseCyberEnv):
    sequence = [
        (EnterpriseActionType.DISCOVER_NETWORK, "seg_dmz"),
        (EnterpriseActionType.ENUMERATE_HOST, "web_host"),
        (EnterpriseActionType.ENUMERATE_SERVICE, "http"),
        (EnterpriseActionType.ENUMERATE_APPLICATION, "portal"),
        (EnterpriseActionType.ASSESS_VULNERABILITY, "http"),
        (EnterpriseActionType.EXPLOIT, "CVE-2021-42013"),
        (EnterpriseActionType.OBTAIN_CREDENTIAL, "svc_orders"),
        (EnterpriseActionType.AUTHENTICATE, "app_host"),
        (EnterpriseActionType.ENUMERATE_HOST, "app_host"),
        (EnterpriseActionType.PIVOT, "db_host"),
        (EnterpriseActionType.ENUMERATE_HOST, "db_host"),
        (EnterpriseActionType.ENUMERATE_APPLICATION, "customer_db"),
        (EnterpriseActionType.AUTHENTICATE, "customer_db"),
        (EnterpriseActionType.ACCESS_ASSET, "customer_records"),
    ]
    result = None
    for kind, target in sequence:
        result = act(env, kind, target)
    return result


def test_generator_is_deterministic_and_typed() -> None:
    first = generate_enterprise(42)
    second = generate_enterprise(42)
    other = generate_enterprise(43)

    assert first.to_dict() == second.to_dict()
    assert first.to_dict() != other.to_dict()
    assert {node.type for node in first.nodes.values()} >= {
        NodeType.NETWORK_SEGMENT,
        NodeType.SECURITY_CONTROL,
        NodeType.HOST,
        NodeType.SERVICE,
        NodeType.APPLICATION,
        NodeType.API,
        NodeType.IDENTITY,
        NodeType.DATABASE,
        NodeType.ASSET,
    }
    web_vuln = first.vulnerabilities["CVE-2021-42013"]
    assert web_vuln.applies_to(first.nodes["http"])
    assert first.nodes["http"].attributes["version"] == "2.4.50"


def test_initial_observation_hides_ground_truth() -> None:
    graph = generate_enterprise(42)
    env = EnterpriseCyberEnv(graph)
    observation, info = env.reset(seed=42)

    assert env.observation_space.contains(observation)
    assert info["known_nodes"] < len(graph.nodes)
    assert "portal" not in env.knowledge.discovered
    assert not env.knowledge.known_vulnerabilities
    assert np.count_nonzero(info["action_mask"]) > 0


def test_discovery_reveals_information_incrementally() -> None:
    env = EnterpriseCyberEnv(generate_enterprise(42))
    env.reset(seed=42)

    _, reward, terminated, truncated, info = act(
        env, EnterpriseActionType.DISCOVER_NETWORK, "seg_dmz"
    )
    assert reward > 0
    assert not terminated and not truncated
    assert "web_host" in env.knowledge.discovered
    assert "portal" not in env.knowledge.discovered
    assert info["event"].outcomes

    act(env, EnterpriseActionType.ENUMERATE_HOST, "web_host")
    assert "http" in env.knowledge.discovered
    assert "portal" not in env.knowledge.discovered

    act(env, EnterpriseActionType.ENUMERATE_SERVICE, "http")
    assert "portal" in env.knowledge.discovered


def test_invalid_action_is_penalised_without_changing_knowledge() -> None:
    env = EnterpriseCyberEnv(generate_enterprise(42))
    env.reset(seed=42)
    before = env._snapshot()

    _, reward, terminated, truncated, info = act(
        env, EnterpriseActionType.ACCESS_ASSET, "customer_records"
    )

    assert reward == pytest.approx(-5.0)
    assert env._snapshot() == before
    assert not terminated and not truncated
    assert not info["event"].success
    assert "not been discovered" in info["event"].reason


def test_canonical_path_reaches_crown_jewel_and_is_reconstructed() -> None:
    env = EnterpriseCyberEnv(generate_enterprise(42), max_steps=50)
    env.reset(seed=42)
    observation, reward, terminated, truncated, info = complete_attack_path(env)

    assert env.observation_space.contains(observation)
    assert reward == pytest.approx(100.0)
    assert terminated and not truncated
    assert info["goal_reached"]
    assert "customer_records" in env.knowledge.accessed_assets
    path = env.attack_path()
    assert path
    assert path[-1].action.type == EnterpriseActionType.ACCESS_ASSET
    assert all(event.success and event.state_changed for event in path)
    assert any("credential:svc_orders" in event.outcomes for event in path)
    assert any("pivoted:db_host" in event.outcomes for event in path)


def test_reset_replays_stochastic_exploit_from_same_seed() -> None:
    env = EnterpriseCyberEnv(generate_enterprise(42))
    first, _ = env.reset(seed=77)
    second, _ = env.reset(seed=77)
    assert np.array_equal(first, second)


def test_build_and_episode_use_no_network(monkeypatch) -> None:
    def refuse(*args, **kwargs):
        raise AssertionError("enterprise simulator attempted real network access")

    monkeypatch.setattr(socket, "socket", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    env = EnterpriseCyberEnv(generate_enterprise(42))
    env.reset(seed=42)
    complete_attack_path(env)
    assert env.knowledge.accessed_assets == {"customer_records"}


@pytest.mark.parametrize("bad_action", [-1, 10**6, 1.5])
def test_out_of_range_actions_are_rejected(bad_action) -> None:
    env = EnterpriseCyberEnv(generate_enterprise(42))
    env.reset(seed=42)
    with pytest.raises(ValueError, match="outside"):
        env.step(bad_action)
