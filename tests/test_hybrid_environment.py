"""Hybrid topology diversity, fixed spaces and held-out split tests."""

from __future__ import annotations

from rlredteam.enterprise.environment import EnterpriseActionType, EnterpriseCyberEnv
from rlredteam.enterprise.hybrid import (
    GeneralisationSplit,
    HybridCurriculumEnv,
    HybridFamily,
    HybridGeneratorConfig,
    generate_hybrid_enterprise,
)
from rlredteam.enterprise.model import EdgeType, NodeType


def run_feasible_path(env: EnterpriseCyberEnv) -> bool:
    vuln_id = next(iter(env.graph.vulnerabilities))
    sequence = (
        (EnterpriseActionType.DISCOVER_NETWORK, "network_edge"),
        (EnterpriseActionType.ENUMERATE_HOST, "host_entry"),
        (EnterpriseActionType.ENUMERATE_SERVICE, "service_entry"),
        (EnterpriseActionType.ENUMERATE_APPLICATION, "application_entry"),
        (EnterpriseActionType.ASSESS_VULNERABILITY, "service_entry"),
        (EnterpriseActionType.EXPLOIT, vuln_id),
        (EnterpriseActionType.OBTAIN_CREDENTIAL, "identity_bridge"),
        (EnterpriseActionType.AUTHENTICATE, "host_pivot"),
        (EnterpriseActionType.ENUMERATE_HOST, "host_pivot"),
        (EnterpriseActionType.PIVOT, "host_data"),
        (EnterpriseActionType.ENUMERATE_HOST, "host_data"),
        (EnterpriseActionType.ENUMERATE_APPLICATION, "storage_target"),
        (EnterpriseActionType.AUTHENTICATE, "storage_target"),
        (EnterpriseActionType.ACCESS_ASSET, "asset_crown"),
    )
    terminated = False
    for kind, target in sequence:
        _, _, terminated, truncated, info = env.step(env.action_index(kind, target))
        assert info["event"].success, info["event"].reason
        assert not truncated
    return terminated


def test_every_family_contains_cloud_legacy_and_a_boundary() -> None:
    for family in HybridFamily:
        graph = generate_hybrid_enterprise(42, family=family)
        types = {node.type for node in graph.nodes.values()}
        assert NodeType.LEGACY_HOST in types
        assert types & {
            NodeType.CLOUD_ACCOUNT,
            NodeType.CLOUD_NETWORK,
            NodeType.CLOUD_WORKLOAD,
            NodeType.STORAGE,
        }
        assert any(edge.attributes.get("trust_boundary") for edge in graph.edges)
        assert any(edge.type == EdgeType.TRUSTS for edge in graph.edges)


def test_families_are_structurally_distinct() -> None:
    signatures = set()
    for family in HybridFamily:
        graph = generate_hybrid_enterprise(42, family=family)
        signatures.add(
            tuple(
                (node.id, node.type.value)
                for node in sorted(graph.nodes.values(), key=lambda item: item.id)
            )
        )
    assert len(signatures) == len(HybridFamily)


def test_fixed_spaces_match_across_families_and_seeds() -> None:
    config = HybridGeneratorConfig()
    spaces = []
    for seed in (1, 2, 3, 2001, 2002, 2003):
        env = EnterpriseCyberEnv(
            generate_hybrid_enterprise(seed, config=config),
            max_nodes=config.max_nodes,
            max_vulnerabilities=config.max_vulnerabilities,
        )
        spaces.append((env.observation_space, env.action_space))
    assert all(pair == spaces[0] for pair in spaces)


def test_every_family_has_a_feasible_cross_boundary_path() -> None:
    config = HybridGeneratorConfig()
    for family in HybridFamily:
        env = EnterpriseCyberEnv(
            generate_hybrid_enterprise(42, family=family, config=config),
            max_nodes=config.max_nodes,
            max_vulnerabilities=config.max_vulnerabilities,
        )
        env.reset(seed=42)
        assert run_feasible_path(env)
        assert env.knowledge.accessed_assets == {"asset_crown"}


def test_generalisation_splits_are_disjoint() -> None:
    split = GeneralisationSplit()
    assert not set(split.train) & set(split.validation)
    assert not set(split.train) & set(split.test)
    assert not set(split.validation) & set(split.test)


def test_curriculum_reset_selects_only_its_split() -> None:
    seeds = (1, 2, 3)
    env = HybridCurriculumEnv(seeds)
    for reset_seed in range(10):
        observation, info = env.reset(seed=reset_seed)
        assert env.observation_space.contains(observation)
        assert info["topology_seed"] in seeds
        assert info["family"] in {family.value for family in HybridFamily}
