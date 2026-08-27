"""Hybrid cloud/legacy topology families and held-out curriculum environment."""

from __future__ import annotations

import random
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import gymnasium as gym

from rlredteam.enterprise.environment import EnterpriseCyberEnv
from rlredteam.enterprise.model import (
    EdgeType,
    EnterpriseEdge,
    EnterpriseGraph,
    EnterpriseNode,
    NodeType,
    Vulnerability,
)


class HybridFamily(StrEnum):
    LEGACY_TO_CLOUD = "legacy_to_cloud"
    CLOUD_TO_LEGACY = "cloud_to_legacy"
    PARTNER_SAAS_BRIDGE = "partner_saas_bridge"


@dataclass(frozen=True, slots=True)
class HybridGeneratorConfig:
    decoy_hosts: int = 3
    max_nodes: int = 40
    max_vulnerabilities: int = 16

    def __post_init__(self) -> None:
        if self.decoy_hosts < 0:
            raise ValueError("decoy_hosts cannot be negative")
        if self.max_nodes < 24 + self.decoy_hosts:
            raise ValueError("max_nodes is too small for the generated hybrid graph")


@dataclass(frozen=True, slots=True)
class GeneralisationSplit:
    """Pre-registered, disjoint topology seeds for one experiment."""

    train: tuple[int, ...] = tuple(range(1, 61))
    validation: tuple[int, ...] = tuple(range(1001, 1021))
    test: tuple[int, ...] = tuple(range(2001, 2021))

    def __post_init__(self) -> None:
        sets = [set(self.train), set(self.validation), set(self.test)]
        if not all(sets):
            raise ValueError("train, validation and test seed sets must be non-empty")
        if sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2]:
            raise ValueError("generalisation seed sets must be disjoint")


def family_for_seed(seed: int) -> HybridFamily:
    return tuple(HybridFamily)[seed % len(HybridFamily)]


def generate_hybrid_enterprise(
    seed: int,
    *,
    family: HybridFamily | str | None = None,
    config: HybridGeneratorConfig | None = None,
) -> EnterpriseGraph:
    """Generate one hybrid graph from a structurally distinct family."""
    config = config or HybridGeneratorConfig()
    selected = HybridFamily(family) if family is not None else family_for_seed(seed)
    rng = random.Random(seed)
    nodes: dict[str, EnterpriseNode] = {}
    edges: list[EnterpriseEdge] = []

    def node(node_id: str, kind: NodeType, name: str, **attributes: Any) -> None:
        nodes[node_id] = EnterpriseNode(node_id, kind, name, attributes)

    def edge(source: str, target: str, kind: EdgeType, **attributes: Any) -> None:
        edges.append(EnterpriseEdge(source, target, kind, attributes))

    family_details = {
        HybridFamily.LEGACY_TO_CLOUD: {
            "entry_type": NodeType.LEGACY_HOST,
            "entry_name": "Legacy public portal",
            "entry_os": "windows-server-2012",
            "pivot_type": NodeType.HOST,
            "pivot_name": "On-premises identity bridge",
            "data_type": NodeType.CLOUD_WORKLOAD,
            "data_name": "Cloud application workload",
            "store_type": NodeType.STORAGE,
            "store_name": "Cloud object storage",
            "product": "apache_http_server",
            "version": "2.4.50",
            "vuln": "CVE-2021-42013",
            "cvss": 9.8,
            "boundary": "legacy_to_cloud",
        },
        HybridFamily.CLOUD_TO_LEGACY: {
            "entry_type": NodeType.CLOUD_WORKLOAD,
            "entry_name": "Cloud bastion workload",
            "entry_os": "ubuntu-22.04",
            "pivot_type": NodeType.HOST,
            "pivot_name": "Site-to-site VPN connector",
            "data_type": NodeType.LEGACY_HOST,
            "data_name": "Legacy database host",
            "store_type": NodeType.DATABASE,
            "store_name": "Legacy finance database",
            "product": "openssh",
            "version": "9.2p1",
            "vuln": "CVE-2024-6387",
            "cvss": 8.1,
            "boundary": "cloud_to_legacy",
        },
        HybridFamily.PARTNER_SAAS_BRIDGE: {
            "entry_type": NodeType.CLOUD_WORKLOAD,
            "entry_name": "Partner-facing SaaS gateway",
            "entry_os": "container-linux",
            "pivot_type": NodeType.LEGACY_HOST,
            "pivot_name": "Legacy federation server",
            "data_type": NodeType.CLOUD_WORKLOAD,
            "data_name": "Private cloud workload",
            "store_type": NodeType.STORAGE,
            "store_name": "Private cloud file store",
            "product": "synthetic_saas_gateway",
            "version": "3.2",
            "vuln": "SYN-SAAS-001",
            "cvss": 7.7,
            "boundary": "partner_to_legacy_to_cloud",
        },
    }[selected]

    node("entry_internet", NodeType.ENTRY_POINT, "Internet / partner entry")
    node("network_edge", NodeType.NETWORK_SEGMENT, "External services zone")
    node("network_legacy", NodeType.NETWORK_SEGMENT, "Legacy on-premises zone")
    node("network_cloud", NodeType.CLOUD_NETWORK, "Cloud VPC/VNet", cidr="10.80.0.0/16")
    node("control_edge", NodeType.SECURITY_CONTROL, "Edge firewall/WAF", control="waf")
    node("control_cloud", NodeType.SECURITY_CONTROL, "Cloud security group", control="sg")
    node("cloud_account", NodeType.CLOUD_ACCOUNT, "Enterprise cloud account", tenant="synthetic")
    node("identity_provider", NodeType.IDENTITY_PROVIDER, "Hybrid identity provider")
    node(
        "host_entry",
        family_details["entry_type"],
        family_details["entry_name"],
        os=family_details["entry_os"],
        environment="cloud"
        if family_details["entry_type"] == NodeType.CLOUD_WORKLOAD
        else "legacy",
    )
    node(
        "service_entry",
        NodeType.SERVICE,
        "Externally reachable service",
        port=443 if selected != HybridFamily.CLOUD_TO_LEGACY else 22,
        product=family_details["product"],
        version=family_details["version"],
    )
    node("application_entry", NodeType.APPLICATION, "Entry application")
    node("api_bridge", NodeType.API, "Hybrid integration API")
    node("identity_bridge", NodeType.IDENTITY, "Federated service identity")
    node("iam_role", NodeType.IAM_ROLE, "Cloud workload role", privilege="application")
    node(
        "host_pivot",
        family_details["pivot_type"],
        family_details["pivot_name"],
        environment="hybrid_bridge",
    )
    node(
        "host_data",
        family_details["data_type"],
        family_details["data_name"],
        environment="cloud"
        if family_details["data_type"] == NodeType.CLOUD_WORKLOAD
        else "legacy",
    )
    node("storage_target", family_details["store_type"], family_details["store_name"])
    node(
        "asset_crown",
        NodeType.ASSET,
        "Restricted hybrid business data",
        classification="restricted",
        value=100,
    )

    edge("entry_internet", "network_edge", EdgeType.CONNECTS)
    edge("control_edge", "network_edge", EdgeType.PROTECTS)
    edge("host_entry", "network_edge", EdgeType.LOCATED_IN)
    edge("host_entry", "service_entry", EdgeType.HOSTS)
    edge("service_entry", "application_entry", EdgeType.EXPOSES)
    edge("application_entry", "api_bridge", EdgeType.CALLS)
    edge("host_pivot", "api_bridge", EdgeType.HOSTS)
    edge("host_entry", "identity_bridge", EdgeType.YIELDS_CREDENTIAL)
    edge("identity_bridge", "host_pivot", EdgeType.HAS_ACCESS, privilege="user")
    edge("identity_bridge", "storage_target", EdgeType.HAS_ACCESS, privilege="read")
    edge(
        "host_pivot",
        "host_data",
        EdgeType.PIVOTS_TO,
        trust_boundary=family_details["boundary"],
    )
    edge("host_data", "storage_target", EdgeType.HOSTS)
    edge("storage_target", "asset_crown", EdgeType.CONTAINS)
    edge("identity_provider", "identity_bridge", EdgeType.TRUSTS, federation=True)
    edge("identity_bridge", "iam_role", EdgeType.TRUSTS, federation=True)
    edge("iam_role", "cloud_account", EdgeType.HAS_ACCESS)
    edge("cloud_account", "network_cloud", EdgeType.CONTAINS)
    edge("control_cloud", "network_cloud", EdgeType.PROTECTS)
    edge(
        "network_legacy",
        "network_cloud",
        EdgeType.CONNECTS,
        trust_boundary=family_details["boundary"],
    )

    for index in range(config.decoy_hosts):
        decoy_id = f"host_decoy_{index + 1}"
        decoy_type = rng.choice([NodeType.HOST, NodeType.LEGACY_HOST, NodeType.CLOUD_WORKLOAD])
        node(
            decoy_id,
            decoy_type,
            f"Decoy workload {index + 1}",
            os=rng.choice(["windows-2008", "linux", "container-linux"]),
            environment=rng.choice(["legacy", "on_premises", "cloud"]),
        )
        zone = "network_cloud" if decoy_type == NodeType.CLOUD_WORKLOAD else "network_legacy"
        edge(decoy_id, zone, EdgeType.LOCATED_IN)

    vulnerability = Vulnerability(
        id=str(family_details["vuln"]),
        target="service_entry",
        cvss=float(family_details["cvss"]),
        exploit_probability=1.0,
        grants_access_to="host_entry",
        affected_product=str(family_details["product"]),
        affected_versions=(str(family_details["version"]),),
        description=f"Applicable entry weakness for {selected.value}",
    )
    return EnterpriseGraph(
        name=f"hybrid-{selected.value}-seed-{seed}",
        nodes=nodes,
        edges=edges,
        vulnerabilities={vulnerability.id: vulnerability},
        entry_points=("entry_internet",),
        crown_jewels=("asset_crown",),
    )


class HybridCurriculumEnv(gym.Env):
    """Regenerates a fixed-space environment from a seed pool each episode."""

    metadata = {"render_modes": ["ansi"]}

    def __init__(
        self,
        topology_seeds: tuple[int, ...] | list[int],
        *,
        config: HybridGeneratorConfig | None = None,
        max_steps: int = 200,
    ) -> None:
        super().__init__()
        if not topology_seeds:
            raise ValueError("at least one topology seed is required")
        self.topology_seeds = tuple(int(seed) for seed in topology_seeds)
        self.config = config or HybridGeneratorConfig()
        self.max_steps = max_steps
        self.topology_seed = self.topology_seeds[0]
        self.family = family_for_seed(self.topology_seed)
        self._env = self._make(self.topology_seed)
        self.action_space = self._env.action_space
        self.observation_space = self._env.observation_space

    def _make(self, seed: int) -> EnterpriseCyberEnv:
        graph = generate_hybrid_enterprise(seed, config=self.config)
        return EnterpriseCyberEnv(
            graph,
            max_steps=self.max_steps,
            max_nodes=self.config.max_nodes,
            max_vulnerabilities=self.config.max_vulnerabilities,
        )

    @property
    def graph(self) -> EnterpriseGraph:
        return self._env.graph

    @property
    def knowledge(self):
        return self._env.knowledge

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        requested = (options or {}).get("topology_seed")
        if requested is not None:
            if int(requested) not in self.topology_seeds:
                raise ValueError("requested topology seed is outside this environment's split")
            self.topology_seed = int(requested)
        else:
            self.topology_seed = int(self.np_random.choice(self.topology_seeds))
        self.family = family_for_seed(self.topology_seed)
        self._env = self._make(self.topology_seed)
        if self._env.action_space != self.action_space:
            raise RuntimeError("hybrid action space changed across topology seeds")
        if self._env.observation_space != self.observation_space:
            raise RuntimeError("hybrid observation space changed across topology seeds")
        observation, info = self._env.reset(seed=seed)
        info.update({"topology_seed": self.topology_seed, "family": self.family.value})
        return observation, info

    def step(self, action: int):
        observation, reward, terminated, truncated, info = self._env.step(action)
        info.update({"topology_seed": self.topology_seed, "family": self.family.value})
        return observation, reward, terminated, truncated, info

    def action_index(self, *args, **kwargs) -> int:
        return self._env.action_index(*args, **kwargs)

    def attack_path(self):
        return self._env.attack_path()
