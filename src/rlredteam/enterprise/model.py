"""Plain-data model for a heterogeneous enterprise graph."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any


class NodeType(StrEnum):
    ENTRY_POINT = "entry_point"
    NETWORK_SEGMENT = "network_segment"
    NETWORK_DEVICE = "network_device"
    SECURITY_CONTROL = "security_control"
    HOST = "host"
    SERVICE = "service"
    APPLICATION = "application"
    API = "api"
    IDENTITY = "identity"
    DATABASE = "database"
    CLOUD_RESOURCE = "cloud_resource"
    CLOUD_ACCOUNT = "cloud_account"
    CLOUD_NETWORK = "cloud_network"
    CLOUD_WORKLOAD = "cloud_workload"
    IAM_ROLE = "iam_role"
    STORAGE = "storage"
    LEGACY_HOST = "legacy_host"
    IDENTITY_PROVIDER = "identity_provider"
    ASSET = "asset"


class EdgeType(StrEnum):
    CONNECTS = "connects"
    LOCATED_IN = "located_in"
    HOSTS = "hosts"
    EXPOSES = "exposes"
    CALLS = "calls"
    PROTECTS = "protects"
    TRUSTS = "trusts"
    HAS_ACCESS = "has_access"
    CONTAINS = "contains"
    YIELDS_CREDENTIAL = "yields_credential"
    PIVOTS_TO = "pivots_to"


@dataclass(frozen=True, slots=True)
class EnterpriseNode:
    id: str
    type: NodeType
    name: str
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class EnterpriseEdge:
    source: str
    target: str
    type: EdgeType
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Vulnerability:
    id: str
    target: str
    cvss: float
    exploit_probability: float
    grants_access_to: str
    affected_product: str
    affected_versions: tuple[str, ...]
    privilege: str = "user"
    description: str = "synthetic vulnerability"

    def __post_init__(self) -> None:
        if not 0.0 <= self.cvss <= 10.0:
            raise ValueError("cvss must be between 0 and 10")
        if not 0.0 <= self.exploit_probability <= 1.0:
            raise ValueError("exploit_probability must be between 0 and 1")
        if self.privilege not in {"user", "root"}:
            raise ValueError("privilege must be 'user' or 'root'")
        if not self.affected_product or not self.affected_versions:
            raise ValueError("affected product and versions are required")

    def applies_to(self, node: EnterpriseNode) -> bool:
        """Whether this vulnerability genuinely matches the node fingerprint."""
        product = str(node.attributes.get("product", ""))
        version = str(node.attributes.get("version", ""))
        return product == self.affected_product and (
            "*" in self.affected_versions or version in self.affected_versions
        )


@dataclass(slots=True)
class TrueTopology:
    """Authoritative ground truth hidden from the agent during an episode.

    The environment may consult this object to resolve simulated action
    outcomes.  A policy must never receive it; policy inputs are built from
    :class:`AgentKnowledge` only.
    """

    name: str
    nodes: dict[str, EnterpriseNode]
    edges: list[EnterpriseEdge]
    vulnerabilities: dict[str, Vulnerability]
    entry_points: tuple[str, ...]
    crown_jewels: tuple[str, ...]

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if not self.nodes:
            raise ValueError("enterprise graph cannot be empty")
        if len(self.nodes) != len(set(self.nodes)):
            raise ValueError("node ids must be unique")
        for node_id, node in self.nodes.items():
            if node_id != node.id:
                raise ValueError(f"node key {node_id!r} does not match node id {node.id!r}")
        for edge in self.edges:
            if edge.source not in self.nodes or edge.target not in self.nodes:
                raise ValueError(f"edge references unknown node: {edge}")
        for vuln_id, vuln in self.vulnerabilities.items():
            if vuln_id != vuln.id:
                raise ValueError(f"vulnerability key {vuln_id!r} does not match its id")
            if vuln.target not in self.nodes or vuln.grants_access_to not in self.nodes:
                raise ValueError(f"vulnerability references unknown node: {vuln}")
            if not vuln.applies_to(self.nodes[vuln.target]):
                raise ValueError(
                    f"vulnerability {vuln.id} does not apply to the target product/version"
                )
        for node_id in (*self.entry_points, *self.crown_jewels):
            if node_id not in self.nodes:
                raise ValueError(f"unknown entry point/crown jewel {node_id!r}")
        if not self.entry_points or not self.crown_jewels:
            raise ValueError("at least one entry point and crown jewel are required")

    def outgoing(self, node_id: str, edge_type: EdgeType | None = None) -> list[EnterpriseEdge]:
        return [
            edge
            for edge in self.edges
            if edge.source == node_id and (edge_type is None or edge.type == edge_type)
        ]

    def incoming(self, node_id: str, edge_type: EdgeType | None = None) -> list[EnterpriseEdge]:
        return [
            edge
            for edge in self.edges
            if edge.target == node_id and (edge_type is None or edge.type == edge_type)
        ]

    def children(self, node_id: str, *edge_types: EdgeType) -> set[str]:
        allowed = set(edge_types)
        return {
            edge.target
            for edge in self.outgoing(node_id)
            if not allowed or edge.type in allowed
        }

    def parent_hosts(self, node_id: str) -> set[str]:
        """Hosts directly containing or exposing a component."""
        return {
            edge.source
            for edge in self.incoming(node_id)
            if edge.type in {EdgeType.HOSTS, EdgeType.EXPOSES, EdgeType.CONTAINS}
            and self.nodes[edge.source].type
            in {
                NodeType.HOST,
                NodeType.LEGACY_HOST,
                NodeType.CLOUD_WORKLOAD,
                NodeType.DATABASE,
            }
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "entry_points": list(self.entry_points),
            "crown_jewels": list(self.crown_jewels),
            "nodes": [
                {**asdict(node), "type": node.type.value}
                for node in sorted(self.nodes.values(), key=lambda item: item.id)
            ],
            "edges": [
                {**asdict(edge), "type": edge.type.value}
                for edge in sorted(
                    self.edges, key=lambda item: (item.source, item.target, item.type)
                )
            ],
            "vulnerabilities": [
                asdict(vuln)
                for vuln in sorted(self.vulnerabilities.values(), key=lambda item: item.id)
            ],
        }


# Backwards-compatible name used by the existing demo and hybrid modules.
# New partial-observability code should use ``TrueTopology`` to make the
# research boundary explicit.
EnterpriseGraph = TrueTopology
