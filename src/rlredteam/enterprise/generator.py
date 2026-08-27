"""Seeded generator for valid heterogeneous enterprise environments."""

from __future__ import annotations

import random
from dataclasses import dataclass

from rlredteam.enterprise.model import (
    EdgeType,
    EnterpriseEdge,
    EnterpriseGraph,
    EnterpriseNode,
    NodeType,
    Vulnerability,
)


@dataclass(frozen=True, slots=True)
class EnterpriseGeneratorConfig:
    extra_workstations: int = 2
    extra_services: int = 2
    include_cloud: bool = True

    def __post_init__(self) -> None:
        if self.extra_workstations < 0 or self.extra_services < 0:
            raise ValueError("extra entity counts cannot be negative")


def generate_enterprise(
    seed: int,
    config: EnterpriseGeneratorConfig | None = None,
) -> EnterpriseGraph:
    """Generate one reproducible graph with at least one feasible attack path.

    The graph includes a guaranteed web-to-database path plus seeded distractor
    hosts and services.  Values and exploit probabilities vary by seed, while
    identifiers remain stable enough for experiment tooling and demonstrations.
    """
    config = config or EnterpriseGeneratorConfig()
    rng = random.Random(seed)
    nodes: dict[str, EnterpriseNode] = {}
    edges: list[EnterpriseEdge] = []

    def node(node_id: str, kind: NodeType, name: str, **attributes) -> None:
        nodes[node_id] = EnterpriseNode(node_id, kind, name, attributes)

    def edge(source: str, target: str, kind: EdgeType, **attributes) -> None:
        edges.append(EnterpriseEdge(source, target, kind, attributes))

    node("internet", NodeType.ENTRY_POINT, "Internet")
    node("seg_dmz", NodeType.NETWORK_SEGMENT, "DMZ", cidr="10.10.0.0/24")
    node("seg_app", NodeType.NETWORK_SEGMENT, "Application", cidr="10.20.0.0/24")
    node("seg_data", NodeType.NETWORK_SEGMENT, "Data", cidr="10.30.0.0/24")
    node("fw_edge", NodeType.SECURITY_CONTROL, "Edge firewall", control="firewall")
    node("web_host", NodeType.HOST, "Public web server", os="linux", ip="10.10.0.10")
    node(
        "http",
        NodeType.SERVICE,
        "HTTPS",
        port=443,
        protocol="tcp",
        product="apache_http_server",
        version="2.4.50",
    )
    node("portal", NodeType.APPLICATION, "Customer portal", framework="synthetic-web")
    node(
        "app_host",
        NodeType.HOST,
        "Application server",
        os="linux",
        ip="10.20.0.10",
        product="synthetic_linux_host",
        version="1.0",
    )
    node("orders_api", NodeType.API, "Orders API", protocol="https")
    node("svc_orders", NodeType.IDENTITY, "Orders service account", privilege="service")
    node("db_host", NodeType.HOST, "Database server", os="linux", ip="10.30.0.10")
    node("customer_db", NodeType.DATABASE, "Customer database", engine="postgresql")
    node(
        "customer_records",
        NodeType.ASSET,
        "Customer records",
        classification="restricted",
        value=100,
    )
    node("edr_app", NodeType.SECURITY_CONTROL, "Application EDR", control="edr")

    edge("internet", "seg_dmz", EdgeType.CONNECTS)
    edge("fw_edge", "seg_dmz", EdgeType.PROTECTS)
    edge("seg_dmz", "seg_app", EdgeType.CONNECTS, filtered=True)
    edge("seg_app", "seg_data", EdgeType.CONNECTS, filtered=True)
    edge("web_host", "seg_dmz", EdgeType.LOCATED_IN)
    edge("app_host", "seg_app", EdgeType.LOCATED_IN)
    edge("db_host", "seg_data", EdgeType.LOCATED_IN)
    edge("web_host", "http", EdgeType.HOSTS)
    edge("http", "portal", EdgeType.EXPOSES)
    edge("portal", "orders_api", EdgeType.CALLS)
    edge("app_host", "orders_api", EdgeType.HOSTS)
    edge("web_host", "svc_orders", EdgeType.YIELDS_CREDENTIAL)
    edge("svc_orders", "app_host", EdgeType.HAS_ACCESS, privilege="user")
    edge("svc_orders", "customer_db", EdgeType.HAS_ACCESS, privilege="read")
    edge("app_host", "db_host", EdgeType.PIVOTS_TO)
    edge("db_host", "customer_db", EdgeType.HOSTS)
    edge("customer_db", "customer_records", EdgeType.CONTAINS)
    edge("edr_app", "app_host", EdgeType.PROTECTS, detection=rng.uniform(0.1, 0.4))

    service_names = ["ssh", "smb", "rdp", "dns", "ldap"]
    for index in range(config.extra_workstations):
        host_id = f"workstation_{index + 1}"
        node(
            host_id,
            NodeType.HOST,
            f"Employee workstation {index + 1}",
            os=rng.choice(["windows", "linux"]),
            ip=f"10.20.0.{30 + index}",
        )
        edge(host_id, "seg_app", EdgeType.LOCATED_IN)

    for index in range(config.extra_services):
        service = rng.choice(service_names)
        service_id = f"service_{index + 1}_{service}"
        host_id = rng.choice(["app_host", "db_host", "web_host"])
        node(
            service_id,
            NodeType.SERVICE,
            service.upper(),
            port={"ssh": 22, "smb": 445, "rdp": 3389, "dns": 53, "ldap": 389}[service],
        )
        edge(host_id, service_id, EdgeType.HOSTS)

    if config.include_cloud:
        node("cloud_backup", NodeType.CLOUD_RESOURCE, "Cloud backup", provider="synthetic")
        edge("customer_db", "cloud_backup", EdgeType.CALLS)

    vulnerabilities = {
        "CVE-2021-42013": Vulnerability(
            id="CVE-2021-42013",
            target="http",
            cvss=9.8,
            exploit_probability=1.0,
            grants_access_to="web_host",
            affected_product="apache_http_server",
            affected_versions=("2.4.50",),
            description="Frozen catalogue analogue for Apache HTTP Server 2.4.50",
        ),
        "SYN-APP-PRIV-001": Vulnerability(
            id="SYN-APP-PRIV-001",
            target="app_host",
            cvss=round(rng.uniform(6.0, 8.5), 1),
            exploit_probability=1.0,
            grants_access_to="app_host",
            affected_product="synthetic_linux_host",
            affected_versions=("1.0",),
            privilege="root",
            description="Synthetic local privilege-escalation flaw",
        ),
    }

    return EnterpriseGraph(
        name=f"enterprise-seed-{seed}",
        nodes=nodes,
        edges=edges,
        vulnerabilities=vulnerabilities,
        entry_points=("internet",),
        crown_jewels=("customer_records",),
    )
