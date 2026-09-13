"""Seeded generator for catalogue-defined enterprise environments."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from rlredteam.catalogues import load_world_catalogue
from rlredteam.enterprise.model import (
    EdgeType,
    EnterpriseEdge,
    EnterpriseGraph,
    EnterpriseNode,
    NodeType,
    Vulnerability,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_REFERENCE_CATALOGUE = REPO_ROOT / "configs/catalogues/reference_enterprise.yaml"


@dataclass(frozen=True, slots=True)
class EnterpriseGeneratorConfig:
    extra_workstations: int = 2
    extra_services: int = 2
    include_cloud: bool = False

    def __post_init__(self) -> None:
        if self.extra_workstations < 0 or self.extra_services < 0:
            raise ValueError("extra entity counts cannot be negative")


def load_reference_enterprise_catalogue(path: Path | None = None) -> dict[str, Any]:
    """Load and minimally validate the versioned reference-world document."""
    selected = path or DEFAULT_REFERENCE_CATALOGUE
    document = yaml.safe_load(selected.read_text())
    raw = document.get("catalogue") if isinstance(document, dict) else None
    if not isinstance(raw, dict):
        raise ValueError("reference enterprise catalogue root is missing")
    if raw.get("version") != "security-rl-reference-enterprise-v1":
        raise ValueError("unsupported reference enterprise catalogue version")
    required = {
        "nodes",
        "edges",
        "vulnerabilities",
        "entry_points",
        "crown_jewels",
        "extras",
        "demo_path",
        "demo_max_steps",
    }
    if not required <= set(raw):
        raise ValueError("reference enterprise catalogue is incomplete")
    collections = {
        "nodes", "edges", "vulnerabilities", "entry_points", "crown_jewels", "demo_path"
    }
    if not all(isinstance(raw[field], list) and raw[field] for field in collections):
        raise ValueError("reference enterprise catalogue collections cannot be empty")
    if not isinstance(raw["extras"], dict):
        raise ValueError("reference enterprise extras must be a mapping")
    return raw


def generate_enterprise(
    seed: int,
    config: EnterpriseGeneratorConfig | None = None,
    catalogue_path: Path | None = None,
) -> EnterpriseGraph:
    """Generate one reproducible graph from validated catalogue data."""
    config = config or EnterpriseGeneratorConfig()
    raw = load_reference_enterprise_catalogue(catalogue_path)
    world = load_world_catalogue()
    rng = random.Random(seed)
    nodes: dict[str, EnterpriseNode] = {}
    edges: list[EnterpriseEdge] = []

    for item in raw["nodes"]:
        node = EnterpriseNode(
            id=str(item["id"]),
            type=NodeType(item["type"]),
            name=str(item["name"]),
            attributes=dict(item.get("attributes") or {}),
        )
        if node.id in nodes:
            raise ValueError(f"duplicate reference node: {node.id}")
        nodes[node.id] = node

    for item in raw["edges"]:
        attributes = dict(item.get("attributes") or {})
        bounds = attributes.pop("detection_uniform", None)
        if bounds is not None:
            attributes["detection"] = rng.uniform(float(bounds[0]), float(bounds[1]))
        edges.append(
            EnterpriseEdge(
                source=str(item["source"]),
                target=str(item["target"]),
                type=EdgeType(item["type"]),
                attributes=attributes,
            )
        )

    extras = raw["extras"]
    workstation = extras["workstation"]
    os_pool = world.os_pool(str(workstation["os_pool"]))
    for offset in range(config.extra_workstations):
        index = offset + 1
        node_id = str(workstation["id_template"]).format(index=index)
        nodes[node_id] = EnterpriseNode(
            node_id,
            NodeType(workstation["type"]),
            str(workstation["name_template"]).format(index=index),
            {
                "os": rng.choice(os_pool),
                "ip": str(workstation["ip_template"]).format(
                    address=int(workstation["first_address"]) + offset
                ),
            },
        )
        edges.append(
            EnterpriseEdge(node_id, str(workstation["segment"]), EdgeType.LOCATED_IN)
        )

    service_extra = extras["service"]
    for offset in range(config.extra_services):
        index = offset + 1
        service = rng.choice(service_extra["pool"])
        node_id = str(service_extra["id_template"]).format(
            index=index, product=service["product"]
        )
        nodes[node_id] = EnterpriseNode(
            node_id,
            NodeType.SERVICE,
            str(service["product"]).upper(),
            {
                "port": int(service["port"]),
            },
        )
        edges.append(
            EnterpriseEdge(
                str(rng.choice(service_extra["hosts"])), node_id, EdgeType.HOSTS
            )
        )

    if config.include_cloud:
        cloud = extras["cloud"]
        node_id = str(cloud["id"])
        nodes[node_id] = EnterpriseNode(
            node_id,
            NodeType(cloud["type"]),
            str(cloud["name"]),
            dict(cloud.get("attributes") or {}),
        )
        edges.append(
            EnterpriseEdge(
                str(cloud["source"]), node_id, EdgeType(cloud["edge_type"])
            )
        )

    vulnerabilities: dict[str, Vulnerability] = {}
    for item in raw["vulnerabilities"]:
        bounds = item.get("cvss_uniform")
        cvss = (
            round(rng.uniform(float(bounds[0]), float(bounds[1])), 1)
            if bounds is not None
            else float(item["cvss"])
        )
        vulnerability = Vulnerability(
            id=str(item["id"]),
            target=str(item["target"]),
            cvss=cvss,
            exploit_probability=float(item["exploit_probability"]),
            grants_access_to=str(item["grants_access_to"]),
            affected_product=str(item["affected_product"]),
            affected_versions=tuple(map(str, item["affected_versions"])),
            privilege=str(item["privilege"]),
            description=str(item["description"]),
        )
        vulnerabilities[vulnerability.id] = vulnerability

    return EnterpriseGraph(
        name=str(raw["name_template"]).format(seed=seed),
        nodes=nodes,
        edges=edges,
        vulnerabilities=vulnerabilities,
        entry_points=tuple(map(str, raw["entry_points"])),
        crown_jewels=tuple(map(str, raw["crown_jewels"])),
    )
