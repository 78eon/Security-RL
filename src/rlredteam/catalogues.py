"""Validated, versioned catalogues for simulated-world and framework data."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WORLD_CATALOGUE = REPO_ROOT / "configs/catalogues/simulated_world.yaml"
DEFAULT_MITRE_CATALOGUE = REPO_ROOT / "configs/catalogues/mitre.yaml"


def _read_catalogue(path: Path) -> dict[str, Any]:
    try:
        document = yaml.safe_load(path.read_text())
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"cannot load catalogue {path}: {exc}") from exc
    if not isinstance(document, dict) or not isinstance(document.get("catalogue"), dict):
        raise ValueError(f"catalogue root is missing in {path}")
    return document["catalogue"]


def _version(raw: dict[str, Any], path: Path) -> str:
    value = str(raw.get("version", "")).strip()
    if not value or not value.endswith("-v1"):
        raise ValueError(f"catalogue version must be a frozen *-v1 identifier: {path}")
    return value


@dataclass(frozen=True, slots=True)
class ServiceDefinition:
    product: str
    version: str
    port: int
    protocol: str
    vulnerability: str | None = None


@dataclass(frozen=True, slots=True)
class CrownJewelTemplate:
    node_type: str
    name: str
    classification: str
    value: int


@dataclass(frozen=True, slots=True)
class WorldCatalogue:
    version: str
    services: dict[str, tuple[ServiceDefinition, ...]]
    os_choices: dict[str, tuple[str, ...]]
    application_types: tuple[str, ...]
    identity_types: tuple[str, ...]
    cloud_resource_types: tuple[str, ...]
    vulnerability_defaults: dict[str, float | str]
    crown_jewel_templates: dict[str, CrownJewelTemplate]
    generator_bindings: dict[str, dict[str, Any]]
    digest: str

    def service_pool(self, name: str) -> tuple[ServiceDefinition, ...]:
        try:
            return self.services[name]
        except KeyError as exc:
            raise ValueError(f"unknown service pool: {name}") from exc

    def os_pool(self, name: str) -> tuple[str, ...]:
        try:
            return self.os_choices[name]
        except KeyError as exc:
            raise ValueError(f"unknown OS pool: {name}") from exc

    def crown_jewel(self, name: str) -> CrownJewelTemplate:
        try:
            return self.crown_jewel_templates[name]
        except KeyError as exc:
            raise ValueError(f"unknown crown-jewel template: {name}") from exc


def load_world_catalogue(path: Path | None = None) -> WorldCatalogue:
    selected = path or DEFAULT_WORLD_CATALOGUE
    raw = _read_catalogue(selected)
    version = _version(raw, selected)
    services: dict[str, tuple[ServiceDefinition, ...]] = {}
    for pool, values in (raw.get("services") or {}).items():
        if not isinstance(values, list) or not values:
            raise ValueError(f"service pool {pool!r} must be a non-empty list")
        parsed: list[ServiceDefinition] = []
        for item in values:
            service = ServiceDefinition(
                product=str(item["product"]),
                version=str(item["version"]),
                port=int(item["port"]),
                protocol=str(item["protocol"]),
                vulnerability=(
                    str(item["vulnerability"]) if item.get("vulnerability") else None
                ),
            )
            if not service.product or not service.version or not 1 <= service.port <= 65535:
                raise ValueError(f"invalid service in pool {pool!r}: {item!r}")
            if service.protocol not in {"tcp", "udp"}:
                raise ValueError(f"invalid service protocol: {service.protocol}")
            parsed.append(service)
        services[str(pool)] = tuple(parsed)
    os_choices = {
        str(name): tuple(map(str, values))
        for name, values in (raw.get("os_choices") or {}).items()
    }
    if not services or not os_choices or any(not values for values in os_choices.values()):
        raise ValueError("services and OS choice pools are required")
    vulnerabilities = raw.get("vulnerabilities", {}).get("defaults", {})
    cvss_min, cvss_max = float(vulnerabilities["cvss_min"]), float(
        vulnerabilities["cvss_max"]
    )
    probability = float(vulnerabilities["exploit_probability"])
    if not 0 <= cvss_min <= cvss_max <= 10 or not 0 <= probability <= 1:
        raise ValueError("invalid vulnerability defaults")
    templates = {
        str(name): CrownJewelTemplate(
            node_type=str(item["node_type"]),
            name=str(item["name"]),
            classification=str(item["classification"]),
            value=int(item["value"]),
        )
        for name, item in (raw.get("crown_jewel_templates") or {}).items()
    }
    if not templates:
        raise ValueError("at least one crown-jewel template is required")
    typed_collections = {
        "application_types": tuple(map(str, raw.get("application_types") or ())),
        "identity_types": tuple(map(str, raw.get("identity_types") or ())),
        "cloud_resource_types": tuple(map(str, raw.get("cloud_resource_types") or ())),
    }
    if any(not values or len(values) != len(set(values)) for values in typed_collections.values()):
        raise ValueError("world type catalogues must be non-empty and unique")
    from rlredteam.enterprise.model import NodeType

    try:
        for values in typed_collections.values():
            tuple(NodeType(value) for value in values)
    except ValueError as exc:
        raise ValueError(f"world catalogue contains an unknown node type: {exc}") from exc
    bindings = raw.get("generator_bindings")
    if not isinstance(bindings, dict) or not {"onprem", "enterprise_profile"} <= set(
        bindings
    ):
        raise ValueError("world catalogue generator bindings are incomplete")
    canonical = json.dumps(raw, sort_keys=True, separators=(",", ":"))
    return WorldCatalogue(
        version=version,
        services=services,
        os_choices=os_choices,
        application_types=typed_collections["application_types"],
        identity_types=typed_collections["identity_types"],
        cloud_resource_types=typed_collections["cloud_resource_types"],
        vulnerability_defaults={
            "cvss_min": cvss_min,
            "cvss_max": cvss_max,
            "exploit_probability": probability,
            "privilege": str(vulnerabilities["privilege"]),
        },
        crown_jewel_templates=templates,
        generator_bindings={
            str(name): dict(value)
            for name, value in bindings.items()
        },
        digest=hashlib.sha256(canonical.encode()).hexdigest(),
    )


def load_mitre_catalogue(path: Path | None = None) -> dict[str, Any]:
    selected = path or DEFAULT_MITRE_CATALOGUE
    raw = _read_catalogue(selected)
    _version(raw, selected)
    frameworks = raw.get("frameworks")
    if not isinstance(frameworks, dict) or set(frameworks) != {"attack-enterprise", "atlas"}:
        raise ValueError("MITRE catalogue requires ATT&CK Enterprise and ATLAS")
    ids: set[str] = set()
    framework_ids: dict[str, set[str]] = {}
    for framework, definition in frameworks.items():
        techniques = definition.get("techniques") if isinstance(definition, dict) else None
        if not isinstance(techniques, list) or not techniques:
            raise ValueError(f"MITRE framework {framework!r} has no techniques")
        framework_ids[framework] = set()
        for technique in techniques:
            required = {
                "tactic_id", "tactic_name", "technique_id", "technique_name", "source_url"
            }
            if not isinstance(technique, dict) or not required <= set(technique):
                raise ValueError(f"invalid MITRE technique row: {technique!r}")
            technique_id = str(technique["technique_id"])
            if technique_id in ids:
                raise ValueError(f"duplicate MITRE technique ID: {technique_id}")
            if not str(technique["source_url"]).startswith("https://"):
                raise ValueError(f"MITRE source URL must use HTTPS: {technique_id}")
            ids.add(technique_id)
            framework_ids[framework].add(technique_id)
    for field, framework in (
        ("simulator_action_mappings", "attack-enterprise"),
        ("ai_behavior_mappings", "atlas"),
    ):
        mappings = raw.get(field)
        if not isinstance(mappings, dict):
            raise ValueError(f"MITRE catalogue is missing {field}")
        unknown = set(map(str, mappings.values())) - framework_ids[framework]
        if unknown:
            raise ValueError(f"{field} references unknown techniques: {sorted(unknown)}")
    return raw
