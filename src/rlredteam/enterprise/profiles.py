"""Configuration-driven legacy, cloud and hybrid enterprise profiles."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import gymnasium as gym
import yaml

from rlredteam.enterprise.environment import EnterpriseCyberEnv
from rlredteam.enterprise.model import (
    EdgeType,
    EnterpriseEdge,
    EnterpriseNode,
    NodeType,
    TrueTopology,
    Vulnerability,
)
from rlredteam.enterprise.onprem import generate_onprem_topology, topology_digest

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PROFILE_CONFIG = REPO_ROOT / "configs" / "enterprise_profiles.yaml"


class DeploymentProfile(StrEnum):
    ON_PREMISES = "on_premises"
    LEGACY = "legacy"
    CLOUD = "cloud"
    HYBRID = "hybrid"


@dataclass(frozen=True, slots=True)
class ProfileDefinition:
    network_types: tuple[NodeType, ...]
    host_types: tuple[NodeType, ...]
    identity_types: tuple[NodeType, ...]
    store_types: tuple[NodeType, ...]
    control: str
    boundary: str


@dataclass(frozen=True, slots=True)
class EnterpriseProfileConfig:
    min_segments: int
    max_segments: int
    min_hosts: int
    max_hosts: int
    min_pivots: int
    max_pivots: int
    min_decoy_services: int
    max_decoy_services: int
    max_nodes: int
    max_vulnerabilities: int
    max_steps: int
    profiles: dict[DeploymentProfile, ProfileDefinition]

    @classmethod
    def from_yaml(cls, path: Path | None = None) -> EnterpriseProfileConfig:
        raw = yaml.safe_load((path or DEFAULT_PROFILE_CONFIG).read_text())["enterprise_profiles"]
        definitions = {
            DeploymentProfile(name): ProfileDefinition(
                network_types=tuple(NodeType(item) for item in value["network_types"]),
                host_types=tuple(NodeType(item) for item in value["host_types"]),
                identity_types=tuple(NodeType(item) for item in value["identity_types"]),
                store_types=tuple(NodeType(item) for item in value["store_types"]),
                control=str(value["control"]),
                boundary=str(value["boundary"]),
            )
            for name, value in raw.pop("profiles").items()
        }
        config = cls(profiles=definitions, **raw)
        config.validate()
        return config

    def validate(self) -> None:
        if set(self.profiles) != {
            DeploymentProfile.LEGACY,
            DeploymentProfile.CLOUD,
            DeploymentProfile.HYBRID,
        }:
            raise ValueError("legacy, cloud and hybrid definitions are required")
        if not (1 <= self.min_segments <= self.max_segments <= 5):
            raise ValueError("segment range must remain within 1..5")
        if not (10 <= self.min_hosts <= self.max_hosts <= 60):
            raise ValueError("host range must remain within 10..60")
        if self.max_nodes < 7 + self.max_segments + self.max_hosts + self.max_decoy_services:
            raise ValueError("max_nodes cannot fit the worst-case generated profile")
        allowed = {
            "network_types": {NodeType.NETWORK_SEGMENT, NodeType.CLOUD_NETWORK},
            "host_types": {NodeType.HOST, NodeType.LEGACY_HOST, NodeType.CLOUD_WORKLOAD},
            "identity_types": {NodeType.IDENTITY, NodeType.IAM_ROLE},
            "store_types": {NodeType.DATABASE, NodeType.STORAGE, NodeType.CLOUD_RESOURCE},
        }
        for definition in self.profiles.values():
            for field, choices in allowed.items():
                values = set(getattr(definition, field))
                if not values or not values <= choices:
                    raise ValueError(f"invalid {field}: {sorted(map(str, values))}")

    def digest(self) -> str:
        payload = {
            "bounds": {
                name: getattr(self, name)
                for name in (
                    "min_segments", "max_segments", "min_hosts", "max_hosts",
                    "min_pivots", "max_pivots", "min_decoy_services",
                    "max_decoy_services", "max_nodes", "max_vulnerabilities", "max_steps",
                )
            },
            "profiles": {
                name.value: {
                    "network_types": [item.value for item in definition.network_types],
                    "host_types": [item.value for item in definition.host_types],
                    "identity_types": [item.value for item in definition.identity_types],
                    "store_types": [item.value for item in definition.store_types],
                    "control": definition.control,
                    "boundary": definition.boundary,
                }
                for name, definition in sorted(self.profiles.items(), key=lambda item: item[0])
            },
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()


def _cycle_with_required(
    rng: random.Random, choices: tuple[NodeType, ...], count: int
) -> list[NodeType]:
    values = list(choices[:count])
    values.extend(rng.choice(choices) for _ in range(count - len(values)))
    rng.shuffle(values)
    return values


def generate_profile_topology(
    profile: DeploymentProfile | str,
    seed: int,
    config: EnterpriseProfileConfig | None = None,
) -> TrueTopology:
    """Generate one profile using the shared typed graph and action semantics."""
    selected = DeploymentProfile(profile)
    if selected == DeploymentProfile.ON_PREMISES:
        return generate_onprem_topology(seed)
    config = config or EnterpriseProfileConfig.from_yaml()
    definition = config.profiles[selected]
    rng = random.Random(int(seed))
    segment_count = rng.randint(config.min_segments, config.max_segments)
    if selected == DeploymentProfile.HYBRID:
        # Every hybrid sample must cross a real modelled boundary.  Seeded
        # variation may change its size, but cannot silently collapse it into
        # a single-environment graph.
        segment_count = max(segment_count, len(definition.network_types))
    host_count = rng.randint(config.min_hosts, config.max_hosts)
    pivot_count = rng.randint(config.min_pivots, config.max_pivots)
    decoy_services = rng.randint(config.min_decoy_services, config.max_decoy_services)
    nodes: dict[str, EnterpriseNode] = {}
    edges: list[EnterpriseEdge] = []

    def node(node_id: str, kind: NodeType, name: str, **attributes) -> None:
        nodes[node_id] = EnterpriseNode(node_id, kind, name, attributes)

    def edge(source: str, target: str, kind: EdgeType, **attributes) -> None:
        edges.append(EnterpriseEdge(source, target, kind, attributes))

    node("entry", NodeType.ENTRY_POINT, "External entry")
    network_types = _cycle_with_required(rng, definition.network_types, segment_count)
    segments = [f"network_{index + 1}" for index in range(segment_count)]
    for segment, kind in zip(segments, network_types, strict=True):
        node(segment, kind, f"{selected.value} network {segment}", profile=selected.value)
    node("control_edge", NodeType.SECURITY_CONTROL, "Boundary control", control=definition.control)
    edge("entry", segments[0], EdgeType.CONNECTS)
    edge("control_edge", segments[0], EdgeType.PROTECTS)
    for source, target in zip(segments, segments[1:], strict=False):
        edge(source, target, EdgeType.CONNECTS, trust_boundary=definition.boundary)

    products = (
        ("synthetic_tls", "1.0", 443, "SYN-PROFILE-TLS"),
        ("synthetic_gateway", "2.0", 8443, "SYN-PROFILE-GATEWAY"),
        ("synthetic_remote", "3.0", 22, "SYN-PROFILE-REMOTE"),
    )
    product, version, port, vulnerability_id = rng.choice(products)
    all_host_types = _cycle_with_required(rng, definition.host_types, host_count)
    path_host_types = all_host_types[: pivot_count + 2]
    node("host_entry", path_host_types[0], "Entry workload", profile=selected.value)
    node(
        "service_entry", NodeType.SERVICE, "External service",
        product=product, version=version, port=port,
    )
    edge("host_entry", segments[0], EdgeType.LOCATED_IN)
    edge("host_entry", "service_entry", EdgeType.HOSTS)
    node("application_entry", NodeType.APPLICATION, "Business application")
    edge("service_entry", "application_entry", EdgeType.EXPOSES)

    identity_types = _cycle_with_required(rng, definition.identity_types, 2)
    identity_entry_type = identity_types[0]
    node("identity_entry", identity_entry_type, "Workload identity")
    edge("host_entry", "identity_entry", EdgeType.YIELDS_CREDENTIAL)
    path_hosts = [f"host_pivot_{index + 1}" for index in range(pivot_count)] + ["host_data"]
    for index, host_id in enumerate(path_hosts):
        node(host_id, path_host_types[index + 1], f"Path workload {index + 1}")
        edge(host_id, segments[index % len(segments)], EdgeType.LOCATED_IN)
    edge("identity_entry", path_hosts[0], EdgeType.HAS_ACCESS)
    for source, target in zip(path_hosts, path_hosts[1:], strict=False):
        edge(source, target, EdgeType.PIVOTS_TO, trust_boundary=definition.boundary)

    identity_data_type = identity_types[1]
    store_type = rng.choice(definition.store_types)
    node("identity_data", identity_data_type, "Data identity")
    node("store_crown", store_type, "Restricted enterprise store")
    node("asset_crown", NodeType.ASSET, "Restricted business data", value=100)
    edge("host_data", "identity_data", EdgeType.YIELDS_CREDENTIAL)
    edge("identity_data", "store_crown", EdgeType.HAS_ACCESS)
    edge("host_data", "store_crown", EdgeType.HOSTS)
    edge("store_crown", "asset_crown", EdgeType.CONTAINS)

    decoy_count = host_count - len(path_host_types)
    for index in range(decoy_count):
        host_id = f"host_decoy_{index + 1}"
        node(
            host_id,
            all_host_types[len(path_host_types) + index],
            f"Decoy workload {index + 1}",
        )
        edge(host_id, rng.choice(segments), EdgeType.LOCATED_IN)
    for index in range(decoy_services):
        service_id = f"service_decoy_{index + 1}"
        node(service_id, NodeType.SERVICE, f"Decoy service {index + 1}", port=8000 + index)
        edge(rng.choice(["host_entry", *path_hosts]), service_id, EdgeType.HOSTS)

    vulnerability = Vulnerability(
        id=vulnerability_id,
        target="service_entry",
        cvss=round(rng.uniform(7.0, 9.8), 1),
        exploit_probability=1.0,
        grants_access_to="host_entry",
        affected_product=product,
        affected_versions=(version,),
        description=f"Synthetic {selected.value} profile weakness",
    )
    return TrueTopology(
        name=f"enterprise-{selected.value}-v1-seed-{seed}",
        nodes=nodes,
        edges=edges,
        vulnerabilities={vulnerability.id: vulnerability},
        entry_points=("entry",),
        crown_jewels=("asset_crown",),
    )


class InfrastructureCurriculumEnv(gym.Env):
    """Hide both topology seed and deployment profile behind fixed spaces."""

    metadata = {"render_modes": ["ansi"]}

    def __init__(
        self,
        topology_seeds: tuple[int, ...] | list[int],
        profiles: tuple[DeploymentProfile, ...] | list[DeploymentProfile],
        *,
        config: EnterpriseProfileConfig | None = None,
    ) -> None:
        super().__init__()
        if not topology_seeds or not profiles:
            raise ValueError("topology seeds and profiles are required")
        self.topology_seeds = tuple(map(int, topology_seeds))
        self.profiles = tuple(DeploymentProfile(item) for item in profiles)
        if DeploymentProfile.ON_PREMISES in self.profiles:
            raise ValueError("the frozen on-prem control uses OnPremCurriculumEnv")
        self.config = config or EnterpriseProfileConfig.from_yaml()
        self.topology_seed = self.topology_seeds[0]
        self.profile = self.profiles[0]
        self._env = self._make(self.profile, self.topology_seed)
        self.action_space = self._env.action_space
        self.observation_space = self._env.observation_space

    def _make(self, profile: DeploymentProfile, seed: int) -> EnterpriseCyberEnv:
        return EnterpriseCyberEnv(
            generate_profile_topology(profile, seed, self.config),
            max_steps=self.config.max_steps,
            max_nodes=self.config.max_nodes,
            max_vulnerabilities=self.config.max_vulnerabilities,
        )

    @property
    def true_topology(self) -> TrueTopology:
        return self._env.true_topology

    @property
    def knowledge(self):
        return self._env.knowledge

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        options = options or {}
        raw_seed = options.get("topology_seed")
        raw_profile = options.get("profile")
        requested_seed = (
            int(raw_seed)
            if raw_seed is not None
            else int(self.np_random.choice(self.topology_seeds))
        )
        requested_profile = (
            DeploymentProfile(raw_profile)
            if raw_profile is not None
            else DeploymentProfile(self.np_random.choice(self.profiles))
        )
        if requested_seed not in self.topology_seeds or requested_profile not in self.profiles:
            raise ValueError("requested seed/profile is outside this curriculum")
        self.topology_seed, self.profile = requested_seed, requested_profile
        self._env = self._make(self.profile, self.topology_seed)
        if self._env.action_space != self.action_space:
            raise RuntimeError("action space changed across infrastructure profiles")
        if self._env.observation_space != self.observation_space:
            raise RuntimeError("observation space changed across infrastructure profiles")
        observation, info = self._env.reset(seed=seed)
        info.update(
            {
                "profile": self.profile.value,
                "topology_seed": self.topology_seed,
                "topology_hash": topology_digest(self.true_topology),
                "profile_config_hash": self.config.digest(),
            }
        )
        return observation, info

    def step(self, action: int):
        observation, reward, terminated, truncated, info = self._env.step(action)
        info.update(
            {
                "profile": self.profile.value,
                "topology_seed": self.topology_seed,
                "topology_hash": topology_digest(self.true_topology),
            }
        )
        return observation, reward, terminated, truncated, info

    def action_masks(self):
        return self._env.action_masks()

    def action_index(self, *args, **kwargs) -> int:
        return self._env.action_index(*args, **kwargs)

    def attack_path(self):
        return self._env.attack_path()
