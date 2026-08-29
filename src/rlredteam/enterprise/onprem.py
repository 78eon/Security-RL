"""Seeded on-prem topology distribution and fixed-space curriculum environment."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import gymnasium as gym
import yaml

from rlredteam.enterprise.environment import EnterpriseActionType, EnterpriseCyberEnv
from rlredteam.enterprise.model import (
    EdgeType,
    EnterpriseEdge,
    EnterpriseNode,
    NodeType,
    TrueTopology,
    Vulnerability,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "onprem_topology.yaml"

ON_PREM_TYPES = frozenset(
    {
        NodeType.ENTRY_POINT,
        NodeType.NETWORK_SEGMENT,
        NodeType.NETWORK_DEVICE,
        NodeType.SECURITY_CONTROL,
        NodeType.HOST,
        NodeType.SERVICE,
        NodeType.APPLICATION,
        NodeType.API,
        NodeType.IDENTITY,
        NodeType.DATABASE,
        NodeType.IDENTITY_PROVIDER,
        NodeType.ASSET,
    }
)


@dataclass(frozen=True, slots=True)
class OnPremTopologyConfig:
    min_segments: int = 1
    max_segments: int = 5
    min_hosts: int = 10
    max_hosts: int = 60
    min_pivots: int = 0
    max_pivots: int = 3
    min_decoy_services: int = 1
    max_decoy_services: int = 10
    min_application_hops: int = 1
    max_application_hops: int = 3
    max_nodes: int = 96
    max_vulnerabilities: int = 8
    max_steps: int = 200

    def __post_init__(self) -> None:
        for name in (
            "segments",
            "hosts",
            "pivots",
            "decoy_services",
            "application_hops",
        ):
            minimum = getattr(self, f"min_{name}")
            maximum = getattr(self, f"max_{name}")
            if minimum < 0 or maximum < minimum:
                raise ValueError(f"invalid {name} range: {minimum}..{maximum}")
        if self.min_segments < 1:
            raise ValueError("at least one network segment is required")
        if self.min_hosts < self.max_pivots + 2:
            raise ValueError("minimum hosts cannot fit the maximum attack path")
        if self.min_application_hops < 1:
            raise ValueError("at least one application hop is required")
        if self.max_nodes < self.maximum_generated_nodes:
            raise ValueError(
                f"max_nodes {self.max_nodes} is below worst-case topology size "
                f"{self.maximum_generated_nodes}"
            )
        if self.max_vulnerabilities < 1 or self.max_steps <= 0:
            raise ValueError("vulnerability and step capacities must be positive")

    @property
    def maximum_generated_nodes(self) -> int:
        non_hosts = 7  # entry, control, entry service, two identities, DB and asset
        return (
            non_hosts
            + self.max_segments
            + self.max_hosts
            + self.max_application_hops
            + self.max_decoy_services
        )

    @classmethod
    def from_yaml(cls, path: Path | None = None) -> OnPremTopologyConfig:
        raw = yaml.safe_load((path or DEFAULT_CONFIG).read_text())["onprem_topology"]
        return cls(**raw)

    def digest(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class OnPremGeneralisationSplit:
    """Preregistered disjoint topology seed sets."""

    train: tuple[int, ...] = tuple(range(1, 61))
    validation: tuple[int, ...] = tuple(range(1001, 1021))
    test: tuple[int, ...] = tuple(range(2001, 2021))

    def __post_init__(self) -> None:
        groups = tuple(map(set, (self.train, self.validation, self.test)))
        if not all(groups):
            raise ValueError("topology seed sets must be non-empty")
        if groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2]:
            raise ValueError("topology seed sets must be disjoint")


def topology_digest(topology: TrueTopology) -> str:
    canonical = json.dumps(topology.to_dict(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def generate_onprem_topology(
    seed: int,
    config: OnPremTopologyConfig | None = None,
) -> TrueTopology:
    """Generate a bounded, structurally variable topology with a feasible path."""
    config = config or OnPremTopologyConfig.from_yaml()
    rng = random.Random(int(seed))
    segment_count = rng.randint(config.min_segments, config.max_segments)
    host_count = rng.randint(config.min_hosts, config.max_hosts)
    pivot_count = rng.randint(config.min_pivots, config.max_pivots)
    decoy_host_count = host_count - pivot_count - 2  # entry + data + pivots
    decoy_service_count = rng.randint(
        config.min_decoy_services, config.max_decoy_services
    )
    application_hops = rng.randint(
        config.min_application_hops, config.max_application_hops
    )
    nodes: dict[str, EnterpriseNode] = {}
    edges: list[EnterpriseEdge] = []

    def node(node_id: str, kind: NodeType, name: str, **attributes) -> None:
        nodes[node_id] = EnterpriseNode(node_id, kind, name, attributes)

    def edge(source: str, target: str, kind: EdgeType, **attributes) -> None:
        edges.append(EnterpriseEdge(source, target, kind, attributes))

    node("entry", NodeType.ENTRY_POINT, "External entry")
    segments = ["segment_edge"] + [f"segment_internal_{i}" for i in range(1, segment_count)]
    for index, segment in enumerate(segments):
        node(segment, NodeType.NETWORK_SEGMENT, f"On-prem segment {index + 1}")
    node("control_edge", NodeType.SECURITY_CONTROL, "Edge firewall", control="firewall")
    edge("entry", segments[0], EdgeType.CONNECTS)
    edge("control_edge", segments[0], EdgeType.PROTECTS)
    for left, right in zip(segments, segments[1:], strict=False):
        edge(left, right, EdgeType.CONNECTS, filtered=True)

    service_profiles = (
        ("synthetic_https", "1.0", 443, "SYN-ENTRY-HTTPS"),
        ("synthetic_ssh", "2.1", 22, "SYN-ENTRY-SSH"),
        ("synthetic_gateway", "3.0", 8443, "SYN-ENTRY-GATEWAY"),
    )
    product, version, port, vulnerability_id = rng.choice(service_profiles)
    node("host_entry", NodeType.HOST, "Public service host", os="linux")
    node(
        "service_entry",
        NodeType.SERVICE,
        "Externally reachable service",
        product=product,
        version=version,
        port=port,
        protocol="tcp",
    )
    edge("host_entry", segments[0], EdgeType.LOCATED_IN)
    edge("host_entry", "service_entry", EdgeType.HOSTS)

    previous = "service_entry"
    for index in range(application_hops):
        application_id = f"application_{index + 1}"
        kind = NodeType.APPLICATION if index == 0 else NodeType.API
        node(application_id, kind, f"Application component {index + 1}")
        edge(
            previous,
            application_id,
            EdgeType.EXPOSES if index == 0 else EdgeType.CALLS,
        )
        previous = application_id

    node("identity_entry", NodeType.IDENTITY, "Application service identity")
    edge("host_entry", "identity_entry", EdgeType.YIELDS_CREDENTIAL)

    internal_hosts = [f"host_pivot_{i}" for i in range(1, pivot_count + 1)] + [
        "host_data"
    ]
    for index, host_id in enumerate(internal_hosts):
        segment = (
            segments[0]
            if len(segments) == 1
            else segments[1 + (index % (len(segments) - 1))]
        )
        node(host_id, NodeType.HOST, f"Internal host {index + 1}", os="linux")
        edge(host_id, segment, EdgeType.LOCATED_IN)
    edge("identity_entry", internal_hosts[0], EdgeType.HAS_ACCESS, privilege="user")
    for source, target in zip(internal_hosts, internal_hosts[1:], strict=False):
        edge(source, target, EdgeType.PIVOTS_TO)

    node("identity_data", NodeType.IDENTITY, "Database service identity")
    node("database_crown", NodeType.DATABASE, "Restricted records database")
    node(
        "asset_crown",
        NodeType.ASSET,
        "Restricted business records",
        classification="restricted",
        value=100,
    )
    edge("host_data", "identity_data", EdgeType.YIELDS_CREDENTIAL)
    edge("identity_data", "database_crown", EdgeType.HAS_ACCESS, privilege="read")
    edge("host_data", "database_crown", EdgeType.HOSTS)
    edge("database_crown", "asset_crown", EdgeType.CONTAINS)

    for index in range(decoy_host_count):
        host_id = f"host_decoy_{index + 1}"
        segment = rng.choice(segments)
        node(host_id, NodeType.HOST, f"Decoy host {index + 1}", os=rng.choice(("linux", "windows")))
        edge(host_id, segment, EdgeType.LOCATED_IN)
    service_names = (("ssh", 22), ("smb", 445), ("dns", 53), ("rdp", 3389))
    all_hosts = ["host_entry", *internal_hosts] + [
        f"host_decoy_{index + 1}" for index in range(decoy_host_count)
    ]
    for index in range(decoy_service_count):
        service, service_port = rng.choice(service_names)
        service_id = f"service_decoy_{index + 1}_{service}"
        node(service_id, NodeType.SERVICE, f"Decoy {service.upper()}", port=service_port)
        edge(rng.choice(all_hosts), service_id, EdgeType.HOSTS)

    vulnerability = Vulnerability(
        id=vulnerability_id,
        target="service_entry",
        cvss=round(rng.uniform(7.0, 9.8), 1),
        exploit_probability=1.0,
        grants_access_to="host_entry",
        affected_product=product,
        affected_versions=(version,),
        description="Synthetic entry weakness for the on-prem simulation",
    )
    topology = TrueTopology(
        name=f"onprem-v1-seed-{seed}",
        nodes=nodes,
        edges=edges,
        vulnerabilities={vulnerability.id: vulnerability},
        entry_points=("entry",),
        crown_jewels=("asset_crown",),
    )
    if not {node.type for node in topology.nodes.values()} <= ON_PREM_TYPES:
        raise RuntimeError("on-prem generator emitted an out-of-scope node type")
    return topology


class OnPremCurriculumEnv(gym.Env):
    """Choose a hidden on-prem topology from an allowed seed pool per episode."""

    metadata = {"render_modes": ["ansi"]}

    def __init__(
        self,
        topology_seeds: tuple[int, ...] | list[int],
        *,
        config: OnPremTopologyConfig | None = None,
    ) -> None:
        super().__init__()
        if not topology_seeds:
            raise ValueError("at least one topology seed is required")
        self.topology_seeds = tuple(int(seed) for seed in topology_seeds)
        self.config = config or OnPremTopologyConfig.from_yaml()
        self.topology_seed = self.topology_seeds[0]
        self._env = self._make(self.topology_seed)
        self.action_space = self._env.action_space
        self.observation_space = self._env.observation_space

    def _make(self, seed: int) -> EnterpriseCyberEnv:
        return EnterpriseCyberEnv(
            generate_onprem_topology(seed, self.config),
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
        requested = (options or {}).get("topology_seed")
        if requested is not None:
            if int(requested) not in self.topology_seeds:
                raise ValueError("requested topology seed is outside this split")
            self.topology_seed = int(requested)
        else:
            self.topology_seed = int(self.np_random.choice(self.topology_seeds))
        self._env = self._make(self.topology_seed)
        if self._env.action_space != self.action_space:
            raise RuntimeError("action space changed across topology seeds")
        if self._env.observation_space != self.observation_space:
            raise RuntimeError("observation space changed across topology seeds")
        observation, info = self._env.reset(seed=seed)
        info.update(
            {
                "topology_seed": self.topology_seed,
                "topology_hash": topology_digest(self.true_topology),
                "topology_config_hash": self.config.digest(),
            }
        )
        return observation, info

    def step(self, action: int):
        observation, reward, terminated, truncated, info = self._env.step(action)
        info.update(
            {
                "topology_seed": self.topology_seed,
                "topology_hash": topology_digest(self.true_topology),
            }
        )
        return observation, reward, terminated, truncated, info

    def action_masks(self):
        """Return the current knowledge-only mask for MaskablePPO."""
        return self._env.action_masks()

    def action_index(self, *args, **kwargs) -> int:
        return self._env.action_index(*args, **kwargs)

    def attack_path(self):
        return self._env.attack_path()


def knowledge_policy_action(env: EnterpriseCyberEnv) -> int:
    """Select one valid action using only the public action mask and catalogue.

    This deterministic feasibility oracle validates generated environments; it
    is not used for PPO training or evaluation evidence.
    """
    progress_priority = (
        EnterpriseActionType.ACCESS_ASSET,
        EnterpriseActionType.EXPLOIT,
        EnterpriseActionType.OBTAIN_CREDENTIAL,
        EnterpriseActionType.AUTHENTICATE,
        EnterpriseActionType.PIVOT,
        EnterpriseActionType.DISCOVER_NETWORK,
        EnterpriseActionType.ESCALATE_PRIVILEGE,
        EnterpriseActionType.ENUMERATE_SERVICE,
    )
    mask = env.valid_action_mask()
    for action_type in progress_priority:
        for index, catalogue_action in enumerate(env.actions):
            if catalogue_action.type == action_type and mask[index]:
                return index
    for index, catalogue_action in enumerate(env.actions):
        if catalogue_action.type != EnterpriseActionType.ASSESS_VULNERABILITY:
            continue
        slot = int(catalogue_action.target.removeprefix(env._NODE_SLOT_PREFIX))
        if (
            mask[index]
            and slot < len(env.knowledge.discovery_order)
            and env.knowledge.known_node_types[env.knowledge.discovery_order[slot]]
            == NodeType.SERVICE
        ):
            return index
    discovery_priority = (
        EnterpriseActionType.ENUMERATE_HOST,
        EnterpriseActionType.ENUMERATE_APPLICATION,
    )
    for action_type in discovery_priority:
        for index, catalogue_action in enumerate(env.actions):
            if catalogue_action.type == action_type and mask[index]:
                return index
    for index, catalogue_action in enumerate(env.actions):
        if (
            catalogue_action.type == EnterpriseActionType.ASSESS_VULNERABILITY
            and mask[index]
        ):
            return index
    raise RuntimeError("no valid knowledge-derived action remains")
