"""Gymnasium environment for partial-observability attack-path discovery.

All discovery and attack effects are state transitions over an in-memory graph.
No sockets, subprocesses, scanners or exploit tools are used.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import gymnasium as gym
import numpy as np

from rlredteam.enterprise.model import EdgeType, EnterpriseGraph, NodeType, TrueTopology
from rlredteam.enterprise.state import AgentKnowledge, Observation


class EnterpriseActionType(StrEnum):
    DISCOVER_NETWORK = "discover_network"
    ENUMERATE_HOST = "enumerate_host"
    ENUMERATE_SERVICE = "enumerate_service"
    ENUMERATE_APPLICATION = "enumerate_application"
    ASSESS_VULNERABILITY = "assess_vulnerability"
    EXPLOIT = "exploit"
    OBTAIN_CREDENTIAL = "obtain_credential"
    AUTHENTICATE = "authenticate"
    PIVOT = "pivot"
    ESCALATE_PRIVILEGE = "escalate_privilege"
    ACCESS_ASSET = "access_asset"


@dataclass(frozen=True, slots=True)
class EnterpriseAction:
    type: EnterpriseActionType
    target: str

    @property
    def name(self) -> str:
        return f"{self.type.value}:{self.target}"


@dataclass(frozen=True, slots=True)
class EnterpriseEvent:
    step: int
    action: EnterpriseAction
    success: bool
    state_changed: bool
    reward: float
    prerequisites: tuple[str, ...] = ()
    outcomes: tuple[str, ...] = ()
    reason: str | None = None
    goal_reached: bool = False


class EnterpriseCyberEnv(gym.Env):
    """A fixed-graph, discrete-action environment suitable for PPO.

    ``TrueTopology`` resolves simulated outcomes, while observations are built
    only from ``AgentKnowledge``.  Node and vulnerability actions address
    discovery-order slots, so the action catalogue cannot reveal hidden entity
    identifiers. ``info["action_mask"]`` identifies currently valid actions.
    """

    metadata = {"render_modes": ["ansi"]}

    _TYPE_ORDER = tuple(NodeType)
    _FEATURES_PER_NODE = Observation.feature_count(_TYPE_ORDER)
    _NODE_SLOT_PREFIX = "node_slot_"
    _VULNERABILITY_SLOT_PREFIX = "vulnerability_slot_"

    def __init__(
        self,
        graph: EnterpriseGraph | TrueTopology,
        *,
        max_steps: int = 200,
        max_nodes: int = 32,
        max_vulnerabilities: int = 16,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()
        if max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if render_mode not in {None, "ansi"}:
            raise ValueError("render_mode must be None or 'ansi'")
        self.true_topology = graph
        self.max_steps = max_steps
        self.max_nodes = max_nodes
        self.max_vulnerabilities = max_vulnerabilities
        self.render_mode = render_mode
        if len(graph.nodes) > max_nodes:
            raise ValueError(f"graph has {len(graph.nodes)} nodes; capacity is {max_nodes}")
        if len(graph.vulnerabilities) > max_vulnerabilities:
            raise ValueError(
                f"graph has {len(graph.vulnerabilities)} vulnerabilities; "
                f"capacity is {max_vulnerabilities}"
            )
        self.actions = self._build_actions()
        self.action_space = gym.spaces.Discrete(len(self.actions))
        observation_size = (
            max_nodes * self._FEATURES_PER_NODE + max_nodes * max_nodes + 1
        )
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(observation_size,), dtype=np.float32
        )
        self.knowledge = AgentKnowledge()
        self.events: list[EnterpriseEvent] = []
        self._step = 0

    @property
    def graph(self) -> TrueTopology:
        """Ground truth for analyst/debug tooling; never a policy input."""
        return self.true_topology

    def _build_actions(self) -> tuple[EnterpriseAction, ...]:
        actions: list[EnterpriseAction] = []
        non_exploit = [
            kind for kind in EnterpriseActionType if kind != EnterpriseActionType.EXPLOIT
        ]
        for action_type in non_exploit:
            for index in range(self.max_nodes):
                actions.append(
                    EnterpriseAction(action_type, f"{self._NODE_SLOT_PREFIX}{index}")
                )
        for index in range(self.max_vulnerabilities):
            actions.append(
                EnterpriseAction(
                    EnterpriseActionType.EXPLOIT,
                    f"{self._VULNERABILITY_SLOT_PREFIX}{index}",
                )
            )
        return tuple(actions)

    def action_index(self, action_type: EnterpriseActionType, target: str) -> int:
        if action_type == EnterpriseActionType.EXPLOIT:
            slot = self.knowledge.vulnerability_slot(target)
            slot_target = f"{self._VULNERABILITY_SLOT_PREFIX}{slot}"
        else:
            slot = self.knowledge.node_slot(target)
            slot_target = f"{self._NODE_SLOT_PREFIX}{slot}"
        needle = EnterpriseAction(action_type, slot_target)
        try:
            return self.actions.index(needle)
        except ValueError:
            message = f"action not in this environment: {action_type.value}:{target}"
            raise KeyError(message) from None

    def _resolve_action(self, action: EnterpriseAction) -> EnterpriseAction:
        prefix = (
            self._VULNERABILITY_SLOT_PREFIX
            if action.type == EnterpriseActionType.EXPLOIT
            else self._NODE_SLOT_PREFIX
        )
        try:
            slot = int(action.target.removeprefix(prefix))
        except ValueError:
            return action
        known_targets = (
            self.knowledge.vulnerability_order
            if action.type == EnterpriseActionType.EXPLOIT
            else self.knowledge.discovery_order
        )
        if slot >= len(known_targets):
            return EnterpriseAction(action.type, f"__unknown_slot_{slot}")
        return EnterpriseAction(action.type, known_targets[slot])

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._step = 0
        self.events = []
        self.knowledge = AgentKnowledge()
        for entry in self.true_topology.entry_points:
            self._reveal_node(entry)
        for entry in self.true_topology.entry_points:
            for edge in self.true_topology.outgoing(entry, EdgeType.CONNECTS):
                self._reveal_node(edge.target, reachable=True)
                self._reveal_edge(edge.source, edge.target, edge.type)
        return self._observation(), self._info()

    def _reveal_node(self, node_id: str, *, reachable: bool = False) -> None:
        node = self.true_topology.nodes[node_id]
        self.knowledge.discover(node_id, node.type, reachable=reachable)

    def _reveal_edge(self, source: str, target: str, edge_type: EdgeType) -> None:
        self.knowledge.learn_edge(source, target, edge_type)

    def _known_edges(self, edge_type: EdgeType):
        """Return agent-known endpoint pairs for one relationship type."""
        return (
            (source, target)
            for source, target, known_type in self.knowledge.known_edges
            if known_type == edge_type
        )

    def _observation(self) -> np.ndarray:
        return Observation.from_knowledge(
            self.knowledge,
            max_nodes=self.max_nodes,
            step=self._step,
            max_steps=self.max_steps,
            type_order=self._TYPE_ORDER,
        ).as_array()

    def _info(self) -> dict[str, Any]:
        return {
            "action_mask": self.valid_action_mask(),
            "known_nodes": len(self.knowledge.discovered),
            "known_vulnerabilities": len(self.knowledge.known_vulnerabilities),
            "compromised_nodes": sorted(self.knowledge.access),
            "credentials": sorted(self.knowledge.credentials),
            "goal_reached": bool(
                self.knowledge.accessed_assets & set(self.true_topology.crown_jewels)
            ),
        }

    def valid_action_mask(self) -> np.ndarray:
        return np.asarray(
            [self._can_execute(self._resolve_action(action))[0] for action in self.actions],
            dtype=np.int8,
        )

    def _can_execute(self, action: EnterpriseAction) -> tuple[bool, str]:
        known = self.knowledge
        target = action.target
        kind = action.type
        if kind == EnterpriseActionType.EXPLOIT:
            if target not in known.known_vulnerabilities:
                return False, "vulnerability has not been discovered"
            if known.vulnerability_grants_access[target] in known.access:
                return False, "target is already compromised"
            return True, ""
        if target not in known.discovered:
            return False, "target has not been discovered"
        target_type = known.known_node_types[target]
        allowed_types = {
            EnterpriseActionType.DISCOVER_NETWORK: {
                NodeType.NETWORK_SEGMENT,
                NodeType.CLOUD_NETWORK,
            },
            EnterpriseActionType.ENUMERATE_HOST: {
                NodeType.HOST,
                NodeType.LEGACY_HOST,
                NodeType.CLOUD_WORKLOAD,
            },
            EnterpriseActionType.ENUMERATE_SERVICE: {NodeType.SERVICE},
            EnterpriseActionType.ENUMERATE_APPLICATION: {
                NodeType.APPLICATION,
                NodeType.API,
                NodeType.DATABASE,
                NodeType.CLOUD_RESOURCE,
                NodeType.CLOUD_ACCOUNT,
                NodeType.STORAGE,
            },
            EnterpriseActionType.ASSESS_VULNERABILITY: set(NodeType),
            EnterpriseActionType.OBTAIN_CREDENTIAL: {
                NodeType.IDENTITY,
                NodeType.IAM_ROLE,
            },
            EnterpriseActionType.AUTHENTICATE: {
                NodeType.HOST,
                NodeType.LEGACY_HOST,
                NodeType.CLOUD_WORKLOAD,
                NodeType.DATABASE,
                NodeType.CLOUD_RESOURCE,
                NodeType.STORAGE,
            },
            EnterpriseActionType.PIVOT: {
                NodeType.HOST,
                NodeType.LEGACY_HOST,
                NodeType.CLOUD_WORKLOAD,
            },
            EnterpriseActionType.ESCALATE_PRIVILEGE: {
                NodeType.HOST,
                NodeType.LEGACY_HOST,
                NodeType.CLOUD_WORKLOAD,
            },
            EnterpriseActionType.ACCESS_ASSET: {NodeType.ASSET},
        }
        if target_type not in allowed_types[kind]:
            return False, f"{kind.value} does not apply to {target_type.value}"
        if kind == EnterpriseActionType.DISCOVER_NETWORK:
            return target not in known.enumerated, "network already discovered"
        if kind in {
            EnterpriseActionType.ENUMERATE_HOST,
            EnterpriseActionType.ENUMERATE_SERVICE,
            EnterpriseActionType.ENUMERATE_APPLICATION,
        }:
            return target not in known.enumerated, "target already enumerated"
        if kind == EnterpriseActionType.ASSESS_VULNERABILITY:
            return target not in known.assessed, "target already assessed"
        if kind == EnterpriseActionType.OBTAIN_CREDENTIAL:
            if target in known.credentials:
                return False, "credential already obtained"
            possible = any(
                edge_target == target and source in known.access
                for source, edge_target in self._known_edges(
                    EdgeType.YIELDS_CREDENTIAL
                )
            )
            return possible, "no compromised source yields this credential"
        if kind == EnterpriseActionType.AUTHENTICATE:
            if target in known.access:
                return False, "target is already accessed"
            possible = any(
                edge_target == target and source in known.credentials
                for source, edge_target in self._known_edges(EdgeType.HAS_ACCESS)
            )
            return possible, "no owned credential grants access"
        if kind == EnterpriseActionType.PIVOT:
            if target in known.access:
                return False, "target is already compromised"
            possible = any(
                edge_target == target and source in known.access
                for source, edge_target in self._known_edges(EdgeType.PIVOTS_TO)
            )
            return possible, "no compromised source can pivot to target"
        if kind == EnterpriseActionType.ESCALATE_PRIVILEGE:
            if known.access_level(target) != 1:
                return False, "user access is required"
            possible = any(
                known.vulnerability_targets[vulnerability_id] == target
                and known.vulnerability_privileges[vulnerability_id] == "root"
                for vulnerability_id in known.known_vulnerabilities
            )
            return possible, "no known privilege-escalation vulnerability"
        if kind == EnterpriseActionType.ACCESS_ASSET:
            if target in known.accessed_assets:
                return False, "asset already accessed"
            possible = any(
                edge_target == target and source in known.access
                for source, edge_target in self._known_edges(EdgeType.CONTAINS)
            )
            return possible, "containing system has not been accessed"
        return False, "unsupported action"

    def step(self, action: int):
        if not self.action_space.contains(action):
            raise ValueError(f"action {action!r} outside [0, {self.action_space.n})")
        selected = self._resolve_action(self.actions[int(action)])
        valid, reason = self._can_execute(selected)
        self._step += 1
        if valid:
            success, changed, reward, prerequisites, outcomes, reason = self._execute(selected)
        else:
            success, changed, reward = False, False, -5.0
            prerequisites, outcomes = (), ()

        goal = bool(
            self.knowledge.accessed_assets & set(self.true_topology.crown_jewels)
        )
        event = EnterpriseEvent(
            step=self._step - 1,
            action=selected,
            success=success,
            state_changed=changed,
            reward=reward,
            prerequisites=prerequisites,
            outcomes=outcomes,
            reason=reason or None,
            goal_reached=goal,
        )
        self.events.append(event)
        terminated = goal
        truncated = self._step >= self.max_steps and not terminated
        info = self._info()
        info["event"] = event
        return self._observation(), reward, terminated, truncated, info

    def _execute(
        self, action: EnterpriseAction
    ) -> tuple[bool, bool, float, tuple[str, ...], tuple[str, ...], str]:
        before = self._snapshot()
        known = self.knowledge
        target = action.target
        kind = action.type
        prerequisites: list[str] = []
        outcomes: list[str] = []

        if kind == EnterpriseActionType.DISCOVER_NETWORK:
            known.enumerated.add(target)
            for edge in self.true_topology.edges:
                if edge.type == EdgeType.LOCATED_IN and edge.target == target:
                    self._reveal_node(edge.source, reachable=True)
                    self._reveal_edge(edge.source, edge.target, edge.type)
                    outcomes.append(f"discovered:{edge.source}")
                elif edge.type == EdgeType.PROTECTS and edge.target == target:
                    self._reveal_node(edge.source)
                    self._reveal_edge(edge.source, edge.target, edge.type)
                    outcomes.append(f"control:{edge.source}")
            prerequisites.append(f"reachable:{target}")
        elif kind == EnterpriseActionType.ENUMERATE_HOST:
            known.enumerated.add(target)
            for edge in self.true_topology.outgoing(target):
                if edge.type in {EdgeType.HOSTS, EdgeType.YIELDS_CREDENTIAL, EdgeType.PIVOTS_TO}:
                    self._reveal_node(
                        edge.target, reachable=edge.type != EdgeType.YIELDS_CREDENTIAL
                    )
                    self._reveal_edge(edge.source, edge.target, edge.type)
                    outcomes.append(f"discovered:{edge.target}")
            prerequisites.append(f"known_host:{target}")
        elif kind == EnterpriseActionType.ENUMERATE_SERVICE:
            known.enumerated.add(target)
            for child in self.true_topology.children(target, EdgeType.EXPOSES):
                self._reveal_node(child, reachable=True)
                self._reveal_edge(target, child, EdgeType.EXPOSES)
                outcomes.append(f"discovered:{child}")
            prerequisites.append(f"known_service:{target}")
        elif kind == EnterpriseActionType.ENUMERATE_APPLICATION:
            known.enumerated.add(target)
            for edge in self.true_topology.outgoing(target):
                if edge.type in {EdgeType.CALLS, EdgeType.CONTAINS}:
                    self._reveal_node(edge.target, reachable=True)
                    self._reveal_edge(edge.source, edge.target, edge.type)
                    outcomes.append(f"discovered:{edge.target}")
                    for host in self.true_topology.parent_hosts(edge.target):
                        self._reveal_node(host, reachable=True)
                        for parent_edge in self.true_topology.incoming(edge.target):
                            if parent_edge.source == host:
                                self._reveal_edge(
                                    parent_edge.source,
                                    parent_edge.target,
                                    parent_edge.type,
                                )
                        outcomes.append(f"discovered:{host}")
            prerequisites.append(f"known_component:{target}")
        elif kind == EnterpriseActionType.ASSESS_VULNERABILITY:
            known.assessed.add(target)
            for vuln in self.true_topology.vulnerabilities.values():
                if vuln.target == target and vuln.applies_to(
                    self.true_topology.nodes[target]
                ):
                    known.learn_vulnerability(
                        vuln.id,
                        target,
                        grants_access_to=vuln.grants_access_to,
                        privilege=vuln.privilege,
                    )
                    outcomes.append(f"vulnerability:{vuln.id}")
            prerequisites.append(f"known_target:{target}")
        elif kind == EnterpriseActionType.EXPLOIT:
            vuln = self.true_topology.vulnerabilities[target]
            prerequisites.append(f"known_vulnerability:{target}")
            if self.np_random.random() > vuln.exploit_probability:
                return False, False, -5.0, tuple(prerequisites), (), "exploit attempt failed"
            known.access[vuln.grants_access_to] = vuln.privilege
            self._reveal_node(vuln.grants_access_to)
            for edge in self.true_topology.outgoing(
                vuln.grants_access_to, EdgeType.YIELDS_CREDENTIAL
            ):
                self._reveal_node(edge.target)
                self._reveal_edge(edge.source, edge.target, edge.type)
                outcomes.append(f"credential_source:{edge.target}")
            outcomes.append(f"{vuln.privilege}_access:{vuln.grants_access_to}")
        elif kind == EnterpriseActionType.OBTAIN_CREDENTIAL:
            sources = [
                source
                for source, edge_target in self._known_edges(
                    EdgeType.YIELDS_CREDENTIAL
                )
                if edge_target == target
                if source in known.access
            ]
            known.credentials.add(target)
            for edge in self.true_topology.outgoing(target, EdgeType.HAS_ACCESS):
                self._reveal_node(edge.target, reachable=True)
                self._reveal_edge(edge.source, edge.target, edge.type)
                outcomes.append(f"access_target:{edge.target}")
            prerequisites.extend(f"access:{source}" for source in sources)
            outcomes.append(f"credential:{target}")
        elif kind == EnterpriseActionType.AUTHENTICATE:
            identities = [
                source
                for source, edge_target in self._known_edges(EdgeType.HAS_ACCESS)
                if edge_target == target
                if source in known.credentials
            ]
            host_types = {NodeType.HOST, NodeType.LEGACY_HOST, NodeType.CLOUD_WORKLOAD}
            known.access[target] = (
                "user" if known.known_node_types[target] in host_types else "read"
            )
            self._reveal_host_segments(target, outcomes)
            prerequisites.extend(f"credential:{identity}" for identity in identities)
            outcomes.append(f"authenticated:{target}")
        elif kind == EnterpriseActionType.PIVOT:
            sources = [
                source
                for source, edge_target in self._known_edges(EdgeType.PIVOTS_TO)
                if edge_target == target
                if source in known.access
            ]
            known.access[target] = "user"
            known.reachable.add(target)
            self._reveal_host_segments(target, outcomes)
            prerequisites.extend(f"access:{source}" for source in sources)
            outcomes.append(f"pivoted:{target}")
        elif kind == EnterpriseActionType.ESCALATE_PRIVILEGE:
            known.access[target] = "root"
            prerequisites.append(f"user_access:{target}")
            outcomes.append(f"root_access:{target}")
        elif kind == EnterpriseActionType.ACCESS_ASSET:
            containers = [
                source
                for source, edge_target in self._known_edges(EdgeType.CONTAINS)
                if edge_target == target
                if source in known.access
            ]
            known.accessed_assets.add(target)
            prerequisites.extend(f"access:{source}" for source in containers)
            outcomes.append(f"asset:{target}")

        changed = before != self._snapshot()
        reward = self._reward(kind, target, changed)
        return True, changed, reward, tuple(prerequisites), tuple(outcomes), ""

    def _reveal_host_segments(self, target: str, outcomes: list[str]) -> None:
        for edge in self.true_topology.outgoing(target, EdgeType.LOCATED_IN):
            self._reveal_node(edge.target, reachable=True)
            self._reveal_edge(edge.source, edge.target, edge.type)
            outcomes.append(f"network:{edge.target}")

    def _reward(self, kind: EnterpriseActionType, target: str, changed: bool) -> float:
        if not changed:
            return -1.0
        if (
            kind == EnterpriseActionType.ACCESS_ASSET
            and target in self.true_topology.crown_jewels
        ):
            return 100.0
        if kind == EnterpriseActionType.EXPLOIT:
            return 0.5 + self.true_topology.vulnerabilities[target].cvss / 10.0
        return {
            EnterpriseActionType.DISCOVER_NETWORK: 1.0,
            EnterpriseActionType.ENUMERATE_HOST: 1.0,
            EnterpriseActionType.ENUMERATE_SERVICE: 1.0,
            EnterpriseActionType.ENUMERATE_APPLICATION: 1.0,
            EnterpriseActionType.ASSESS_VULNERABILITY: 1.0,
            EnterpriseActionType.OBTAIN_CREDENTIAL: 2.0,
            EnterpriseActionType.AUTHENTICATE: 3.0,
            EnterpriseActionType.PIVOT: 3.0,
            EnterpriseActionType.ESCALATE_PRIVILEGE: 4.0,
        }.get(kind, 0.0)

    def _snapshot(self) -> tuple:
        return self.knowledge.snapshot()

    def attack_path(self) -> tuple[EnterpriseEvent, ...]:
        """Return the successful, state-changing actions from this episode."""
        return tuple(event for event in self.events if event.success and event.state_changed)

    def render(self) -> str | None:
        if self.render_mode != "ansi":
            return None
        return (
            f"{self.true_topology.name} step={self._step} "
            f"known={len(self.knowledge.discovered)}/{len(self.true_topology.nodes)} "
            f"access={sorted(self.knowledge.access)} "
            f"assets={sorted(self.knowledge.accessed_assets)}"
        )
