"""Gymnasium environment for partial-observability attack-path discovery.

All discovery and attack effects are state transitions over an in-memory graph.
No sockets, subprocesses, scanners or exploit tools are used.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import gymnasium as gym
import numpy as np

from rlredteam.enterprise.model import EdgeType, EnterpriseGraph, NodeType


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


@dataclass(slots=True)
class AgentKnowledge:
    discovered: set[str] = field(default_factory=set)
    enumerated: set[str] = field(default_factory=set)
    reachable: set[str] = field(default_factory=set)
    known_vulnerabilities: set[str] = field(default_factory=set)
    credentials: set[str] = field(default_factory=set)
    access: dict[str, str] = field(default_factory=dict)
    accessed_assets: set[str] = field(default_factory=set)

    def access_level(self, node_id: str) -> int:
        return {"none": 0, "user": 1, "read": 1, "root": 2}.get(
            self.access.get(node_id, "none"), 0
        )


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

    The observation is a flat node-feature matrix. Ground-truth attributes,
    edges and vulnerabilities are not exposed until a matching discovery
    action succeeds. ``info["action_mask"]`` identifies currently valid
    actions for evaluation or a mask-aware policy.
    """

    metadata = {"render_modes": ["ansi"]}

    _TYPE_ORDER = tuple(NodeType)
    _FEATURES_PER_NODE = 7 + len(_TYPE_ORDER)

    def __init__(
        self,
        graph: EnterpriseGraph,
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
        self.graph = graph
        self.max_steps = max_steps
        self.max_nodes = max_nodes
        self.max_vulnerabilities = max_vulnerabilities
        self.render_mode = render_mode
        self.node_ids = tuple(sorted(graph.nodes))
        if len(self.node_ids) > max_nodes:
            raise ValueError(f"graph has {len(self.node_ids)} nodes; capacity is {max_nodes}")
        if len(graph.vulnerabilities) > max_vulnerabilities:
            raise ValueError(
                f"graph has {len(graph.vulnerabilities)} vulnerabilities; "
                f"capacity is {max_vulnerabilities}"
            )
        self._node_slots: tuple[str | None, ...] = self.node_ids + (None,) * (
            max_nodes - len(self.node_ids)
        )
        vulnerability_ids = tuple(sorted(graph.vulnerabilities))
        self._vulnerability_slots: tuple[str | None, ...] = vulnerability_ids + (None,) * (
            max_vulnerabilities - len(vulnerability_ids)
        )
        self.actions = self._build_actions()
        self.action_space = gym.spaces.Discrete(len(self.actions))
        observation_size = max_nodes * self._FEATURES_PER_NODE + 1
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(observation_size,), dtype=np.float32
        )
        self.knowledge = AgentKnowledge()
        self.events: list[EnterpriseEvent] = []
        self._step = 0

    def _build_actions(self) -> tuple[EnterpriseAction, ...]:
        actions: list[EnterpriseAction] = []
        non_exploit = [
            kind for kind in EnterpriseActionType if kind != EnterpriseActionType.EXPLOIT
        ]
        for action_type in non_exploit:
            for index, node_id in enumerate(self._node_slots):
                target = node_id if node_id is not None else f"__empty_node_{index}"
                actions.append(EnterpriseAction(action_type, target))
        for index, vuln_id in enumerate(self._vulnerability_slots):
            target = vuln_id if vuln_id is not None else f"__empty_vulnerability_{index}"
            actions.append(EnterpriseAction(EnterpriseActionType.EXPLOIT, target))
        return tuple(actions)

    def action_index(self, action_type: EnterpriseActionType, target: str) -> int:
        needle = EnterpriseAction(action_type, target)
        try:
            return self.actions.index(needle)
        except ValueError:
            raise KeyError(f"action not in this environment: {needle.name}") from None

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._step = 0
        self.events = []
        self.knowledge = AgentKnowledge(discovered=set(self.graph.entry_points))
        for entry in self.graph.entry_points:
            for edge in self.graph.outgoing(entry, EdgeType.CONNECTS):
                self.knowledge.discovered.add(edge.target)
                self.knowledge.reachable.add(edge.target)
        return self._observation(), self._info()

    def _node_features(self, node_id: str) -> list[float]:
        node = self.graph.nodes[node_id]
        discovered = node_id in self.knowledge.discovered
        known_vulns = sum(
            vuln.target == node_id and vuln.id in self.knowledge.known_vulnerabilities
            for vuln in self.graph.vulnerabilities.values()
        )
        features = [
            float(discovered),
            float(node_id in self.knowledge.enumerated),
            float(node_id in self.knowledge.reachable),
            float(self.knowledge.access_level(node_id) > 0),
            float(self.knowledge.access_level(node_id) == 2),
            float(node_id in self.knowledge.credentials),
            min(float(known_vulns), 1.0),
        ]
        features.extend(float(discovered and node.type == kind) for kind in self._TYPE_ORDER)
        return features

    def _observation(self) -> np.ndarray:
        values: list[float] = []
        for node_id in self._node_slots:
            if node_id is None:
                values.extend([0.0] * self._FEATURES_PER_NODE)
            else:
                values.extend(self._node_features(node_id))
        values.append(min(self._step / self.max_steps, 1.0))
        return np.asarray(values, dtype=np.float32)

    def _info(self) -> dict[str, Any]:
        return {
            "action_mask": self.valid_action_mask(),
            "known_nodes": len(self.knowledge.discovered),
            "known_vulnerabilities": len(self.knowledge.known_vulnerabilities),
            "compromised_nodes": sorted(self.knowledge.access),
            "credentials": sorted(self.knowledge.credentials),
            "goal_reached": bool(self.knowledge.accessed_assets & set(self.graph.crown_jewels)),
        }

    def valid_action_mask(self) -> np.ndarray:
        return np.asarray([self._can_execute(action)[0] for action in self.actions], dtype=np.int8)

    def _can_execute(self, action: EnterpriseAction) -> tuple[bool, str]:
        known = self.knowledge
        target = action.target
        kind = action.type
        if kind == EnterpriseActionType.EXPLOIT:
            if target not in self.graph.vulnerabilities:
                return False, "empty vulnerability slot"
            if target not in known.known_vulnerabilities:
                return False, "vulnerability has not been discovered"
            vuln = self.graph.vulnerabilities[target]
            if vuln.grants_access_to in known.access:
                return False, "target is already compromised"
            return True, ""
        if target not in self.graph.nodes:
            return False, "empty entity slot"
        target_type = self.graph.nodes[target].type
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
        if target not in known.discovered:
            return False, "target has not been discovered"
        if kind == EnterpriseActionType.DISCOVER_NETWORK:
            return target not in known.enumerated, "network already discovered"
        if kind in {
            EnterpriseActionType.ENUMERATE_HOST,
            EnterpriseActionType.ENUMERATE_SERVICE,
            EnterpriseActionType.ENUMERATE_APPLICATION,
        }:
            return target not in known.enumerated, "target already enumerated"
        if kind == EnterpriseActionType.ASSESS_VULNERABILITY:
            vulns = [v for v in self.graph.vulnerabilities.values() if v.target == target]
            if not vulns:
                return False, "target has no vulnerability in ground truth"
            return any(v.id not in known.known_vulnerabilities for v in vulns), "already assessed"
        if kind == EnterpriseActionType.OBTAIN_CREDENTIAL:
            if target in known.credentials:
                return False, "credential already obtained"
            possible = any(
                edge.target == target and edge.source in known.access
                for edge in self.graph.edges
                if edge.type == EdgeType.YIELDS_CREDENTIAL
            )
            return possible, "no compromised source yields this credential"
        if kind == EnterpriseActionType.AUTHENTICATE:
            if target in known.access:
                return False, "target is already accessed"
            possible = any(
                edge.target == target and edge.source in known.credentials
                for edge in self.graph.edges
                if edge.type == EdgeType.HAS_ACCESS
            )
            return possible, "no owned credential grants access"
        if kind == EnterpriseActionType.PIVOT:
            if target in known.access:
                return False, "target is already compromised"
            possible = any(
                edge.target == target and edge.source in known.access
                for edge in self.graph.edges
                if edge.type == EdgeType.PIVOTS_TO
            )
            return possible, "no compromised source can pivot to target"
        if kind == EnterpriseActionType.ESCALATE_PRIVILEGE:
            if known.access_level(target) != 1:
                return False, "user access is required"
            possible = any(
                vuln.target == target
                and vuln.privilege == "root"
                and vuln.id in known.known_vulnerabilities
                for vuln in self.graph.vulnerabilities.values()
            )
            return possible, "no known privilege-escalation vulnerability"
        if kind == EnterpriseActionType.ACCESS_ASSET:
            if target in known.accessed_assets:
                return False, "asset already accessed"
            possible = any(
                edge.target == target and edge.source in known.access
                for edge in self.graph.edges
                if edge.type == EdgeType.CONTAINS
            )
            return possible, "containing system has not been accessed"
        return False, "unsupported action"

    def step(self, action: int):
        if not self.action_space.contains(action):
            raise ValueError(f"action {action!r} outside [0, {self.action_space.n})")
        selected = self.actions[int(action)]
        valid, reason = self._can_execute(selected)
        self._step += 1
        if valid:
            success, changed, reward, prerequisites, outcomes, reason = self._execute(selected)
        else:
            success, changed, reward = False, False, -5.0
            prerequisites, outcomes = (), ()

        goal = bool(self.knowledge.accessed_assets & set(self.graph.crown_jewels))
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
            for edge in self.graph.edges:
                if edge.type == EdgeType.LOCATED_IN and edge.target == target:
                    known.discovered.add(edge.source)
                    known.reachable.add(edge.source)
                    outcomes.append(f"discovered:{edge.source}")
                elif edge.type == EdgeType.PROTECTS and edge.target == target:
                    known.discovered.add(edge.source)
                    outcomes.append(f"control:{edge.source}")
            prerequisites.append(f"reachable:{target}")
        elif kind == EnterpriseActionType.ENUMERATE_HOST:
            known.enumerated.add(target)
            for edge in self.graph.outgoing(target):
                if edge.type in {EdgeType.HOSTS, EdgeType.YIELDS_CREDENTIAL, EdgeType.PIVOTS_TO}:
                    known.discovered.add(edge.target)
                    if edge.type != EdgeType.YIELDS_CREDENTIAL:
                        known.reachable.add(edge.target)
                    outcomes.append(f"discovered:{edge.target}")
            prerequisites.append(f"known_host:{target}")
        elif kind == EnterpriseActionType.ENUMERATE_SERVICE:
            known.enumerated.add(target)
            for child in self.graph.children(target, EdgeType.EXPOSES):
                known.discovered.add(child)
                known.reachable.add(child)
                outcomes.append(f"discovered:{child}")
            prerequisites.append(f"known_service:{target}")
        elif kind == EnterpriseActionType.ENUMERATE_APPLICATION:
            known.enumerated.add(target)
            for edge in self.graph.outgoing(target):
                if edge.type in {EdgeType.CALLS, EdgeType.CONTAINS}:
                    known.discovered.add(edge.target)
                    known.reachable.add(edge.target)
                    outcomes.append(f"discovered:{edge.target}")
                    for host in self.graph.parent_hosts(edge.target):
                        known.discovered.add(host)
                        known.reachable.add(host)
                        outcomes.append(f"discovered:{host}")
            prerequisites.append(f"known_component:{target}")
        elif kind == EnterpriseActionType.ASSESS_VULNERABILITY:
            for vuln in self.graph.vulnerabilities.values():
                if vuln.target == target and vuln.applies_to(self.graph.nodes[target]):
                    known.known_vulnerabilities.add(vuln.id)
                    outcomes.append(f"vulnerability:{vuln.id}")
            prerequisites.append(f"known_target:{target}")
        elif kind == EnterpriseActionType.EXPLOIT:
            vuln = self.graph.vulnerabilities[target]
            prerequisites.append(f"known_vulnerability:{target}")
            if self.np_random.random() > vuln.exploit_probability:
                return False, False, -5.0, tuple(prerequisites), (), "exploit attempt failed"
            known.access[vuln.grants_access_to] = vuln.privilege
            known.discovered.add(vuln.grants_access_to)
            for edge in self.graph.outgoing(vuln.grants_access_to, EdgeType.YIELDS_CREDENTIAL):
                known.discovered.add(edge.target)
                outcomes.append(f"credential_source:{edge.target}")
            outcomes.append(f"{vuln.privilege}_access:{vuln.grants_access_to}")
        elif kind == EnterpriseActionType.OBTAIN_CREDENTIAL:
            sources = [
                edge.source
                for edge in self.graph.incoming(target, EdgeType.YIELDS_CREDENTIAL)
                if edge.source in known.access
            ]
            known.credentials.add(target)
            prerequisites.extend(f"access:{source}" for source in sources)
            outcomes.append(f"credential:{target}")
        elif kind == EnterpriseActionType.AUTHENTICATE:
            identities = [
                edge.source
                for edge in self.graph.incoming(target, EdgeType.HAS_ACCESS)
                if edge.source in known.credentials
            ]
            host_types = {NodeType.HOST, NodeType.LEGACY_HOST, NodeType.CLOUD_WORKLOAD}
            known.access[target] = "user" if self.graph.nodes[target].type in host_types else "read"
            prerequisites.extend(f"credential:{identity}" for identity in identities)
            outcomes.append(f"authenticated:{target}")
        elif kind == EnterpriseActionType.PIVOT:
            sources = [
                edge.source
                for edge in self.graph.incoming(target, EdgeType.PIVOTS_TO)
                if edge.source in known.access
            ]
            known.access[target] = "user"
            known.reachable.add(target)
            prerequisites.extend(f"access:{source}" for source in sources)
            outcomes.append(f"pivoted:{target}")
        elif kind == EnterpriseActionType.ESCALATE_PRIVILEGE:
            known.access[target] = "root"
            prerequisites.append(f"user_access:{target}")
            outcomes.append(f"root_access:{target}")
        elif kind == EnterpriseActionType.ACCESS_ASSET:
            containers = [
                edge.source
                for edge in self.graph.incoming(target, EdgeType.CONTAINS)
                if edge.source in known.access
            ]
            known.accessed_assets.add(target)
            prerequisites.extend(f"access:{source}" for source in containers)
            outcomes.append(f"asset:{target}")

        changed = before != self._snapshot()
        reward = self._reward(kind, target, changed)
        return True, changed, reward, tuple(prerequisites), tuple(outcomes), ""

    def _reward(self, kind: EnterpriseActionType, target: str, changed: bool) -> float:
        if not changed:
            return -1.0
        if kind == EnterpriseActionType.ACCESS_ASSET and target in self.graph.crown_jewels:
            return 100.0
        if kind == EnterpriseActionType.EXPLOIT:
            return 0.5 + self.graph.vulnerabilities[target].cvss / 10.0
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
        known = self.knowledge
        return (
            frozenset(known.discovered),
            frozenset(known.enumerated),
            frozenset(known.reachable),
            frozenset(known.known_vulnerabilities),
            frozenset(known.credentials),
            tuple(sorted(known.access.items())),
            frozenset(known.accessed_assets),
        )

    def attack_path(self) -> tuple[EnterpriseEvent, ...]:
        """Return the successful, state-changing actions from this episode."""
        return tuple(event for event in self.events if event.success and event.state_changed)

    def render(self) -> str | None:
        if self.render_mode != "ansi":
            return None
        return (
            f"{self.graph.name} step={self._step} "
            f"known={len(self.knowledge.discovered)}/{len(self.graph.nodes)} "
            f"access={sorted(self.knowledge.access)} "
            f"assets={sorted(self.knowledge.accessed_assets)}"
        )
