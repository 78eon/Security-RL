"""Agent-visible state and observation encoding.

This module deliberately has no reference to ``TrueTopology``.  That makes it
impossible for observation construction to read undiscovered nodes, edges or
vulnerabilities by accident.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from rlredteam.enterprise.model import EdgeType, NodeType


@dataclass(slots=True)
class AgentKnowledge:
    """Only facts legitimately revealed to the agent during an episode."""

    discovered: set[str] = field(default_factory=set)
    discovery_order: list[str] = field(default_factory=list)
    known_node_types: dict[str, NodeType] = field(default_factory=dict)
    known_edges: set[tuple[str, str, EdgeType]] = field(default_factory=set)
    enumerated: set[str] = field(default_factory=set)
    assessed: set[str] = field(default_factory=set)
    reachable: set[str] = field(default_factory=set)
    known_vulnerabilities: set[str] = field(default_factory=set)
    vulnerability_order: list[str] = field(default_factory=list)
    vulnerability_targets: dict[str, str] = field(default_factory=dict)
    vulnerability_grants_access: dict[str, str] = field(default_factory=dict)
    vulnerability_privileges: dict[str, str] = field(default_factory=dict)
    credentials: set[str] = field(default_factory=set)
    access: dict[str, str] = field(default_factory=dict)
    accessed_assets: set[str] = field(default_factory=set)

    def discover(self, node_id: str, node_type: NodeType, *, reachable: bool = False) -> None:
        if node_id not in self.discovered:
            self.discovered.add(node_id)
            self.discovery_order.append(node_id)
        self.known_node_types[node_id] = node_type
        if reachable:
            self.reachable.add(node_id)

    def learn_edge(self, source: str, target: str, edge_type: EdgeType) -> None:
        if source not in self.discovered or target not in self.discovered:
            raise ValueError("both edge endpoints must be discovered before revealing an edge")
        self.known_edges.add((source, target, edge_type))

    def learn_vulnerability(
        self,
        vulnerability_id: str,
        target: str,
        *,
        grants_access_to: str,
        privilege: str,
    ) -> None:
        if target not in self.discovered:
            raise ValueError("a vulnerability target must be discovered first")
        if vulnerability_id not in self.known_vulnerabilities:
            self.known_vulnerabilities.add(vulnerability_id)
            self.vulnerability_order.append(vulnerability_id)
        self.vulnerability_targets[vulnerability_id] = target
        self.vulnerability_grants_access[vulnerability_id] = grants_access_to
        self.vulnerability_privileges[vulnerability_id] = privilege

    def node_slot(self, node_id: str) -> int:
        try:
            return self.discovery_order.index(node_id)
        except ValueError:
            raise KeyError(f"node is not known to the agent: {node_id}") from None

    def vulnerability_slot(self, vulnerability_id: str) -> int:
        try:
            return self.vulnerability_order.index(vulnerability_id)
        except ValueError:
            raise KeyError(
                f"vulnerability is not known to the agent: {vulnerability_id}"
            ) from None

    def access_level(self, node_id: str) -> int:
        return {"none": 0, "user": 1, "read": 1, "root": 2}.get(
            self.access.get(node_id, "none"), 0
        )

    def snapshot(self) -> tuple:
        """Hashable research/debug snapshot of all agent-visible state."""
        return (
            frozenset(self.discovered),
            tuple(self.discovery_order),
            tuple(
                sorted(
                    (node_id, kind.value)
                    for node_id, kind in self.known_node_types.items()
                )
            ),
            tuple(
                sorted(
                    (source, target, kind.value)
                    for source, target, kind in self.known_edges
                )
            ),
            frozenset(self.enumerated),
            frozenset(self.assessed),
            frozenset(self.reachable),
            frozenset(self.known_vulnerabilities),
            tuple(self.vulnerability_order),
            tuple(sorted(self.vulnerability_targets.items())),
            tuple(sorted(self.vulnerability_grants_access.items())),
            tuple(sorted(self.vulnerability_privileges.items())),
            frozenset(self.credentials),
            tuple(sorted(self.access.items())),
            frozenset(self.accessed_assets),
        )


@dataclass(frozen=True, slots=True)
class Observation:
    """Immutable policy observation derived exclusively from AgentKnowledge."""

    values: tuple[float, ...]

    @staticmethod
    def feature_count(type_order: tuple[NodeType, ...]) -> int:
        # discovered, enumerated, assessed, reachable, accessed, root,
        # credential-owned, and known-vulnerability flags.
        return 8 + len(type_order)

    @classmethod
    def from_knowledge(
        cls,
        knowledge: AgentKnowledge,
        *,
        max_nodes: int,
        step: int,
        max_steps: int,
        type_order: tuple[NodeType, ...] = tuple(NodeType),
    ) -> Observation:
        if len(knowledge.discovery_order) > max_nodes:
            raise ValueError("agent knowledge exceeds observation node capacity")

        feature_count = cls.feature_count(type_order)
        values: list[float] = []
        for index in range(max_nodes):
            if index >= len(knowledge.discovery_order):
                values.extend([0.0] * feature_count)
                continue
            node_id = knowledge.discovery_order[index]
            known_vulns = sum(
                target == node_id for target in knowledge.vulnerability_targets.values()
            )
            values.extend(
                [
                    1.0,
                    float(node_id in knowledge.enumerated),
                    float(node_id in knowledge.assessed),
                    float(node_id in knowledge.reachable),
                    float(knowledge.access_level(node_id) > 0),
                    float(knowledge.access_level(node_id) == 2),
                    float(node_id in knowledge.credentials),
                    min(float(known_vulns), 1.0),
                ]
            )
            known_type = knowledge.known_node_types[node_id]
            values.extend(float(known_type == kind) for kind in type_order)

        slot_by_node = {
            node_id: index for index, node_id in enumerate(knowledge.discovery_order)
        }
        adjacency = [0.0] * (max_nodes * max_nodes)
        for source, target, _ in knowledge.known_edges:
            source_slot = slot_by_node[source]
            target_slot = slot_by_node[target]
            adjacency[source_slot * max_nodes + target_slot] = 1.0
        values.extend(adjacency)
        values.append(min(step / max_steps, 1.0))
        return cls(tuple(values))

    def as_array(self) -> np.ndarray:
        return np.asarray(self.values, dtype=np.float32)
