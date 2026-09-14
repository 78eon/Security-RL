"""Minimal semantic boundary shared by simulator integrations.

Adapters stop at Security-RL semantics.  Framework mapping is deliberately a
separate operation performed when a transition is serialised for reporting:

native action -> semantic transition -> versioned catalogue mapping -> report
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from rlredteam.enterprise.state import AgentKnowledge
from rlredteam.events import AttackEvent
from rlredteam.frameworks import event_framework_fields


@dataclass(frozen=True, slots=True)
class SemanticAction:
    """A stable policy action with simulator-native identity kept separate."""

    index: int
    native_kind: str
    simulator_action: str
    semantic_behavior: str
    native_parameters: tuple[int, ...] = ()
    simulator_vulnerability_id: str | None = None
    cve_id: str | None = None


@dataclass(frozen=True, slots=True)
class KnowledgeDelta:
    """Only newly observed facts from one simulator step."""

    discovered_nodes: tuple[str, ...] = ()
    discovered_properties: tuple[tuple[str, str], ...] = ()
    discovered_services: tuple[tuple[str, str], ...] = ()
    access_changes: tuple[tuple[str, str], ...] = ()
    observed_edges: tuple[tuple[str, str, str], ...] = ()
    credential_targets: tuple[str, ...] = ()

    @property
    def changed(self) -> bool:
        return any(asdict(self).values())

    def to_dict(self) -> dict[str, list[Any]]:
        return {key: [list(item) if isinstance(item, tuple) else item for item in value]
                for key, value in asdict(self).items() if value}


@dataclass(frozen=True, slots=True)
class SimulatorTransition:
    """One adapter step after native values have acquired stable semantics."""

    observation: np.ndarray
    reward: float
    terminated: bool
    truncated: bool
    event: AttackEvent
    action: SemanticAction
    semantic_behavior: str
    simulator_action: str
    simulator_vulnerability_id: str | None
    knowledge_delta: KnowledgeDelta

    def to_trajectory_event(self) -> dict[str, Any]:
        """Map semantics through the catalogue; never map the raw action index."""
        event = self.event
        target: str | list[int] | None
        if isinstance(event.target, tuple):
            target = list(event.target)
        else:
            target = event.target
        prerequisites: set[str] = set()
        outcomes: set[str] = set()
        for source, edge_target, _ in self.knowledge_delta.observed_edges:
            prerequisites.update({f"access:{source}", f"credential:{edge_target}"})
            outcomes.update({f"access:{edge_target}", f"pivoted:{edge_target}"})
        if not self.knowledge_delta.observed_edges and isinstance(event.target, str):
            prerequisites.add(f"access:{event.target}")
        outcomes.update(
            f"discovered:{node}" for node in self.knowledge_delta.discovered_nodes
        )
        outcomes.update(
            f"credential:{node}" for node in self.knowledge_delta.credential_targets
        )
        outcomes.update(
            f"access:{node}" for node, _ in self.knowledge_delta.access_changes
        )
        return {
            "step": event.step,
            "action_kind": self.semantic_behavior,
            **event_framework_fields(
                rl_action_index=event.rl_action_index,
                simulator_action=self.simulator_action,
                action_kind=self.semantic_behavior,
            ),
            "target": target,
            "success": event.success,
            "state_changed": self.knowledge_delta.changed,
            "native_reward": event.native_reward,
            "reward": self.reward,
            "cve_id": event.cve_id,
            "cvss_score": event.cvss_base,
            "simulator_vulnerability_id": self.simulator_vulnerability_id,
            "access_gained": int(event.access_gained),
            "newly_discovered": event.newly_discovered,
            "is_crown_jewel": event.is_crown_jewel,
            "goal_reached": event.goal_reached,
            "terminal": event.terminal,
            "error": event.error,
            "knowledge_delta": self.knowledge_delta.to_dict() or None,
            "prerequisites": sorted(prerequisites),
            "outcomes": sorted(outcomes),
        }


class SimulatorAdapter(ABC):
    """The simulator semantics Security-RL consumes, and nothing more."""

    @property
    @abstractmethod
    def agent_knowledge(self) -> AgentKnowledge: ...

    @abstractmethod
    def reset(self, *, seed: int | None = None) -> tuple[np.ndarray, dict[str, Any]]: ...

    @abstractmethod
    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]: ...

    @abstractmethod
    def action_catalogue(self) -> tuple[SemanticAction, ...]: ...

    @abstractmethod
    def action_mask(self) -> np.ndarray | None: ...

    @abstractmethod
    def convert_observation(self, native_observation: Any) -> np.ndarray: ...

    @abstractmethod
    def update_agent_knowledge(
        self,
        native_observation: Any,
        *,
        native_info: Mapping[str, Any] | None = None,
    ) -> KnowledgeDelta: ...

    @abstractmethod
    def native_result_to_event(
        self, native_result: Mapping[str, Any]
    ) -> tuple[AttackEvent, str]: ...
