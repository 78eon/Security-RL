"""Defender-visible state for simulation-only red-blue interaction.

This module deliberately models only telemetry already revealed by simulated
events. It has no topology, graph, node, edge, or crown-jewel dependency.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np

from rlredteam.enterprise.environment import EnterpriseActionType


class DefenderActionType(StrEnum):
    MONITOR = "monitor"
    LOWER_IDS_THRESHOLD = "lower_ids_threshold"
    RAISE_IDS_THRESHOLD = "raise_ids_threshold"
    SCHEDULE_PATCH = "schedule_patch"


MONITOR_ACTION = 0
LOWER_IDS_ACTION = 1
RAISE_IDS_ACTION = 2
PATCH_ACTION_OFFSET = 3
IDS_LEVEL_COUNT = 3


def defender_action_count(max_vulnerabilities: int) -> int:
    if max_vulnerabilities <= 0:
        raise ValueError("max_vulnerabilities must be positive")
    return PATCH_ACTION_OFFSET + int(max_vulnerabilities)


def defender_action_name(action: int, max_vulnerabilities: int) -> str:
    if action == MONITOR_ACTION:
        return DefenderActionType.MONITOR.value
    if action == LOWER_IDS_ACTION:
        return DefenderActionType.LOWER_IDS_THRESHOLD.value
    if action == RAISE_IDS_ACTION:
        return DefenderActionType.RAISE_IDS_THRESHOLD.value
    slot = action - PATCH_ACTION_OFFSET
    if 0 <= slot < max_vulnerabilities:
        return f"{DefenderActionType.SCHEDULE_PATCH.value}:vulnerability_slot_{slot}"
    raise ValueError(f"defender action {action!r} is outside the fixed catalogue")


@dataclass(frozen=True, slots=True)
class DefenderDecision:
    action: int
    name: str
    state_changed: bool
    scheduled_vulnerability: str | None = None
    activation_episode: int | None = None


@dataclass(slots=True)
class DefenderKnowledge:
    """Facts available to the defensive policy and nothing more."""

    max_vulnerabilities: int
    patch_delay_episodes: int
    initial_ids_level: int = 1
    ids_level: int = field(init=False)
    episode_index: int = field(init=False, default=-1)
    alert_count: int = field(init=False, default=0)
    last_red_action: EnterpriseActionType | None = field(init=False, default=None)
    last_detected: bool = field(init=False, default=False)
    observed_vulnerabilities: list[str] = field(init=False, default_factory=list)
    pending_patches: dict[str, int] = field(init=False, default_factory=dict)
    active_patches: set[str] = field(init=False, default_factory=set)
    newly_activated: tuple[str, ...] = field(init=False, default=())
    decision_count: int = field(init=False, default=0)
    threshold_adjustments: int = field(init=False, default=0)
    patch_schedules: int = field(init=False, default=0)
    patch_activations: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        if self.max_vulnerabilities <= 0:
            raise ValueError("max_vulnerabilities must be positive")
        if self.patch_delay_episodes != 5:
            raise ValueError("Phase 13 requires the preregistered five-episode patch delay")
        if not 0 <= self.initial_ids_level < IDS_LEVEL_COUNT:
            raise ValueError("initial IDS level is outside the fixed catalogue")
        self.ids_level = self.initial_ids_level

    def reset_campaign(self) -> None:
        """Discard all state only at an explicit seeded campaign boundary."""
        self.ids_level = self.initial_ids_level
        self.episode_index = -1
        self.alert_count = 0
        self.last_red_action = None
        self.last_detected = False
        self.observed_vulnerabilities.clear()
        self.pending_patches.clear()
        self.active_patches.clear()
        self.newly_activated = ()
        self.decision_count = 0
        self.threshold_adjustments = 0
        self.patch_schedules = 0
        self.patch_activations = 0

    def begin_episode(self) -> tuple[str, ...]:
        self.episode_index += 1
        due = tuple(
            vulnerability
            for vulnerability, episode in self.pending_patches.items()
            if episode <= self.episode_index
        )
        for vulnerability in due:
            self.pending_patches.pop(vulnerability)
            self.active_patches.add(vulnerability)
        self.newly_activated = due
        self.patch_activations += len(due)
        self.alert_count = 0
        self.last_red_action = None
        self.last_detected = False
        return due

    def observe(
        self,
        action_type: EnterpriseActionType,
        *,
        detected: bool,
        revealed_vulnerabilities: tuple[str, ...] = (),
    ) -> None:
        self.last_red_action = EnterpriseActionType(action_type)
        self.last_detected = bool(detected)
        self.alert_count += int(detected)
        for vulnerability in revealed_vulnerabilities:
            if vulnerability in self.observed_vulnerabilities:
                continue
            if len(self.observed_vulnerabilities) >= self.max_vulnerabilities:
                raise RuntimeError("defender knowledge exceeds vulnerability-slot capacity")
            self.observed_vulnerabilities.append(vulnerability)

    def action_mask(self) -> np.ndarray:
        mask = np.zeros(defender_action_count(self.max_vulnerabilities), dtype=bool)
        mask[MONITOR_ACTION] = True
        mask[LOWER_IDS_ACTION] = self.ids_level > 0
        mask[RAISE_IDS_ACTION] = self.ids_level < IDS_LEVEL_COUNT - 1
        for slot, vulnerability in enumerate(self.observed_vulnerabilities):
            mask[PATCH_ACTION_OFFSET + slot] = (
                vulnerability not in self.pending_patches
                and vulnerability not in self.active_patches
            )
        return mask

    def apply_action(self, action: int) -> DefenderDecision:
        selected = int(action)
        mask = self.action_mask()
        if not 0 <= selected < len(mask) or not mask[selected]:
            raise ValueError("defender selected an action outside DefenderKnowledge mask")
        before_level = self.ids_level
        scheduled = None
        activation_episode = None
        if selected == LOWER_IDS_ACTION:
            self.ids_level -= 1
        elif selected == RAISE_IDS_ACTION:
            self.ids_level += 1
        elif selected >= PATCH_ACTION_OFFSET:
            slot = selected - PATCH_ACTION_OFFSET
            scheduled = self.observed_vulnerabilities[slot]
            activation_episode = self.episode_index + self.patch_delay_episodes
            self.pending_patches[scheduled] = activation_episode
            self.patch_schedules += 1
        changed = self.ids_level != before_level or scheduled is not None
        self.threshold_adjustments += int(self.ids_level != before_level)
        self.decision_count += 1
        return DefenderDecision(
            action=selected,
            name=defender_action_name(selected, self.max_vulnerabilities),
            state_changed=changed,
            scheduled_vulnerability=scheduled,
            activation_episode=activation_episode,
        )

    def snapshot(self) -> tuple:
        return (
            self.max_vulnerabilities,
            self.patch_delay_episodes,
            self.initial_ids_level,
            self.ids_level,
            self.episode_index,
            self.alert_count,
            self.last_red_action.value if self.last_red_action else None,
            self.last_detected,
            tuple(self.observed_vulnerabilities),
            tuple(sorted(self.pending_patches.items())),
            frozenset(self.active_patches),
            self.newly_activated,
            self.decision_count,
            self.threshold_adjustments,
            self.patch_schedules,
            self.patch_activations,
        )


@dataclass(frozen=True, slots=True)
class DefenderObservation:
    """Fixed-size policy input derived exclusively from DefenderKnowledge."""

    values: tuple[float, ...]

    @classmethod
    def from_knowledge(
        cls,
        knowledge: DefenderKnowledge,
        *,
        max_steps: int,
    ) -> DefenderObservation:
        if max_steps <= 0:
            raise ValueError("max_steps must be positive")
        values: list[float] = [
            float(knowledge.ids_level == level) for level in range(IDS_LEVEL_COUNT)
        ]
        values.extend(
            [
                min(knowledge.alert_count / max_steps, 1.0),
                float(knowledge.last_detected),
            ]
        )
        values.extend(
            float(knowledge.last_red_action == kind) for kind in EnterpriseActionType
        )
        for slot in range(knowledge.max_vulnerabilities):
            values.append(float(slot < len(knowledge.observed_vulnerabilities)))
        for slot in range(knowledge.max_vulnerabilities):
            if slot >= len(knowledge.observed_vulnerabilities):
                values.append(0.0)
                continue
            vulnerability = knowledge.observed_vulnerabilities[slot]
            due = knowledge.pending_patches.get(vulnerability)
            remaining = 0 if due is None else max(0, due - knowledge.episode_index)
            values.append(min(remaining / knowledge.patch_delay_episodes, 1.0))
        for slot in range(knowledge.max_vulnerabilities):
            active = (
                slot < len(knowledge.observed_vulnerabilities)
                and knowledge.observed_vulnerabilities[slot] in knowledge.active_patches
            )
            values.append(float(active))
        return cls(tuple(values))

    @staticmethod
    def size(max_vulnerabilities: int) -> int:
        return IDS_LEVEL_COUNT + 2 + len(EnterpriseActionType) + 3 * max_vulnerabilities

    def as_array(self) -> np.ndarray:
        return np.asarray(self.values, dtype=np.float32)


def defender_observation_schema(max_vulnerabilities: int) -> dict[str, int | str]:
    return {
        "source": "DefenderKnowledge",
        "ids_level_values": IDS_LEVEL_COUNT,
        "telemetry_values": 2,
        "red_action_type_values": len(EnterpriseActionType),
        "observed_vulnerability_values": max_vulnerabilities,
        "pending_patch_values": max_vulnerabilities,
        "active_patch_values": max_vulnerabilities,
        "observation_values": DefenderObservation.size(max_vulnerabilities),
        "hidden_topology_fields": 0,
    }


__all__ = [
    "DefenderActionType",
    "DefenderDecision",
    "DefenderKnowledge",
    "DefenderObservation",
    "IDS_LEVEL_COUNT",
    "LOWER_IDS_ACTION",
    "MONITOR_ACTION",
    "PATCH_ACTION_OFFSET",
    "RAISE_IDS_ACTION",
    "defender_action_count",
    "defender_action_name",
    "defender_observation_schema",
]
