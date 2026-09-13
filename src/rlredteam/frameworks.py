"""Versioned MITRE semantics for simulator events.

This module is deliberately independent of the policy, environment topology,
reward engine and database.  A policy action index is not a MITRE technique;
the resolved simulator behaviour is mapped only when the model implements the
behaviour described by ATT&CK or ATLAS.  Unknown and abstract actions remain
unmapped instead of being coerced into a framework.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from enum import StrEnum

from rlredteam.catalogues import load_mitre_catalogue

_CATALOGUE = load_mitre_catalogue()
MAPPING_VERSION = str(_CATALOGUE["version"])


class Framework(StrEnum):
    ATTACK_ENTERPRISE = "attack-enterprise"
    ATLAS = "atlas"


class AIBehavior(StrEnum):
    """AI-targeting behaviours that a simulator must model explicitly."""

    MODEL_INFERENCE_API_ACCESS = "model_inference_api_access"
    FULL_MODEL_ACCESS = "full_model_access"
    DISCOVER_AI_ARTIFACTS = "discover_ai_artifacts"
    DISCOVER_MODEL_FAMILY = "discover_model_family"
    DISCOVER_MODEL_ONTOLOGY = "discover_model_ontology"
    AI_ARTIFACT_COLLECTION = "ai_artifact_collection"
    EXFILTRATE_VIA_INFERENCE_API = "exfiltrate_via_inference_api"


@dataclass(frozen=True, slots=True)
class TechniqueMapping:
    framework: Framework
    tactic_id: str
    tactic_name: str
    technique_id: str
    technique_name: str
    source_url: str
    mapping_version: str = MAPPING_VERSION

    def as_dict(self) -> dict[str, str]:
        value = asdict(self)
        value["framework"] = self.framework.value
        return value


@dataclass(frozen=True, slots=True)
class TechniqueProgress:
    technique_id: str
    attempts: int
    successful: int
    progressed: int
    current: bool


def _techniques(framework: Framework) -> tuple[TechniqueMapping, ...]:
    return tuple(
        TechniqueMapping(
            framework=framework,
            tactic_id=str(item["tactic_id"]),
            tactic_name=str(item["tactic_name"]),
            technique_id=str(item["technique_id"]),
            technique_name=str(item["technique_name"]),
            source_url=str(item["source_url"]),
        )
        for item in _CATALOGUE["frameworks"][framework.value]["techniques"]
    )


ATTACK_TECHNIQUES = _techniques(Framework.ATTACK_ENTERPRISE)
ATLAS_TECHNIQUES = _techniques(Framework.ATLAS)

_BY_ID = {
    mapping.technique_id: mapping
    for mapping in (*ATTACK_TECHNIQUES, *ATLAS_TECHNIQUES)
}

# Exact simulator semantics.  ``obtain_credential`` is intentionally absent:
# the simulator models an abstract state transition but not the credential
# acquisition mechanism needed to support a specific ATT&CK technique.
_ACTION_TECHNIQUE = {
    str(action): str(technique)
    for action, technique in _CATALOGUE["simulator_action_mappings"].items()
}
_AI_TECHNIQUE = {
    AIBehavior(action): str(technique)
    for action, technique in _CATALOGUE["ai_behavior_mappings"].items()
}


def framework_catalog(framework: Framework | str) -> tuple[TechniqueMapping, ...]:
    selected = Framework(framework)
    return ATTACK_TECHNIQUES if selected is Framework.ATTACK_ENTERPRISE else ATLAS_TECHNIQUES


def map_simulator_behavior(
    action_kind: str,
    *,
    ai_behavior: AIBehavior | str | None = None,
) -> tuple[TechniqueMapping, ...]:
    """Map only supported simulator semantics; return empty for unknowns."""
    output: list[TechniqueMapping] = []
    attack_id = _ACTION_TECHNIQUE.get(str(action_kind))
    if attack_id:
        output.append(_BY_ID[attack_id])
    if ai_behavior is not None:
        try:
            selected_ai = AIBehavior(ai_behavior)
        except ValueError:
            selected_ai = None
        if selected_ai is not None:
            output.append(_BY_ID[_AI_TECHNIQUE[selected_ai]])
    return tuple(output)


def event_framework_fields(
    *,
    rl_action_index: int | None,
    simulator_action: str,
    action_kind: str,
    ai_behavior: AIBehavior | str | None = None,
) -> dict[str, object]:
    """Return storage/UI fields without topology or policy-observation data."""
    if rl_action_index is not None and rl_action_index < 0:
        raise ValueError("RL action index cannot be negative")
    return {
        "rl_action_index": rl_action_index,
        "simulator_action": simulator_action,
        "framework_mappings": [
            mapping.as_dict()
            for mapping in map_simulator_behavior(action_kind, ai_behavior=ai_behavior)
        ],
    }


def technique_progress(
    events: Iterable[Mapping[str, object]],
    framework: Framework | str,
) -> dict[str, TechniqueProgress]:
    """Aggregate matrix state from event mappings only.

    The function has no topology argument by design.  A matrix therefore
    cannot reveal a technique merely because a compatible hidden node exists.
    """
    selected = Framework(framework)
    rows = list(events)
    counts: dict[str, list[int | bool]] = {}
    for event_index, event in enumerate(rows):
        mappings = event.get("framework_mappings")
        if not isinstance(mappings, list | tuple):
            continue
        for raw in mappings:
            if not isinstance(raw, Mapping) or raw.get("framework") != selected.value:
                continue
            technique_id = str(raw.get("technique_id") or "")
            if technique_id not in _BY_ID:
                continue
            values = counts.setdefault(technique_id, [0, 0, 0, False])
            values[0] = int(values[0]) + 1
            if bool(event.get("success")):
                values[1] = int(values[1]) + 1
            if bool(event.get("success")) and bool(event.get("state_changed")):
                values[2] = int(values[2]) + 1
            if event_index == len(rows) - 1:
                values[3] = True
    return {
        technique_id: TechniqueProgress(
            technique_id=technique_id,
            attempts=int(values[0]),
            successful=int(values[1]),
            progressed=int(values[2]),
            current=bool(values[3]),
        )
        for technique_id, values in counts.items()
    }
