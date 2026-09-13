"""Deterministic, evidence-only attack-path reporting for Phase 14.

The module preserves the semantic boundary:

RL action -> simulator action -> recorded event -> versioned MITRE mapping -> facts.

It never accepts topology state as an input.  Reports can therefore describe only
what the recorded trajectory observed.  In particular, observed path criticality
is not a counterfactual patching claim.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from rlredteam.catalogues import mitre_catalogue_digest
from rlredteam.enterprise.trajectory import reconstruct_attack_path
from rlredteam.frameworks import MAPPING_VERSION, AIBehavior, map_simulator_behavior

REPORT_SCHEMA_VERSION = "security-rl-attack-path-report-v1"
CRITICALITY_METHOD = "observed_path_criticality_not_counterfactual"
_CVE = re.compile(r"^CVE-\d{4}-\d{4,}$")
_FORBIDDEN_TOPOLOGY_KEYS = frozenset(
    {
        "true_topology",
        "truetopology",
        "hidden_topology",
        "hidden_nodes",
        "true_nodes",
        "ground_truth",
        "undiscovered_nodes",
    }
)


class EvidenceValidationError(ValueError):
    """Recorded input cannot safely support the requested report facts."""


class ReportMode(StrEnum):
    SIMULATION = "simulation_report"
    EVIDENCE = "evidence_report"


@dataclass(frozen=True, slots=True)
class ReportFact:
    episode_id: str
    run_id: str
    framework: str | None
    tactic_id: str | None
    tactic_name: str | None
    technique_id: str | None
    technique_name: str | None
    trajectory_step: int
    rl_action_index: int | None
    simulator_action: str
    action_kind: str
    target: str | list[int] | None
    cve_id: str | None
    cvss_score: float | None
    success: bool
    access_gained: int | str | None
    crown_jewel_objective: bool
    state_changed: bool
    knowledge_delta: dict[str, Any] | None
    prerequisites: tuple[str, ...]
    outcomes: tuple[str, ...]
    reward: float | None
    source_event_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return _json_value(asdict(self))


@dataclass(frozen=True, slots=True)
class AttackPhase:
    episode_id: str
    framework: str | None
    tactic_id: str | None
    tactic_name: str
    first_step: int
    last_step: int
    fact_indexes: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class ObservedPathCriticality:
    cve_id: str
    successful_trajectories_containing: int
    observed_successful_paths_broken: int
    observed_path_criticality_percentage: float
    cvss_score: float | None
    framework: str | None
    tactic_id: str | None
    tactic_name: str | None
    technique_id: str | None
    technique_name: str | None
    observed_targets: tuple[str, ...]
    statement: str


@dataclass(frozen=True, slots=True)
class ReportProvenance:
    source_trajectory_sha256: str
    facts_payload_sha256: str
    mitre_catalogue_sha256: str
    mitre_catalogue_version: str
    experiment_id: str
    run_ids: tuple[str, ...]
    source_checkpoint_hashes: dict[str, str]
    report_schema_version: str = REPORT_SCHEMA_VERSION


@dataclass(frozen=True, slots=True)
class AttackPathReport:
    report_id: str
    report_mode: str
    provenance: ReportProvenance
    facts: tuple[ReportFact, ...]
    phases: tuple[AttackPhase, ...]
    observed_path_criticality: tuple[ObservedPathCriticality, ...]
    cross_episode_analysis: dict[str, Any]
    observed_graph: dict[str, list[dict[str, Any]]]
    methodology: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return _json_value(asdict(self))


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def _json_value(value: Any) -> Any:
    """Normalise tuples to the exact representation PostgreSQL JSONB returns."""
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def source_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def generate_report(
    source_path: Path,
    *,
    mode: ReportMode | str,
    experiment_id: str,
    checkpoint_hashes: Mapping[str, str] | None = None,
) -> AttackPathReport:
    """Build a deterministic report from a recorded trajectory document."""
    path = Path(source_path)
    try:
        document = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise EvidenceValidationError(f"cannot read trajectory evidence: {exc}") from exc
    return build_report(
        document,
        source_trajectory_hash=source_sha256(path),
        mode=mode,
        experiment_id=experiment_id,
        checkpoint_hashes=checkpoint_hashes,
    )


def build_report(
    document: Any,
    *,
    source_trajectory_hash: str,
    mode: ReportMode | str,
    experiment_id: str,
    checkpoint_hashes: Mapping[str, str] | None = None,
) -> AttackPathReport:
    selected_mode = ReportMode(mode)
    trajectories = _trajectory_list(document)
    if not trajectories:
        raise EvidenceValidationError("trajectory evidence contains no episodes")
    _require_sha256(source_trajectory_hash, "source trajectory hash")
    if not str(experiment_id).strip():
        raise EvidenceValidationError("experiment ID is required")

    facts: list[ReportFact] = []
    episode_events: dict[str, list[dict[str, Any]]] = {}
    run_ids: set[str] = set()
    successful: set[str] = set()
    failed_action_observation_complete = True
    for trajectory_index, trajectory in enumerate(trajectories):
        if not isinstance(trajectory, Mapping):
            raise EvidenceValidationError(f"trajectory {trajectory_index} is not an object")
        _reject_hidden_topology(trajectory)
        _validate_mode(trajectory, selected_mode)
        run_id = str(
            trajectory.get("run_name")
            or trajectory.get("run_id")
            or f"{experiment_id}:source"
        )
        episode_suffix = trajectory.get(
            "evaluation_seed", trajectory.get("episode_id", trajectory_index)
        )
        episode_id = f"{run_id}:{episode_suffix}"
        run_ids.add(run_id)
        goal_reached = bool(
            trajectory.get("goal_reached", trajectory.get("objective_reached", False))
        )
        if goal_reached:
            successful.add(episode_id)
        raw_events, progress_contract = _events(trajectory)
        failed_action_observation_complete &= not progress_contract
        normalised: list[dict[str, Any]] = []
        for raw_index, raw in enumerate(raw_events):
            event, event_facts = _normalise_event(
                raw,
                episode_id=episode_id,
                run_id=run_id,
                raw_index=raw_index,
                goal_reached=goal_reached,
                progress_contract=progress_contract,
                mode=selected_mode,
            )
            normalised.append(event)
            facts.extend(event_facts)
        episode_events[episode_id] = normalised

    facts_payload = [fact.to_dict() for fact in facts]
    facts_hash = canonical_sha256(facts_payload)
    mitre_hash = mitre_catalogue_digest()
    checkpoints = _validate_checkpoint_hashes(checkpoint_hashes or {}, run_ids)
    provenance = ReportProvenance(
        source_trajectory_sha256=source_trajectory_hash,
        facts_payload_sha256=facts_hash,
        mitre_catalogue_sha256=mitre_hash,
        mitre_catalogue_version=MAPPING_VERSION,
        experiment_id=str(experiment_id),
        run_ids=tuple(sorted(run_ids)),
        source_checkpoint_hashes=checkpoints,
    )
    criticality = _criticality(facts, episode_events, successful)
    phases = _phases(facts)
    analysis = _cross_episode(
        facts,
        episode_events,
        successful,
        failed_action_observation_complete=failed_action_observation_complete,
    )
    observed_graph = _observed_graph(facts)
    methodology = {
        "criticality_method": CRITICALITY_METHOD,
        "criticality_disclaimer": (
            "Observed path criticality describes recorded successful trajectories only; "
            "it is not a mitigated-environment counterfactual or a claim about all attacks."
        ),
        "ground_truth_policy": (
            "Normal reports contain recorded observations only and never hidden TrueTopology."
        ),
        "mode_claim_boundary": (
            "Synthetic PPO/simulator behavior only; targets are synthetic identifiers."
            if selected_mode is ReportMode.SIMULATION
            else (
                "Imported observations only; this report does not claim PPO exploited "
                "a real target."
            )
        ),
    }
    report_identity = {
        "mode": selected_mode.value,
        "provenance": asdict(provenance),
        "facts": facts_payload,
        "phases": [asdict(item) for item in phases],
        "observed_path_criticality": [asdict(item) for item in criticality],
        "cross_episode_analysis": analysis,
        "observed_graph": observed_graph,
        "methodology": methodology,
    }
    return AttackPathReport(
        report_id=canonical_sha256(report_identity),
        report_mode=selected_mode.value,
        provenance=provenance,
        facts=tuple(facts),
        phases=phases,
        observed_path_criticality=criticality,
        cross_episode_analysis=analysis,
        observed_graph=observed_graph,
        methodology=methodology,
    )


def write_report(report: AttackPathReport, output: Path) -> None:
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True, allow_nan=False) + "\n"
    )


def checkpoint_hashes_from_metadata(path: Path) -> dict[str, str]:
    """Extract only recorded run/checkpoint hashes from experiment metadata."""
    try:
        raw = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise EvidenceValidationError(f"cannot read experiment metadata: {exc}") from exc
    output: dict[str, str] = {}
    for run in raw.get("runs", []):
        if not isinstance(run, Mapping):
            continue
        name, digest = str(run.get("run_name", "")), str(run.get("checkpoint_sha256", ""))
        if name and digest:
            _require_sha256(digest, f"checkpoint hash for {name}")
            output[name] = digest
    return output


def _trajectory_list(document: Any) -> list[Any]:
    if isinstance(document, list):
        return document
    if isinstance(document, Mapping) and isinstance(document.get("trajectories"), list):
        return list(document["trajectories"])
    raise EvidenceValidationError("trajectory evidence must be a list or trajectories object")


def _events(trajectory: Mapping[str, Any]) -> tuple[list[Any], bool]:
    if isinstance(trajectory.get("events"), list):
        return list(trajectory["events"]), False
    if isinstance(trajectory.get("steps"), list):
        return list(trajectory["steps"]), False
    if isinstance(trajectory.get("progress_steps"), list):
        return list(trajectory["progress_steps"]), True
    raise EvidenceValidationError("trajectory has no recorded events or progress steps")


def _validate_mode(trajectory: Mapping[str, Any], mode: ReportMode) -> None:
    if mode is not ReportMode.EVIDENCE:
        return
    simulation_claims = {
        "reward_mode",
        "policy_return",
        "native_return",
        "training_seed",
        "checkpoint_sha256",
    }
    present = sorted(key for key in simulation_claims if trajectory.get(key) is not None)
    if present:
        raise EvidenceValidationError(
            "evidence_report cannot contain PPO/simulator claims: " + ", ".join(present)
        )


def _normalise_event(
    raw: Any,
    *,
    episode_id: str,
    run_id: str,
    raw_index: int,
    goal_reached: bool,
    progress_contract: bool,
    mode: ReportMode,
) -> tuple[dict[str, Any], list[ReportFact]]:
    if not isinstance(raw, Mapping):
        raise EvidenceValidationError(f"event {episode_id}:{raw_index} is not an object")
    _reject_hidden_topology(raw)
    action_kind = str(raw.get("action_kind") or "").strip()
    simulator_action = str(raw.get("simulator_action") or raw.get("action") or "").strip()
    if not action_kind or not simulator_action:
        raise EvidenceValidationError(f"event {episode_id}:{raw_index} lacks action semantics")
    step = _integer(raw.get("step", raw.get("trajectory_step", raw_index)), "trajectory step")
    rl_action = raw.get("rl_action_index")
    if mode is ReportMode.EVIDENCE and rl_action is not None:
        raise EvidenceValidationError("evidence_report cannot claim an RL action index")
    rl_action_index = None if rl_action is None else _integer(rl_action, "RL action index")
    if rl_action_index is not None and rl_action_index < 0:
        raise EvidenceValidationError("RL action index cannot be negative")
    success = True if progress_contract and "success" not in raw else bool(raw.get("success"))
    target = _target(raw.get("target_entity", raw.get("target")))
    cve_id = raw.get("cve_id")
    if cve_id is not None:
        cve_id = str(cve_id)
        if not _CVE.fullmatch(cve_id):
            raise EvidenceValidationError(f"invalid recorded CVE ID: {cve_id}")
    cvss = raw.get("cvss_score", raw.get("cvss_base"))
    cvss_score = None if cvss is None else float(cvss)
    if cvss_score is not None and not 0.0 <= cvss_score <= 10.0:
        raise EvidenceValidationError(f"invalid recorded CVSS score: {cvss_score}")
    ai_behavior = raw.get("ai_behavior")
    if ai_behavior is not None:
        try:
            AIBehavior(str(ai_behavior))
        except ValueError as exc:
            raise EvidenceValidationError(
                f"unsupported recorded AI behavior: {ai_behavior}"
            ) from exc
    mappings = map_simulator_behavior(
        action_kind,
        ai_behavior=ai_behavior,
    )
    _validate_recorded_mapping(raw, mappings)
    knowledge_delta = _knowledge_delta(raw)
    objective = bool(raw.get("goal_reached")) or bool(raw.get("is_crown_jewel"))
    objective = objective and goal_reached
    reward = raw.get("reward", raw.get("policy_reward"))
    if mode is ReportMode.EVIDENCE and reward is not None:
        raise EvidenceValidationError("evidence_report cannot contain simulator reward")
    reward_value = None if reward is None else float(reward)
    prerequisites = tuple(map(str, raw.get("prerequisites") or ()))
    outcomes = tuple(map(str, raw.get("outcomes") or ()))
    event_payload = {
        "step": step,
        "action_kind": action_kind,
        "simulator_action": simulator_action,
        "target": target,
        "cve_id": cve_id,
        "cvss_score": cvss_score,
        "success": success,
        "state_changed": bool(raw.get("state_changed", progress_contract)),
        "goal_reached": objective,
        "prerequisites": list(prerequisites),
        "outcomes": list(outcomes),
    }
    event_hash = canonical_sha256(event_payload)
    mapped = list(mappings) or [None]
    facts = [
        ReportFact(
            episode_id=episode_id,
            run_id=run_id,
            framework=mapping.framework.value if mapping else None,
            tactic_id=mapping.tactic_id if mapping else None,
            tactic_name=mapping.tactic_name if mapping else None,
            technique_id=mapping.technique_id if mapping else None,
            technique_name=mapping.technique_name if mapping else None,
            trajectory_step=step,
            rl_action_index=rl_action_index,
            simulator_action=simulator_action,
            action_kind=action_kind,
            target=target,
            cve_id=cve_id,
            cvss_score=cvss_score,
            success=success,
            access_gained=raw.get("access_gained"),
            crown_jewel_objective=objective,
            state_changed=bool(raw.get("state_changed", progress_contract)),
            knowledge_delta=knowledge_delta,
            prerequisites=prerequisites,
            outcomes=outcomes,
            reward=reward_value,
            source_event_sha256=event_hash,
        )
        for mapping in mapped
    ]
    return event_payload, facts


def _validate_recorded_mapping(raw: Mapping[str, Any], expected: tuple[Any, ...]) -> None:
    expected_ids = {(item.framework.value, item.technique_id) for item in expected}
    recorded = raw.get("framework_mappings")
    if recorded:
        if not isinstance(recorded, list):
            raise EvidenceValidationError("recorded framework mappings must be a list")
        recorded_ids = {
            (str(item.get("framework")), str(item.get("technique_id")))
            for item in recorded
            if isinstance(item, Mapping)
        }
        if recorded_ids != expected_ids:
            raise EvidenceValidationError(
                "recorded MITRE mapping bypasses or drifts from catalogue"
            )
    legacy_id = raw.get("technique_id")
    if legacy_id and str(legacy_id) not in {item[1] for item in expected_ids}:
        raise EvidenceValidationError("legacy technique ID drifts from the MITRE catalogue")


def _knowledge_delta(raw: Mapping[str, Any]) -> dict[str, Any] | None:
    value = raw.get("knowledge_delta")
    if value is not None:
        if not isinstance(value, Mapping):
            raise EvidenceValidationError("knowledge delta must be an object")
        _reject_hidden_topology(value)
        return dict(value)
    if "newly_discovered" in raw:
        return {"newly_discovered": _integer(raw.get("newly_discovered", 0), "discovery delta")}
    return None


def _phases(facts: list[ReportFact]) -> tuple[AttackPhase, ...]:
    phases: list[AttackPhase] = []
    current: list[int] = []
    current_key: tuple[str, str | None, str | None] | None = None
    for index, fact in enumerate(facts):
        key = (fact.episode_id, fact.framework, fact.tactic_id)
        if current and key != current_key:
            phases.append(_phase(facts, current))
            current = []
        current_key = key
        current.append(index)
    if current:
        phases.append(_phase(facts, current))
    return tuple(phases)


def _phase(facts: list[ReportFact], indexes: list[int]) -> AttackPhase:
    selected = [facts[index] for index in indexes]
    first = selected[0]
    return AttackPhase(
        episode_id=first.episode_id,
        framework=first.framework,
        tactic_id=first.tactic_id,
        tactic_name=first.tactic_name or "Unmapped recorded behavior",
        first_step=min(item.trajectory_step for item in selected),
        last_step=max(item.trajectory_step for item in selected),
        fact_indexes=tuple(indexes),
    )


def _criticality(
    facts: list[ReportFact],
    episode_events: dict[str, list[dict[str, Any]]],
    successful: set[str],
) -> tuple[ObservedPathCriticality, ...]:
    containing: dict[str, set[str]] = defaultdict(set)
    critical: dict[str, set[str]] = defaultdict(set)
    scores: dict[str, set[float]] = defaultdict(set)
    targets: dict[str, set[str]] = defaultdict(set)
    mapping_fact: dict[str, ReportFact] = {}
    for fact in facts:
        if fact.episode_id not in successful or not fact.cve_id:
            continue
        containing[fact.cve_id].add(fact.episode_id)
        if fact.cvss_score is not None:
            scores[fact.cve_id].add(fact.cvss_score)
        if fact.target is not None:
            targets[fact.cve_id].add(_display_target(fact.target))
        if fact.framework == "attack-enterprise" or fact.cve_id not in mapping_fact:
            mapping_fact[fact.cve_id] = fact
    for episode_id in successful:
        events = episode_events.get(episode_id, [])
        causal = reconstruct_attack_path(events)
        for event in causal:
            cve_id = event.get("cve_id")
            if cve_id:
                critical[str(cve_id)].add(episode_id)
    denominator = len(successful)
    output = []
    for cve_id in sorted(containing):
        broken = len(critical.get(cve_id, set()))
        percentage = 100.0 * broken / denominator if denominator else 0.0
        mapped = mapping_fact[cve_id]
        score_values = scores.get(cve_id, set())
        score = max(score_values) if score_values else None
        output.append(
            ObservedPathCriticality(
                cve_id=cve_id,
                successful_trajectories_containing=len(containing[cve_id]),
                observed_successful_paths_broken=broken,
                observed_path_criticality_percentage=percentage,
                cvss_score=score,
                framework=mapped.framework,
                tactic_id=mapped.tactic_id,
                tactic_name=mapped.tactic_name,
                technique_id=mapped.technique_id,
                technique_name=mapped.technique_name,
                observed_targets=tuple(sorted(targets.get(cve_id, set()))),
                statement=(
                    f"{cve_id} was path-critical in {percentage:.1f}% of observed "
                    "successful trajectories."
                ),
            )
        )
    return tuple(
        sorted(
            output,
            key=lambda item: (
                -item.observed_path_criticality_percentage,
                -item.successful_trajectories_containing,
                item.cve_id,
            ),
        )
    )


def _cross_episode(
    facts: list[ReportFact],
    episode_events: dict[str, list[dict[str, Any]]],
    successful: set[str],
    *,
    failed_action_observation_complete: bool,
) -> dict[str, Any]:
    tactic_events: Counter[tuple[str, str, str]] = Counter()
    technique_events: Counter[tuple[str, str, str]] = Counter()
    for fact in facts:
        if fact.framework and fact.tactic_id:
            tactic_events[(fact.framework, fact.tactic_id, fact.tactic_name or "")] += 1
        if fact.framework and fact.technique_id:
            technique_events[(fact.framework, fact.technique_id, fact.technique_name or "")] += 1
    lengths = [len(events) for events in episode_events.values()]
    signatures = {
        canonical_sha256(
            [
                {
                    "action": event["action_kind"],
                    "target": event["target"],
                    "cve_id": event["cve_id"],
                    "goal_reached": event["goal_reached"],
                }
                for event in events
            ]
        )
        for events in episode_events.values()
    }
    recurrence: Counter[str] = Counter()
    for events in episode_events.values():
        recurrence.update({str(event["cve_id"]) for event in events if event.get("cve_id")})
    total = len(episode_events)
    failed = sum(not event["success"] for events in episode_events.values() for event in events)
    return {
        "trajectory_count": total,
        "successful_trajectory_count": len(successful),
        "tactic_coverage": [
            {"framework": key[0], "tactic_id": key[1], "tactic_name": key[2], "events": count}
            for key, count in sorted(tactic_events.items())
        ],
        "technique_coverage": [
            {
                "framework": key[0],
                "technique_id": key[1],
                "technique_name": key[2],
                "events": count,
            }
            for key, count in sorted(technique_events.items())
        ],
        "most_frequent_techniques": [
            {
                "framework": key[0],
                "technique_id": key[1],
                "technique_name": key[2],
                "events": count,
            }
            for key, count in sorted(
                technique_events.items(), key=lambda item: (-item[1], item[0])
            )
        ],
        "path_length": {
            "minimum": min(lengths),
            "maximum": max(lengths),
            "mean": sum(lengths) / len(lengths),
        },
        "path_diversity": {
            "unique_recorded_paths": len(signatures),
            "proportion_unique": len(signatures) / total,
        },
        "vulnerability_recurrence": [
            {"cve_id": cve_id, "trajectories": count}
            for cve_id, count in sorted(recurrence.items(), key=lambda item: (-item[1], item[0]))
        ],
        "crown_jewel_reach_rate": len(successful) / total,
        "failed_action_count": failed if failed_action_observation_complete else None,
        "failed_action_observation": (
            "complete_recorded_events"
            if failed_action_observation_complete
            else "unavailable_in_progress_only_source"
        ),
    }


def _observed_graph(facts: list[ReportFact]) -> dict[str, list[dict[str, Any]]]:
    nodes: dict[str, dict[str, Any]] = {}
    for fact in facts:
        if fact.target is None:
            continue
        target = _display_target(fact.target)
        node_id = "observed:" + canonical_sha256(fact.target)[:16]
        nodes[node_id] = {
            "id": node_id,
            "type": "observed_target",
            "name": target,
            "attributes": {"source": "recorded_event"},
        }
    return {"nodes": [nodes[key] for key in sorted(nodes)], "edges": []}


def _validate_checkpoint_hashes(
    values: Mapping[str, str], run_ids: set[str]
) -> dict[str, str]:
    output = {}
    for run_id, digest in values.items():
        if run_id not in run_ids:
            continue
        _require_sha256(str(digest), f"checkpoint hash for {run_id}")
        output[str(run_id)] = str(digest)
    return dict(sorted(output.items()))


def _target(value: Any) -> str | list[int] | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, list | tuple) and all(
        isinstance(item, int) and not isinstance(item, bool) for item in value
    ):
        return list(value)
    raise EvidenceValidationError(f"unsupported recorded target: {value!r}")


def _display_target(value: str | list[int]) -> str:
    return value if isinstance(value, str) else "(" + ", ".join(map(str, value)) + ")"


def _integer(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise EvidenceValidationError(f"{label} must be an integer")
    try:
        integer = int(value)
    except (TypeError, ValueError) as exc:
        raise EvidenceValidationError(f"{label} must be an integer") from exc
    return integer


def _require_sha256(value: str, label: str) -> None:
    if not re.fullmatch(r"[0-9a-f]{64}", value):
        raise EvidenceValidationError(f"{label} must be a lowercase SHA-256 digest")


def _reject_hidden_topology(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if str(key).casefold() in _FORBIDDEN_TOPOLOGY_KEYS:
                raise EvidenceValidationError(f"hidden topology field is forbidden: {key}")
            _reject_hidden_topology(item)
    elif isinstance(value, list | tuple):
        for item in value:
            _reject_hidden_topology(item)
