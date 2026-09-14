"""Evidence-only causal attack and knowledge-flow graph derivation."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any

CAUSAL_GRAPH_SCHEMA_VERSION = "security-rl-causal-attack-path-v1"
UNKNOWN = "unknown"
FORBIDDEN_KEYS = frozenset(
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


class CausalGraphError(ValueError):
    """Recorded evidence cannot support a safe causal graph."""


class RelationshipType(StrEnum):
    DISCOVERED = "DISCOVERED"
    REVEALED = "REVEALED"
    YIELDED_CREDENTIAL = "YIELDED_CREDENTIAL"
    GRANTS_ACCESS = "GRANTS_ACCESS"
    EXPLOITED = "EXPLOITED"
    AUTHENTICATED_TO = "AUTHENTICATED_TO"
    PIVOTED_TO = "PIVOTED_TO"
    CONNECTED_TO = "CONNECTED_TO"
    HOSTS = "HOSTS"
    EXPOSES = "EXPOSES"
    CONTAINS = "CONTAINS"


@dataclass(frozen=True, slots=True)
class KnowledgeEvidence:
    evidence_id: str
    report_id: str
    episode_id: str
    step: int
    source_event_sha256: str
    evidence_kind: str
    payload: dict[str, Any]


@dataclass(frozen=True, slots=True)
class CausalEdge:
    edge_id: str
    source_node: str
    target_node: str
    relationship_type: str
    simulator_action: str
    discovery_method: str
    service: str
    port: str
    protocol: str
    product: str
    version: str
    vulnerability_id: str
    cve_id: str
    credential_id: str
    identity: str
    access_before: str
    access_after: str
    success: bool
    mitre_mappings: tuple[dict[str, str], ...]
    episode_id: str
    step: int
    evidence_steps: tuple[int, ...]
    evidence_ids: tuple[str, ...]
    new_information: tuple[str, ...]
    explanation: str
    categories: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CausalNode:
    node_id: str
    node_type: str
    first_discovery_step: int | str
    discovered_by: str
    discovered_services: tuple[dict[str, str], ...]
    discovered_vulnerabilities: tuple[str, ...]
    current_access_level: str
    credentials: tuple[str, ...]
    mitre_techniques: tuple[str, ...]
    incoming_relationships: tuple[str, ...]
    outgoing_relationships: tuple[str, ...]
    evidence_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CausalGraph:
    graph_id: str
    report_id: str
    schema_version: str
    nodes: tuple[CausalNode, ...]
    edges: tuple[CausalEdge, ...]
    evidence: tuple[KnowledgeEvidence, ...]
    provenance: dict[str, Any]

    def __post_init__(self) -> None:
        evidence_ids = {item.evidence_id for item in self.evidence}
        for edge in self.edges:
            if not edge.evidence_ids:
                raise CausalGraphError(f"edge has no source evidence: {edge.edge_id}")
            if not set(edge.evidence_ids) <= evidence_ids:
                raise CausalGraphError(f"edge references unknown evidence: {edge.edge_id}")
            try:
                RelationshipType(edge.relationship_type)
            except ValueError as exc:
                raise CausalGraphError(
                    f"edge has unsupported relationship: {edge.relationship_type}"
                ) from exc

    def to_dict(self) -> dict[str, Any]:
        return json.loads(json.dumps(asdict(self), sort_keys=True, allow_nan=False))

    def why_actionable(self, node_id: str) -> dict[str, Any]:
        """Reconstruct availability from persisted evidence-linked edges only."""
        node = next((item for item in self.nodes if item.node_id == node_id), None)
        if node is None:
            raise KeyError(f"unknown causal graph node: {node_id}")
        incoming = sorted(
            (edge for edge in self.edges if edge.target_node == node_id),
            key=lambda edge: (edge.episode_id, edge.step, edge.edge_id),
        )
        causal = [
            edge
            for edge in incoming
            if edge.relationship_type
            in {
                RelationshipType.DISCOVERED,
                RelationshipType.REVEALED,
                RelationshipType.YIELDED_CREDENTIAL,
                RelationshipType.AUTHENTICATED_TO,
                RelationshipType.EXPLOITED,
                RelationshipType.GRANTS_ACCESS,
                RelationshipType.PIVOTED_TO,
                RelationshipType.CONNECTED_TO,
            }
        ]
        access_edges = [
            edge
            for edge in causal
            if edge.success
            and edge.relationship_type
            in {
                RelationshipType.AUTHENTICATED_TO,
                RelationshipType.EXPLOITED,
                RelationshipType.GRANTS_ACCESS,
                RelationshipType.PIVOTED_TO,
                RelationshipType.CONNECTED_TO,
            }
        ]
        if access_edges:
            terminal = access_edges[0]
            causal = [
                edge
                for edge in causal
                if edge.episode_id == terminal.episode_id and edge.step <= terminal.step
            ]
        elif causal:
            causal = [causal[0]]
        if not causal:
            return {
                "node_id": node_id,
                "actionable": False,
                "explanation": f"No recorded evidence explains why {node_id} became actionable.",
                "edge_ids": [],
                "evidence_ids": [],
            }
        clauses = []
        for edge in causal:
            source = (
                "an unknown recorded source"
                if edge.source_node == UNKNOWN
                else edge.source_node
            )
            relation = edge.relationship_type.replace("_", " ").lower()
            detail = f"{source} {relation} {node_id} using {edge.simulator_action}"
            if edge.credential_id != UNKNOWN:
                detail += f" with credential {edge.credential_id}"
            elif edge.identity != UNKNOWN:
                detail += f" as identity {edge.identity}"
            elif edge.vulnerability_id != UNKNOWN:
                detail += f" through {edge.vulnerability_id}"
            if edge.service != UNKNOWN:
                endpoint = edge.service
                if edge.port != UNKNOWN:
                    endpoint += f" on port {edge.port}"
                detail += f" via {endpoint}"
            if edge.new_information:
                detail += "; this revealed " + ", ".join(edge.new_information)
            clauses.append(detail)
        return {
            "node_id": node_id,
            "actionable": any(edge.success for edge in causal),
            "explanation": f"{node_id} became actionable because " + "; then ".join(clauses) + ".",
            "edge_ids": [edge.edge_id for edge in causal],
            "evidence_ids": sorted({item for edge in causal for item in edge.evidence_ids}),
        }


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


def build_causal_graph(
    report: Mapping[str, Any], trajectory_document: Any | None = None
) -> CausalGraph:
    """Derive a causal graph without accepting or consulting topology state."""
    _reject_hidden(report)
    if trajectory_document is not None:
        _reject_hidden(trajectory_document)
    report_id = str(report.get("report_id") or UNKNOWN)
    facts = list(report.get("facts") or ())
    raw_by_step = _raw_event_index(trajectory_document)
    grouped: dict[tuple[str, int, str], list[Mapping[str, Any]]] = defaultdict(list)
    for fact in facts:
        if not isinstance(fact, Mapping):
            raise CausalGraphError("report facts must be objects")
        key = (
            str(fact.get("episode_id") or UNKNOWN),
            int(fact.get("trajectory_step", 0)),
            str(fact.get("source_event_sha256") or UNKNOWN),
        )
        grouped[key].append(fact)

    evidence: list[KnowledgeEvidence] = []
    edges: list[CausalEdge] = []
    node_state: dict[str, dict[str, Any]] = {}
    node_types: dict[str, str] = {}
    access_state: dict[tuple[str, str], tuple[str, str]] = {}
    for key in sorted(grouped):
        episode_id, step, source_hash = key
        event_facts = grouped[key]
        fact = event_facts[0]
        raw = raw_by_step.get((episode_id, step), {})
        merged = dict(fact) | dict(raw)
        target_node = _node(merged.get("target_entity", merged.get("target")))
        source_node = _first_known(merged, "source_node", "source", "origin_node")
        if target_node != UNKNOWN and (
            merged.get("node_type") or merged.get("target_node_type")
        ):
            node_types[target_node] = _known(
                merged.get("target_node_type", merged.get("node_type"))
            )
        if source_node != UNKNOWN and merged.get("source_node_type"):
            node_types[source_node] = _known(merged.get("source_node_type"))
        delta = merged.get("knowledge_delta") or fact.get("knowledge_delta") or {}
        if not isinstance(delta, Mapping):
            raise CausalGraphError("knowledge_delta must be an object")
        evidence_payload = {
            "episode_id": episode_id,
            "step": step,
            "source_event_sha256": source_hash,
            "simulator_action": _known(merged.get("simulator_action") or merged.get("action")),
            "knowledge_delta": dict(delta),
            "prerequisites": list(merged.get("prerequisites") or ()),
            "outcomes": list(merged.get("outcomes") or ()),
        }
        evidence_id = canonical_sha256(evidence_payload)
        evidence.append(
            KnowledgeEvidence(
                evidence_id=evidence_id,
                report_id=report_id,
                episode_id=episode_id,
                step=step,
                source_event_sha256=source_hash,
                evidence_kind="trajectory_and_agent_knowledge_delta",
                payload=evidence_payload,
            )
        )
        mappings = tuple(
            sorted(
                (
                    {
                        "framework": str(item.get("framework") or UNKNOWN),
                        "tactic_id": str(item.get("tactic_id") or UNKNOWN),
                        "tactic_name": str(item.get("tactic_name") or UNKNOWN),
                        "technique_id": str(item.get("technique_id") or UNKNOWN),
                        "technique_name": str(item.get("technique_name") or UNKNOWN),
                    }
                    for item in event_facts
                ),
                key=lambda item: (item["framework"], item["technique_id"]),
            )
        )
        edges.extend(
            _event_edges(
                merged,
                delta,
                mappings,
                episode_id,
                step,
                evidence_id,
                access_state,
            )
        )

    deduped = {edge.edge_id: edge for edge in edges}
    ordered_edges = tuple(sorted(deduped.values(), key=lambda edge: edge.edge_id))
    for edge in sorted(
        ordered_edges, key=lambda item: (item.episode_id, item.step, item.edge_id)
    ):
        _accumulate_nodes(node_state, edge)
    for node_id, node_type in node_types.items():
        if node_id in node_state:
            node_state[node_id]["node_type"] = node_type
    nodes = tuple(_finish_node(node_id, value) for node_id, value in sorted(node_state.items()))
    body = {
        "report_id": report_id,
        "schema_version": CAUSAL_GRAPH_SCHEMA_VERSION,
        "nodes": [asdict(node) for node in nodes],
        "edges": [asdict(edge) for edge in ordered_edges],
        "evidence": [asdict(item) for item in evidence],
    }
    graph_id = canonical_sha256(body)
    return CausalGraph(
        graph_id=graph_id,
        report_id=report_id,
        schema_version=CAUSAL_GRAPH_SCHEMA_VERSION,
        nodes=nodes,
        edges=ordered_edges,
        evidence=tuple(evidence),
        provenance={
            "source": "recorded_phase14_facts_and_agentknowledge_deltas_only",
            "hidden_topology_used": False,
            "unknown_value_policy": "literal_unknown_no_inference",
        },
    )


def why_actionable(graph: CausalGraph | Mapping[str, Any], node_id: str) -> dict[str, Any]:
    if isinstance(graph, CausalGraph):
        return graph.why_actionable(node_id)
    return graph_from_dict(graph).why_actionable(node_id)


def graph_from_dict(value: Mapping[str, Any]) -> CausalGraph:
    return CausalGraph(
        graph_id=str(value["graph_id"]),
        report_id=str(value["report_id"]),
        schema_version=str(value["schema_version"]),
        nodes=tuple(CausalNode(**item) for item in value.get("nodes", ())),
        edges=tuple(
            CausalEdge(
                **dict(item)
                | {
                    "mitre_mappings": tuple(item.get("mitre_mappings", ())),
                    "evidence_steps": tuple(item.get("evidence_steps", ())),
                    "evidence_ids": tuple(item.get("evidence_ids", ())),
                    "new_information": tuple(item.get("new_information", ())),
                    "categories": tuple(item.get("categories", ())),
                }
            )
            for item in value.get("edges", ())
        ),
        evidence=tuple(KnowledgeEvidence(**item) for item in value.get("evidence", ())),
        provenance=dict(value.get("provenance") or {}),
    )


def _event_edges(
    event: Mapping[str, Any],
    delta: Mapping[str, Any],
    mappings: tuple[dict[str, str], ...],
    episode_id: str,
    step: int,
    evidence_id: str,
    access_state: dict[tuple[str, str], tuple[str, str]],
) -> list[CausalEdge]:
    target = _node(event.get("target_entity", event.get("target")))
    source = _first_known(
        event,
        "source_node",
        "source",
        "origin_node",
        "discovered_by",
        "revealed_by",
        "pivot_source",
    )
    behavior = str(event.get("action_kind") or UNKNOWN)
    relationship = _relationship(behavior, event.get("relationship_type"))
    output: list[CausalEdge] = []
    if target != UNKNOWN:
        output.append(
            _make_edge(
                source,
                target,
                relationship,
                event,
                delta,
                mappings,
                episode_id,
                step,
                evidence_id,
                access_state,
            )
        )
    for discovered_node in delta.get("discovered_nodes", ()):
        discovered = _node(discovered_node)
        if discovered != target:
            output.append(
                _make_edge(
                    source,
                    discovered,
                    RelationshipType.DISCOVERED,
                    event,
                    delta,
                    mappings,
                    episode_id,
                    step,
                    evidence_id,
                    access_state,
                )
            )
    for observed in delta.get("observed_edges", ()):
        if isinstance(observed, Sequence) and not isinstance(observed, str) and len(observed) >= 2:
            native_type = observed[2] if len(observed) > 2 else None
            output.append(
                _make_edge(
                    _node(observed[0]),
                    _node(observed[1]),
                    _relationship(behavior, native_type),
                    event,
                    delta,
                    mappings,
                    episode_id,
                    step,
                    evidence_id,
                    access_state,
                )
            )
    for node, service in delta.get("discovered_services", ()):
        host = _node(node)
        service_name = _known(service)
        service_node = f"service:{host}:{service_name}"
        service_event = dict(event) | {"service": service_name}
        output.append(
            _make_edge(
                host,
                service_node,
                RelationshipType.EXPOSES,
                service_event,
                delta,
                mappings,
                episode_id,
                step,
                evidence_id,
                access_state,
            )
        )
    for credential_target in delta.get("credential_targets", ()):
        output.append(
            _make_edge(
                target if target != UNKNOWN else source,
                _node(credential_target),
                RelationshipType.YIELDED_CREDENTIAL,
                event,
                delta,
                mappings,
                episode_id,
                step,
                evidence_id,
                access_state,
            )
        )
    for changed_node, _changed_access in delta.get("access_changes", ()):
        changed = _node(changed_node)
        if changed != target:
            output.append(
                _make_edge(
                    source,
                    changed,
                    RelationshipType.GRANTS_ACCESS,
                    event,
                    delta,
                    mappings,
                    episode_id,
                    step,
                    evidence_id,
                    access_state,
                )
            )
    return output


def _make_edge(
    source: str,
    target: str,
    relationship: RelationshipType,
    event: Mapping[str, Any],
    delta: Mapping[str, Any],
    mappings: tuple[dict[str, str], ...],
    episode_id: str,
    step: int,
    evidence_id: str,
    access_state: dict[tuple[str, str], tuple[str, str]],
) -> CausalEdge:
    state_key = (episode_id, target)
    prior_access, prior_evidence = access_state.get(state_key, (UNKNOWN, UNKNOWN))
    access_before = _known(event.get("access_before"))
    if access_before == UNKNOWN and prior_access != UNKNOWN:
        access_before = prior_access
    access_after = _known(event.get("access_after", event.get("access_gained")))
    for changed_node, changed_access in delta.get("access_changes", ()):
        if _node(changed_node) == target:
            access_after = _known(changed_access)
    evidence_ids = [evidence_id]
    if prior_evidence != UNKNOWN and access_before != UNKNOWN:
        evidence_ids.append(prior_evidence)
    if access_after != UNKNOWN:
        access_state[state_key] = (access_after, evidence_id)
    service = _known(event.get("service"))
    if service == UNKNOWN:
        services = sorted(
            _known(item[1])
            for item in delta.get("discovered_services", ())
            if len(item) >= 2 and _node(item[0]) == target
        )
        if services:
            service = services[0]
    port = _known(event.get("port"))
    protocol = _known(event.get("protocol"))
    product = _known(event.get("product"))
    version = _known(event.get("version"))
    cve_id = _known(event.get("cve_id"))
    native_vulnerability = _known(event.get("simulator_vulnerability_id"))
    vulnerability = cve_id if cve_id != UNKNOWN else native_vulnerability
    credential = _known(event.get("credential_id", event.get("credential")))
    identity = _known(event.get("identity", event.get("username")))
    action = _known(event.get("simulator_action", event.get("action")))
    information = tuple(sorted(_new_information(delta)))
    categories = _categories(relationship, vulnerability, credential, mappings)
    payload = {
        "source": source,
        "target": target,
        "relationship": str(relationship),
        "episode_id": episode_id,
        "step": step,
        "evidence_ids": sorted(evidence_ids),
    }
    edge_id = canonical_sha256(payload)
    explanation = (
        f"{source} {str(relationship).replace('_', ' ').lower()} {target} at step {step} "
        f"using {action}; evidence {evidence_id}."
    )
    return CausalEdge(
        edge_id=edge_id,
        source_node=source,
        target_node=target,
        relationship_type=str(relationship),
        simulator_action=action,
        discovery_method=_known(event.get("discovery_method", event.get("action_kind"))),
        service=service,
        port=port,
        protocol=protocol,
        product=product,
        version=version,
        vulnerability_id=vulnerability,
        cve_id=cve_id,
        credential_id=credential,
        identity=identity,
        access_before=access_before,
        access_after=access_after,
        success=bool(event.get("success", False)),
        mitre_mappings=mappings,
        episode_id=episode_id,
        step=step,
        evidence_steps=(step,),
        evidence_ids=tuple(sorted(evidence_ids)),
        new_information=information,
        explanation=explanation,
        categories=categories,
    )


def _accumulate_nodes(state: dict[str, dict[str, Any]], edge: CausalEdge) -> None:
    for node_id in (edge.source_node, edge.target_node):
        value = state.setdefault(
            node_id,
            {
                "node_type": "unknown",
                "first": UNKNOWN,
                "discovered_by": UNKNOWN,
                "services": {},
                "vulnerabilities": set(),
                "access": UNKNOWN,
                "credentials": set(),
                "techniques": set(),
                "incoming": set(),
                "outgoing": set(),
                "evidence": set(),
            },
        )
        value["evidence"].update(edge.evidence_ids)
        value["techniques"].update(
            mapping["technique_id"]
            for mapping in edge.mitre_mappings
            if mapping["technique_id"] != UNKNOWN
        )
    source, target = state[edge.source_node], state[edge.target_node]
    source["outgoing"].add(edge.edge_id)
    target["incoming"].add(edge.edge_id)
    if target["first"] == UNKNOWN or int(target["first"]) > edge.step:
        target["first"] = edge.step
        target["discovered_by"] = edge.source_node
    if edge.relationship_type == RelationshipType.EXPOSES:
        target["node_type"] = "service"
        source["services"][edge.service] = {
            "service": edge.service,
            "port": edge.port,
            "protocol": edge.protocol,
            "product": edge.product,
            "version": edge.version,
        }
    if edge.vulnerability_id != UNKNOWN:
        target["vulnerabilities"].add(edge.vulnerability_id)
    if edge.access_after != UNKNOWN:
        if _access_rank(edge.access_after) >= _access_rank(target["access"]):
            target["access"] = edge.access_after
    if edge.credential_id != UNKNOWN:
        target["credentials"].add(edge.credential_id)
    if edge.identity != UNKNOWN:
        target["credentials"].add(f"identity:{edge.identity}")


def _finish_node(node_id: str, value: dict[str, Any]) -> CausalNode:
    return CausalNode(
        node_id=node_id,
        node_type=value["node_type"],
        first_discovery_step=value["first"],
        discovered_by=value["discovered_by"],
        discovered_services=tuple(value["services"][key] for key in sorted(value["services"])),
        discovered_vulnerabilities=tuple(sorted(value["vulnerabilities"])),
        current_access_level=value["access"],
        credentials=tuple(sorted(value["credentials"])),
        mitre_techniques=tuple(sorted(value["techniques"])),
        incoming_relationships=tuple(sorted(value["incoming"])),
        outgoing_relationships=tuple(sorted(value["outgoing"])),
        evidence_ids=tuple(sorted(value["evidence"])),
    )


def _raw_event_index(document: Any | None) -> dict[tuple[str, int], Mapping[str, Any]]:
    if document is None:
        return {}
    trajectories = document if isinstance(document, list) else document.get("trajectories", [])
    output = {}
    for index, trajectory in enumerate(trajectories):
        run_id = str(trajectory.get("run_name") or trajectory.get("run_id") or UNKNOWN)
        suffix = trajectory.get("evaluation_seed", trajectory.get("episode_id", index))
        episode_id = f"{run_id}:{suffix}"
        events = trajectory.get(
            "events",
            trajectory.get(
                "steps",
                trajectory.get("progress_steps", trajectory.get("trajectory", ())),
            ),
        )
        for event_index, event in enumerate(events):
            step = int(event.get("step", event.get("trajectory_step", event_index)))
            output[(episode_id, step)] = event
    return output


def _relationship(behavior: str, explicit: Any) -> RelationshipType:
    if explicit is not None:
        candidate = str(explicit).upper()
        try:
            return RelationshipType(candidate)
        except ValueError:
            pass
    value = str(behavior).casefold()
    if "auth" in value or "login" in value or "credential" in value:
        return RelationshipType.AUTHENTICATED_TO
    if "exploit" in value:
        return RelationshipType.EXPLOITED
    if "pivot" in value or "lateral" in value:
        return RelationshipType.PIVOTED_TO
    if "connect" in value:
        return RelationshipType.CONNECTED_TO
    if "discover" in value:
        return RelationshipType.DISCOVERED
    if "access" in value or "priv" in value:
        return RelationshipType.GRANTS_ACCESS
    return RelationshipType.REVEALED


def _categories(
    relationship: RelationshipType,
    vulnerability: str,
    credential: str,
    mappings: tuple[dict[str, str], ...],
) -> tuple[str, ...]:
    values = {"attack_path"}
    if relationship in {
        RelationshipType.DISCOVERED,
        RelationshipType.REVEALED,
        RelationshipType.YIELDED_CREDENTIAL,
        RelationshipType.EXPOSES,
    }:
        values.add("knowledge_flow")
    if vulnerability != UNKNOWN:
        values.add("vulnerabilities")
    if credential != UNKNOWN or relationship == RelationshipType.AUTHENTICATED_TO:
        values.add("credentials")
    if any(mapping["technique_id"] != UNKNOWN for mapping in mappings):
        values.add("mitre")
    return tuple(sorted(values))


def _new_information(delta: Mapping[str, Any]) -> set[str]:
    values = set()
    for key, items in delta.items():
        sequence = items if isinstance(items, list | tuple | set) else (items,)
        for item in sequence:
            if item is None or item is False or item == "" or item == 0:
                continue
            encoded = (
                json.dumps(item, sort_keys=True)
                if isinstance(item, dict | list)
                else str(item)
            )
            values.add(f"{key}:{encoded}")
    return values


def _known(value: Any) -> str:
    if value is None or value == "":
        return UNKNOWN
    return str(value)


def _access_rank(value: str) -> int:
    return {
        UNKNOWN: -1,
        "none": 0,
        "0": 0,
        "read": 1,
        "user": 1,
        "1": 1,
        "root": 2,
        "admin": 2,
        "administrator": 2,
        "2": 2,
    }.get(str(value).casefold(), 0)


def _node(value: Any) -> str:
    if value is None or value == "":
        return UNKNOWN
    if isinstance(value, list | tuple):
        return "(" + ", ".join(map(str, value)) + ")"
    return str(value)


def _first_known(value: Mapping[str, Any], *keys: str) -> str:
    for key in keys:
        if value.get(key) not in (None, ""):
            return _node(value[key])
    return UNKNOWN


def _reject_hidden(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if str(key).casefold() in FORBIDDEN_KEYS:
                raise CausalGraphError(f"hidden topology field is forbidden: {key}")
            _reject_hidden(item)
    elif isinstance(value, list | tuple):
        for item in value:
            _reject_hidden(item)


__all__ = [
    "CAUSAL_GRAPH_SCHEMA_VERSION",
    "UNKNOWN",
    "CausalEdge",
    "CausalGraph",
    "CausalGraphError",
    "CausalNode",
    "KnowledgeEvidence",
    "RelationshipType",
    "build_causal_graph",
    "graph_from_dict",
    "why_actionable",
]
