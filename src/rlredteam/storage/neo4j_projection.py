"""Rebuildable Neo4j projection of PostgreSQL-authoritative causal facts."""

from __future__ import annotations

import json
import os
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

try:  # Installed only in the dedicated Phase 20 Podman client image.
    from neo4j import GraphDatabase
    from neo4j.exceptions import ServiceUnavailable
except ModuleNotFoundError:  # pragma: no cover - exercised by the base-image tests
    GraphDatabase = None  # type: ignore[assignment]
    ServiceUnavailable = OSError  # type: ignore[misc,assignment]

from rlredteam.causal_graph import (
    CAUSAL_GRAPH_SCHEMA_VERSION,
    UNKNOWN,
    CausalGraph,
    RelationshipType,
    canonical_sha256,
)

NEO4J_PROJECTION_SCHEMA_VERSION = "security-rl-neo4j-projection-v1"


class Neo4jProjectionError(RuntimeError):
    """The derived projection is unsafe, incomplete or inconsistent."""


class MissingNeo4jCredentials(Neo4jProjectionError):
    """Required Neo4j connection configuration is absent."""


@dataclass(frozen=True, slots=True)
class Neo4jSettings:
    uri: str
    user: str
    password: str
    database: str = "neo4j"

    @classmethod
    def from_env(cls) -> Neo4jSettings:
        required = {
            "NEO4J_URI": os.environ.get("NEO4J_URI"),
            "NEO4J_USER": os.environ.get("NEO4J_USER"),
            "NEO4J_PASSWORD": os.environ.get("NEO4J_PASSWORD"),
        }
        missing = sorted(key for key, value in required.items() if not value)
        if missing:
            raise MissingNeo4jCredentials(
                "missing required Neo4j settings: " + ", ".join(missing)
            )
        return cls(
            uri=str(required["NEO4J_URI"]),
            user=str(required["NEO4J_USER"]),
            password=str(required["NEO4J_PASSWORD"]),
            database=os.environ.get("NEO4J_DATABASE") or "neo4j",
        )


@dataclass(frozen=True, slots=True)
class ProjectionPlan:
    graph_id: str
    report_id: str
    source_graph_sha256: str
    projection_hash: str
    graph_row: dict[str, Any]
    node_rows: tuple[dict[str, Any], ...]
    evidence_rows: tuple[dict[str, Any], ...]
    edge_rows: tuple[dict[str, Any], ...]
    technique_rows: tuple[dict[str, Any], ...]

    def counts(self) -> dict[str, int]:
        return {
            "nodes": len(self.node_rows),
            "edges": len(self.edge_rows),
            "evidence": len(self.evidence_rows),
            "techniques": len(self.technique_rows),
        }

    def identity_payload(self) -> dict[str, Any]:
        return {
            "schema_version": NEO4J_PROJECTION_SCHEMA_VERSION,
            "graph_id": self.graph_id,
            "report_id": self.report_id,
            "source_graph_sha256": self.source_graph_sha256,
            "nodes": list(self.node_rows),
            "evidence": list(self.evidence_rows),
            "edges": list(self.edge_rows),
            "techniques": list(self.technique_rows),
        }


def build_projection_plan(graph: CausalGraph) -> ProjectionPlan:
    """Flatten one validated Phase 19 graph into Neo4j-safe property rows."""
    if graph.schema_version != CAUSAL_GRAPH_SCHEMA_VERSION:
        raise Neo4jProjectionError(f"unsupported causal graph schema: {graph.schema_version}")
    if graph.provenance.get("hidden_topology_used") is not False:
        raise Neo4jProjectionError("projection requires explicit hidden_topology_used=false")
    evidence_ids = {item.evidence_id for item in graph.evidence}
    node_ids = {node.node_id for node in graph.nodes}
    if len(evidence_ids) != len(graph.evidence) or len(node_ids) != len(graph.nodes):
        raise Neo4jProjectionError("source graph contains duplicate identities")

    node_rows = tuple(
        {
            "projection_key": _key(graph.graph_id, "node", node.node_id),
            "graph_id": graph.graph_id,
            "node_id": node.node_id,
            "node_type": node.node_type,
            "first_discovery_step": str(node.first_discovery_step),
            "discovered_by": node.discovered_by,
            "discovered_services_json": _canonical(list(node.discovered_services)),
            "discovered_vulnerabilities": list(node.discovered_vulnerabilities),
            "current_access_level": node.current_access_level,
            "credentials": list(node.credentials),
            "mitre_techniques": list(node.mitre_techniques),
            "incoming_relationships": list(node.incoming_relationships),
            "outgoing_relationships": list(node.outgoing_relationships),
            "evidence_ids": list(node.evidence_ids),
        }
        for node in sorted(graph.nodes, key=lambda item: item.node_id)
    )
    evidence_rows = tuple(
        {
            "projection_key": _key(graph.graph_id, "evidence", item.evidence_id),
            "graph_id": graph.graph_id,
            "evidence_id": item.evidence_id,
            "report_id": item.report_id,
            "episode_id": item.episode_id,
            "step": item.step,
            "source_event_sha256": item.source_event_sha256,
            "evidence_kind": item.evidence_kind,
            "payload_json": _canonical(item.payload),
        }
        for item in sorted(graph.evidence, key=lambda value: value.evidence_id)
    )
    edge_rows_list: list[dict[str, Any]] = []
    techniques: dict[str, dict[str, Any]] = {}
    allowed = {item.value for item in RelationshipType}
    for edge in sorted(graph.edges, key=lambda item: item.edge_id):
        if edge.relationship_type not in allowed:
            raise Neo4jProjectionError(f"unsupported relationship: {edge.relationship_type}")
        if edge.source_node not in node_ids or edge.target_node not in node_ids:
            raise Neo4jProjectionError(f"edge endpoint absent from source graph: {edge.edge_id}")
        if not edge.evidence_ids or not set(edge.evidence_ids) <= evidence_ids:
            raise Neo4jProjectionError(f"edge evidence is incomplete: {edge.edge_id}")
        technique_ids: list[str] = []
        tactic_ids: list[str] = []
        for mapping in edge.mitre_mappings:
            technique_id = str(mapping.get("technique_id") or UNKNOWN)
            tactic_id = str(mapping.get("tactic_id") or UNKNOWN)
            if technique_id != UNKNOWN:
                technique_ids.append(technique_id)
                techniques[technique_id] = {
                    "projection_key": _key(graph.graph_id, "technique", technique_id),
                    "graph_id": graph.graph_id,
                    "technique_id": technique_id,
                    "technique_name": str(mapping.get("technique_name") or UNKNOWN),
                    "framework": str(mapping.get("framework") or UNKNOWN),
                }
            if tactic_id != UNKNOWN:
                tactic_ids.append(tactic_id)
        edge_rows_list.append(
            {
                "projection_key": _key(graph.graph_id, "edge", edge.edge_id),
                "graph_id": graph.graph_id,
                "edge_id": edge.edge_id,
                "source_key": _key(graph.graph_id, "node", edge.source_node),
                "target_key": _key(graph.graph_id, "node", edge.target_node),
                "source_node": edge.source_node,
                "target_node": edge.target_node,
                "relationship_type": edge.relationship_type,
                "simulator_action": edge.simulator_action,
                "discovery_method": edge.discovery_method,
                "service": edge.service,
                "port": edge.port,
                "protocol": edge.protocol,
                "product": edge.product,
                "version": edge.version,
                "vulnerability_id": edge.vulnerability_id,
                "cve_id": edge.cve_id,
                "credential_id": edge.credential_id,
                "identity": edge.identity,
                "access_before": edge.access_before,
                "access_after": edge.access_after,
                "success": edge.success,
                "episode_id": edge.episode_id,
                "step": edge.step,
                "evidence_steps": list(edge.evidence_steps),
                "evidence_ids": list(edge.evidence_ids),
                "evidence_keys": [
                    _key(graph.graph_id, "evidence", value) for value in edge.evidence_ids
                ],
                "new_information": list(edge.new_information),
                "explanation": edge.explanation,
                "categories": list(edge.categories),
                "mitre_technique_ids": sorted(set(technique_ids)),
                "mitre_tactic_ids": sorted(set(tactic_ids)),
                "mitre_mappings_json": _canonical(list(edge.mitre_mappings)),
            }
        )
    edge_rows = tuple(edge_rows_list)
    technique_rows = tuple(techniques[key] for key in sorted(techniques))
    source_hash = canonical_sha256(graph.to_dict())
    base = {
        "schema_version": NEO4J_PROJECTION_SCHEMA_VERSION,
        "graph_id": graph.graph_id,
        "report_id": graph.report_id,
        "source_graph_sha256": source_hash,
        "nodes": list(node_rows),
        "evidence": list(evidence_rows),
        "edges": list(edge_rows),
        "techniques": list(technique_rows),
    }
    projection_hash = canonical_sha256(base)
    graph_row = {
        "projection_key": _key(graph.graph_id, "graph", graph.graph_id),
        "graph_id": graph.graph_id,
        "report_id": graph.report_id,
        "source_graph_sha256": source_hash,
        "projection_hash": projection_hash,
        "schema_version": NEO4J_PROJECTION_SCHEMA_VERSION,
        "hidden_topology_used": False,
        **{f"{name}_count": count for name, count in {
            "node": len(node_rows),
            "edge": len(edge_rows),
            "evidence": len(evidence_rows),
            "technique": len(technique_rows),
        }.items()},
    }
    return ProjectionPlan(
        graph_id=graph.graph_id,
        report_id=graph.report_id,
        source_graph_sha256=source_hash,
        projection_hash=projection_hash,
        graph_row=graph_row,
        node_rows=node_rows,
        evidence_rows=evidence_rows,
        edge_rows=edge_rows,
        technique_rows=technique_rows,
    )


class Neo4jProjectionStore:
    """Write and query graph-scoped, disposable Neo4j projections."""

    def __init__(self, driver: Any, database: str = "neo4j") -> None:
        self.driver = driver
        self.database = database

    @classmethod
    def connect(
        cls,
        settings: Neo4jSettings | None = None,
        *,
        wait_seconds: float = 45.0,
    ) -> Neo4jProjectionStore:
        value = settings or Neo4jSettings.from_env()
        if GraphDatabase is None:
            raise Neo4jProjectionError(
                "Neo4j driver is available only in the Phase 20 client image"
            )
        driver = GraphDatabase.driver(value.uri, auth=(value.user, value.password))
        deadline = time.monotonic() + max(0.0, wait_seconds)
        while True:
            try:
                driver.verify_connectivity()
                break
            except ServiceUnavailable:
                if time.monotonic() >= deadline:
                    driver.close()
                    raise
                time.sleep(min(1.0, max(0.0, deadline - time.monotonic())))
        return cls(driver, value.database)

    def close(self) -> None:
        self.driver.close()

    def server_version(self) -> str:
        with self.driver.session(database=self.database) as session:
            row = session.run(
                "CALL dbms.components() YIELD versions RETURN versions[0] AS version"
            ).single(strict=True)
        return str(row["version"])

    def rebuild(self, plan: ProjectionPlan) -> dict[str, Any]:
        with self.driver.session(database=self.database) as session:
            for query in _CONSTRAINTS:
                session.run(query).consume()
            session.execute_write(_replace_projection, plan)
        counts = self.counts(plan.graph_id)
        if counts != plan.counts():
            raise Neo4jProjectionError(
                f"projection count mismatch: expected {plan.counts()}, got {counts}"
            )
        return {
            "graph_id": plan.graph_id,
            "projection_hash": plan.projection_hash,
            "counts": counts,
            "neo4j_version": self.server_version(),
        }

    def counts(self, graph_id: str) -> dict[str, int]:
        query = """
        MATCH (g:CausalProjection {graph_id: $graph_id})
        OPTIONAL MATCH (n:CausalNode {graph_id: $graph_id})
        WITH g, count(DISTINCT n) AS nodes
        OPTIONAL MATCH (f:CausalEdgeFact {graph_id: $graph_id})
        WITH g, nodes, count(DISTINCT f) AS edges
        OPTIONAL MATCH (e:CausalEvidence {graph_id: $graph_id})
        WITH g, nodes, edges, count(DISTINCT e) AS evidence
        OPTIONAL MATCH (t:CausalTechnique {graph_id: $graph_id})
        RETURN nodes, edges, evidence, count(DISTINCT t) AS techniques
        """
        with self.driver.session(database=self.database) as session:
            row = session.run(query, graph_id=graph_id).single(strict=True)
        return {name: int(row[name]) for name in ("nodes", "edges", "evidence", "techniques")}

    def analyze(self, graph_id: str) -> dict[str, Any]:
        output: dict[str, Any] = {"graph_id": graph_id}
        with self.driver.session(database=self.database) as session:
            for name, query in _ANALYSIS_QUERIES.items():
                output[name] = [dict(row) for row in session.run(query, graph_id=graph_id)]
        output["analysis_hash"] = canonical_sha256(output)
        return output


def _replace_projection(tx: Any, plan: ProjectionPlan) -> None:
    tx.run("MATCH (n {graph_id: $graph_id}) DETACH DELETE n", graph_id=plan.graph_id).consume()
    tx.run("CREATE (g:CausalProjection) SET g = $row", row=plan.graph_row).consume()
    tx.run(
        "UNWIND $rows AS row CREATE (n:CausalNode) SET n = row",
        rows=list(plan.node_rows),
    ).consume()
    tx.run(
        "UNWIND $rows AS row CREATE (e:CausalEvidence) SET e = row",
        rows=list(plan.evidence_rows),
    ).consume()
    tx.run(
        "UNWIND $rows AS row CREATE (f:CausalEdgeFact) SET f = row",
        rows=list(plan.edge_rows),
    ).consume()
    tx.run(
        "UNWIND $rows AS row CREATE (t:CausalTechnique) SET t = row",
        rows=list(plan.technique_rows),
    ).consume()
    tx.run(
        """
        UNWIND $rows AS row
        MATCH (f:CausalEdgeFact {projection_key: row.projection_key})
        MATCH (s:CausalNode {projection_key: row.source_key})
        MATCH (t:CausalNode {projection_key: row.target_key})
        CREATE (f)-[:FROM]->(s)
        CREATE (f)-[:TARGETS]->(t)
        """,
        rows=list(plan.edge_rows),
    ).consume()
    tx.run(
        """
        UNWIND $rows AS row
        MATCH (f:CausalEdgeFact {projection_key: row.projection_key})
        UNWIND row.evidence_keys AS evidence_key
        MATCH (e:CausalEvidence {projection_key: evidence_key})
        CREATE (f)-[:SUPPORTED_BY]->(e)
        """,
        rows=list(plan.edge_rows),
    ).consume()
    tx.run(
        """
        UNWIND $rows AS row
        MATCH (f:CausalEdgeFact {projection_key: row.projection_key})
        UNWIND row.mitre_technique_ids AS technique_id
        MATCH (t:CausalTechnique {graph_id: row.graph_id, technique_id: technique_id})
        CREATE (f)-[:USES_TECHNIQUE]->(t)
        """,
        rows=list(plan.edge_rows),
    ).consume()
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in plan.edge_rows:
        grouped[row["relationship_type"]].append(row)
    allowed = {item.value for item in RelationshipType}
    for relationship_type, rows in sorted(grouped.items()):
        if relationship_type not in allowed:
            raise Neo4jProjectionError(f"unsafe relationship type: {relationship_type}")
        query = f"""
        UNWIND $rows AS row
        MATCH (s:CausalNode {{projection_key: row.source_key}})
        MATCH (t:CausalNode {{projection_key: row.target_key}})
        CREATE (s)-[r:{relationship_type}]->(t)
        SET r = row
        """
        tx.run(query, rows=rows).consume()


_CONSTRAINTS = (
    "CREATE CONSTRAINT causal_projection_key IF NOT EXISTS "
    "FOR (n:CausalProjection) REQUIRE n.projection_key IS UNIQUE",
    "CREATE CONSTRAINT causal_node_projection_key IF NOT EXISTS "
    "FOR (n:CausalNode) REQUIRE n.projection_key IS UNIQUE",
    "CREATE CONSTRAINT causal_edge_projection_key IF NOT EXISTS "
    "FOR (n:CausalEdgeFact) REQUIRE n.projection_key IS UNIQUE",
    "CREATE CONSTRAINT causal_evidence_projection_key IF NOT EXISTS "
    "FOR (n:CausalEvidence) REQUIRE n.projection_key IS UNIQUE",
    "CREATE CONSTRAINT causal_technique_projection_key IF NOT EXISTS "
    "FOR (n:CausalTechnique) REQUIRE n.projection_key IS UNIQUE",
)

_ANALYSIS_QUERIES = {
    "path_frequency": """
        MATCH (f:CausalEdgeFact {graph_id: $graph_id})
        WITH f ORDER BY f.episode_id, f.step, f.edge_id
        WITH f.episode_id AS episode_id,
             collect(f.source_node + '-[' + f.relationship_type + ']->' +
                     f.target_node) AS path_segments,
             collect(f.edge_id) AS evidence_edge_ids
        ORDER BY episode_id
        WITH path_segments,
             collect(episode_id) AS episode_ids,
             collect(evidence_edge_ids) AS evidence_by_episode
        RETURN path_segments, size(episode_ids) AS episode_count,
               episode_ids[0..25] AS episode_ids,
               evidence_by_episode[0][0..25] AS representative_evidence_edge_ids
        ORDER BY episode_count DESC, path_segments
    """,
    "relationship_frequency": """
        MATCH (f:CausalEdgeFact {graph_id: $graph_id})
        WITH f.source_node AS source_node, f.target_node AS target_node,
             f.relationship_type AS relationship_type, f
        ORDER BY f.edge_id
        RETURN source_node, target_node, relationship_type, count(*) AS edge_count,
               count(DISTINCT f.episode_id) AS episode_count,
               collect(f.edge_id)[0..25] AS evidence_edge_ids
        ORDER BY edge_count DESC, source_node, target_node, relationship_type
    """,
    "bottleneck_assets": """
        MATCH (f:CausalEdgeFact {graph_id: $graph_id})
        WHERE f.success = true
        WITH f.target_node AS node_id, f ORDER BY f.edge_id
        RETURN node_id, count(*) AS successful_edge_count,
               count(DISTINCT f.episode_id) AS episode_count,
               collect(f.edge_id)[0..25] AS evidence_edge_ids
        ORDER BY episode_count DESC, successful_edge_count DESC, node_id
    """,
    "bottleneck_vulnerabilities": """
        MATCH (f:CausalEdgeFact {graph_id: $graph_id})
        WHERE f.vulnerability_id <> 'unknown'
        WITH f.vulnerability_id AS vulnerability_id, f ORDER BY f.edge_id
        RETURN vulnerability_id, count(*) AS edge_count,
               sum(CASE WHEN f.success THEN 1 ELSE 0 END) AS successful_edge_count,
               count(DISTINCT f.episode_id) AS episode_count,
               collect(f.edge_id)[0..25] AS evidence_edge_ids
        ORDER BY episode_count DESC, edge_count DESC, vulnerability_id
    """,
    "degree_centrality": """
        MATCH (n:CausalNode {graph_id: $graph_id})
        WITH n,
             COUNT { (n)-->(:CausalNode {graph_id: $graph_id}) } AS outgoing_degree,
             COUNT { (:CausalNode {graph_id: $graph_id})-->(n) } AS incoming_degree
        RETURN n.node_id AS node_id, outgoing_degree, incoming_degree,
               outgoing_degree + incoming_degree AS total_degree,
               n.evidence_ids AS evidence_ids
        ORDER BY total_degree DESC, node_id
    """,
    "mitre_relationships": """
        MATCH (f:CausalEdgeFact {graph_id: $graph_id})-[:USES_TECHNIQUE]->
              (t:CausalTechnique {graph_id: $graph_id})
        WITH t.technique_id AS technique_id, t.technique_name AS technique_name,
             f.target_node AS node_id, f ORDER BY f.edge_id
        RETURN technique_id, technique_name, node_id, count(*) AS observation_count,
               collect(f.edge_id)[0..25] AS evidence_edge_ids
        ORDER BY observation_count DESC, technique_id, node_id
    """,
}


def _key(graph_id: str, kind: str, identifier: str) -> str:
    return canonical_sha256([graph_id, kind, identifier])


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


__all__ = [
    "NEO4J_PROJECTION_SCHEMA_VERSION",
    "MissingNeo4jCredentials",
    "Neo4jProjectionError",
    "Neo4jProjectionStore",
    "Neo4jSettings",
    "ProjectionPlan",
    "build_projection_plan",
]
