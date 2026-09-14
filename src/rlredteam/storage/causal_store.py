"""PostgreSQL authority for Phase 19 causal graph projections."""

from __future__ import annotations

from typing import Any

import psycopg
from psycopg.types.json import Jsonb

from rlredteam.causal_graph import CausalGraph, graph_from_dict


class CausalGraphPersistenceError(RuntimeError):
    """A graph collision or normalized-row integrity failure."""


class CausalGraphStore:
    def __init__(self, connection: psycopg.Connection) -> None:
        self.connection = connection

    def save(self, graph: CausalGraph) -> str:
        evidence_ids = {item.evidence_id for item in graph.evidence}
        if any(
            not edge.evidence_ids or not set(edge.evidence_ids) <= evidence_ids
            for edge in graph.edges
        ):
            raise CausalGraphPersistenceError(
                "every edge must reference evidence stored in the same graph"
            )
        existing = self.connection.execute(
            "SELECT graph_id FROM causal_attack_graphs WHERE graph_id = %s",
            (graph.graph_id,),
        ).fetchone()
        if existing:
            if self.load(graph.graph_id).to_dict() != graph.to_dict():
                raise CausalGraphPersistenceError(
                    f"immutable causal graph ID collision: {graph.graph_id}"
                )
            return graph.graph_id
        try:
            self.connection.execute(
                """
                INSERT INTO causal_attack_graphs
                    (graph_id, report_id, schema_version, provenance)
                VALUES (%s,%s,%s,%s)
                """,
                (
                    graph.graph_id,
                    graph.report_id,
                    graph.schema_version,
                    Jsonb(graph.provenance),
                ),
            )
            with self.connection.cursor() as cursor:
                cursor.executemany(
                    """
                    INSERT INTO causal_knowledge_evidence
                        (graph_id, evidence_id, episode_key, step_idx,
                         source_event_sha256, evidence_kind, evidence_data)
                    VALUES (%s,%s,%s,%s,%s,%s,%s)
                    """,
                    [
                        (
                            graph.graph_id,
                            item.evidence_id,
                            item.episode_id,
                            item.step,
                            item.source_event_sha256,
                            item.evidence_kind,
                            Jsonb(item.payload),
                        )
                        for item in graph.evidence
                    ],
                )
                cursor.executemany(
                    """
                    INSERT INTO causal_node_facts
                        (graph_id, node_id, node_type, first_discovery_step,
                         discovered_by, evidence_ids, node_data)
                    VALUES (%s,%s,%s,%s,%s,%s,%s)
                    """,
                    [
                        (
                            graph.graph_id,
                            item.node_id,
                            item.node_type,
                            str(item.first_discovery_step),
                            item.discovered_by,
                            list(item.evidence_ids),
                            Jsonb(_node_dict(item)),
                        )
                        for item in graph.nodes
                    ],
                )
                cursor.executemany(
                    """
                    INSERT INTO causal_edge_facts
                        (graph_id, edge_id, source_node, target_node,
                         relationship_type, episode_key, step_idx,
                         evidence_ids, edge_data)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
                    """,
                    [
                        (
                            graph.graph_id,
                            item.edge_id,
                            item.source_node,
                            item.target_node,
                            item.relationship_type,
                            item.episode_id,
                            item.step,
                            list(item.evidence_ids),
                            Jsonb(_edge_dict(item)),
                        )
                        for item in graph.edges
                    ],
                )
            self.connection.commit()
        except Exception:
            self.connection.rollback()
            raise
        return graph.graph_id

    def load(self, graph_id: str) -> CausalGraph:
        header = self.connection.execute(
            """
            SELECT report_id, schema_version, provenance
            FROM causal_attack_graphs WHERE graph_id = %s
            """,
            (graph_id,),
        ).fetchone()
        if header is None:
            raise KeyError(f"causal graph not found: {graph_id}")
        nodes = [
            dict(row[0])
            for row in self.connection.execute(
                """
                SELECT node_data FROM causal_node_facts
                WHERE graph_id = %s ORDER BY node_id
                """,
                (graph_id,),
            ).fetchall()
        ]
        edges = [
            dict(row[0])
            for row in self.connection.execute(
                """
                SELECT edge_data FROM causal_edge_facts
                WHERE graph_id = %s ORDER BY edge_id
                """,
                (graph_id,),
            ).fetchall()
        ]
        evidence = [
            {
                "evidence_id": row[0],
                "report_id": header[0],
                "episode_id": row[1],
                "step": row[2],
                "source_event_sha256": row[3],
                "evidence_kind": row[4],
                "payload": dict(row[5]),
            }
            for row in self.connection.execute(
                """
                SELECT evidence_id, episode_key, step_idx,
                       source_event_sha256, evidence_kind, evidence_data
                FROM causal_knowledge_evidence
                WHERE graph_id = %s ORDER BY episode_key, step_idx, evidence_id
                """,
                (graph_id,),
            ).fetchall()
        ]
        graph = graph_from_dict(
            {
                "graph_id": graph_id,
                "report_id": header[0],
                "schema_version": header[1],
                "nodes": nodes,
                "edges": edges,
                "evidence": evidence,
                "provenance": dict(header[2]),
            }
        )
        if graph.to_dict()["graph_id"] != graph_id:
            raise CausalGraphPersistenceError("reconstructed graph identity mismatch")
        return graph

    def list(self, limit: int = 50) -> list[dict[str, Any]]:
        if not 1 <= int(limit) <= 500:
            raise ValueError("graph limit must be between 1 and 500")
        rows = self.connection.execute(
            """
            SELECT graph_id FROM causal_attack_graphs
            ORDER BY created_at DESC, graph_id LIMIT %s
            """,
            (int(limit),),
        ).fetchall()
        return [self.load(str(row[0])).to_dict() for row in rows]


def _node_dict(value: Any) -> dict[str, Any]:
    from dataclasses import asdict

    return asdict(value)


def _edge_dict(value: Any) -> dict[str, Any]:
    from dataclasses import asdict

    return asdict(value)


__all__ = ["CausalGraphPersistenceError", "CausalGraphStore"]
