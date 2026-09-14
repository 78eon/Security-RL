"""PostgreSQL normalized reconstruction for causal graph facts."""

from __future__ import annotations

import psycopg
import pytest

from rlredteam.causal_graph import build_causal_graph
from rlredteam.storage.causal_store import CausalGraphStore
from rlredteam.storage.postgres_logger import connection_string, ensure_schema
from tests.test_causal_graph import causal_document, causal_report

pytestmark = pytest.mark.postgres


def test_causal_graph_round_trips_normalized_postgres() -> None:
    graph = build_causal_graph(causal_report(), causal_document())
    with psycopg.connect(connection_string()) as connection:
        ensure_schema(connection)
        store = CausalGraphStore(connection)
        store.save(graph)
        restored = store.load(graph.graph_id)
        edge_rows = connection.execute(
            "SELECT count(*) FROM causal_edge_facts WHERE graph_id = %s",
            (graph.graph_id,),
        ).fetchone()[0]
        evidence_rows = connection.execute(
            "SELECT count(*) FROM causal_knowledge_evidence WHERE graph_id = %s",
            (graph.graph_id,),
        ).fetchone()[0]
        connection.execute(
            "DELETE FROM causal_attack_graphs WHERE graph_id = %s", (graph.graph_id,)
        )
        connection.commit()
    assert restored.to_dict() == graph.to_dict()
    assert edge_rows == len(graph.edges)
    assert evidence_rows == len(graph.evidence)
