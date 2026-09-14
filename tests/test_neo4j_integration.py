"""Live Podman Neo4j reconstruction and query checks."""

from __future__ import annotations

import pytest

from rlredteam.causal_graph import build_causal_graph
from rlredteam.storage.neo4j_projection import (
    MissingNeo4jCredentials,
    Neo4jProjectionStore,
    build_projection_plan,
)
from tests.test_causal_graph import causal_document, causal_report


def _connect_or_skip() -> Neo4jProjectionStore:
    try:
        return Neo4jProjectionStore.connect()
    except MissingNeo4jCredentials as exc:
        pytest.skip(f"Neo4j service not configured: {exc}")


def test_live_projection_is_idempotent_and_analyzable() -> None:
    graph = build_causal_graph(causal_report(), causal_document())
    plan = build_projection_plan(graph)
    store = _connect_or_skip()
    try:
        first = store.rebuild(plan)
        analysis_a = store.analyze(graph.graph_id)
        second = store.rebuild(plan)
        analysis_b = store.analyze(graph.graph_id)
        with store.driver.session(database=store.database) as session:
            unsupported = session.run(
                """
                MATCH (:CausalNode {graph_id: $graph_id})-[r]->
                      (:CausalNode {graph_id: $graph_id})
                WHERE r.evidence_ids IS NULL OR size(r.evidence_ids) = 0
                RETURN count(r) AS count
                """,
                graph_id=graph.graph_id,
            ).single(strict=True)["count"]
            session.run(
                "MATCH (n {graph_id: $graph_id}) DETACH DELETE n",
                graph_id=graph.graph_id,
            ).consume()
    finally:
        store.close()
    assert first["projection_hash"] == second["projection_hash"]
    assert first["counts"] == second["counts"] == plan.counts()
    assert analysis_a == analysis_b
    assert unsupported == 0
