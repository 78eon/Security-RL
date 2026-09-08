from pathlib import Path

import pytest

from rlredteam.enterprise.graph_policy_completion import (
    GraphPolicyCompletionError,
    require,
    verify_graph_policy_completion,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CANONICAL_ROOT = REPO_ROOT / "results" / "advanced-rl-graph-policy-v1" / "test"


def test_graph_policy_completion_gate_fails_closed() -> None:
    with pytest.raises(GraphPolicyCompletionError, match="missing proof"):
        require(False, "missing proof")


@pytest.mark.skipif(
    not CANONICAL_ROOT.exists(),
    reason="canonical Phase 10 evidence is intentionally absent from a clean clone",
)
def test_canonical_graph_policy_evidence_passes_completion_gate() -> None:
    report = verify_graph_policy_completion(REPO_ROOT)
    assert report["complete"] is True
    assert set(report["checks"].values()) == {"pass", "not-requested"}
    assert report["evidence"]["runs"] == 20
    assert report["evidence"]["episodes"] == 1200
    assert report["evidence"]["attack_paths"] == 1200
    assert report["analysis"]["pairs"] == 10
    assert (
        report["parameter_counts"]["flat_mlp"] > report["parameter_counts"]["knowledge_graph_gnn"]
    )
