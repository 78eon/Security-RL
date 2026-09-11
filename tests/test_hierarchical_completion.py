from pathlib import Path

import pytest

from rlredteam.enterprise.hierarchical_completion import (
    HierarchicalCompletionError,
    require,
    verify_hierarchical_completion,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CANONICAL_ROOT = REPO_ROOT / "results" / "advanced-rl-hierarchical-policy-v1" / "test"


def test_hierarchical_completion_gate_fails_closed() -> None:
    with pytest.raises(HierarchicalCompletionError, match="missing proof"):
        require(False, "missing proof")


@pytest.mark.skipif(
    not CANONICAL_ROOT.exists(),
    reason="canonical Phase 12 evidence is intentionally absent from a clean clone",
)
def test_canonical_hierarchical_evidence_passes_completion_gate() -> None:
    report = verify_hierarchical_completion(REPO_ROOT)
    assert report["complete"] is True
    assert set(report["checks"].values()) == {"pass", "not-requested"}
    assert report["evidence"]["runs"] == 20
    assert report["evidence"]["episodes"] == 1200
    assert report["evidence"]["attack_paths"] == 1200
    assert report["analysis"]["pairs"] == 10
    assert report["objective_head_parameter_delta"] == 516
    assert (
        report["parameter_counts"]["hierarchical_graph"]
        - report["parameter_counts"]["flat_graph"]
        == 516
    )
