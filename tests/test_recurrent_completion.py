from pathlib import Path

import pytest

from rlredteam.enterprise.recurrent_completion import (
    RecurrentCompletionError,
    require,
    verify_recurrent_completion,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CANONICAL_ROOT = REPO_ROOT / "results" / "advanced-rl-recurrent-v1" / "test"


def test_recurrent_completion_gate_fails_closed() -> None:
    with pytest.raises(RecurrentCompletionError, match="missing proof"):
        require(False, "missing proof")


@pytest.mark.skipif(
    not CANONICAL_ROOT.exists(),
    reason="canonical Phase 8 evidence is intentionally absent from a clean clone",
)
def test_canonical_recurrent_evidence_passes_completion_gate() -> None:
    report = verify_recurrent_completion(REPO_ROOT)
    assert report["complete"] is True
    assert set(report["checks"].values()) == {"pass", "not-requested"}
    assert report["evidence"]["runs"] == 20
    assert report["evidence"]["episodes"] == 1200
    assert report["evidence"]["attack_paths"] == 1200
