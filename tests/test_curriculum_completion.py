from pathlib import Path

import pytest

from rlredteam.enterprise.curriculum_completion import (
    CurriculumCompletionError,
    require,
    verify_curriculum_completion,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CANONICAL_ROOT = REPO_ROOT / "results" / "advanced-rl-curriculum-v1" / "test"


def test_curriculum_completion_gate_fails_closed() -> None:
    with pytest.raises(CurriculumCompletionError, match="missing proof"):
        require(False, "missing proof")


@pytest.mark.skipif(
    not CANONICAL_ROOT.exists(),
    reason="canonical Phase 9 evidence is intentionally absent from a clean clone",
)
def test_canonical_curriculum_evidence_passes_completion_gate() -> None:
    report = verify_curriculum_completion(REPO_ROOT)
    assert report["complete"] is True
    assert set(report["checks"].values()) == {"pass", "not-requested"}
    assert report["evidence"]["runs"] == 20
    assert report["evidence"]["episodes"] == 1200
    assert report["evidence"]["attack_paths"] == 1200
    assert report["analysis"]["pairs"] == 10
