from pathlib import Path

import pytest

from rlredteam.enterprise.completion import (
    CompletionVerificationError,
    require,
    verify_onprem_completion,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CANONICAL_CHECKPOINT = REPO_ROOT / "runs" / "onprem-generalisation" / "model.zip"


def test_completion_gate_fails_closed() -> None:
    with pytest.raises(CompletionVerificationError, match="missing proof"):
        require(False, "missing proof")


@pytest.mark.skipif(
    not CANONICAL_CHECKPOINT.exists(),
    reason="gated canonical weights are intentionally absent from a clean clone",
)
def test_canonical_onprem_evidence_passes_completion_gate() -> None:
    report = verify_onprem_completion(REPO_ROOT)

    assert report["complete"] is True
    assert set(report["checks"].values()) == {"pass", "not-requested"}
    assert report["fixed_baseline"]["interpretation"] == "provisional"
