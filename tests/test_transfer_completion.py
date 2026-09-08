from pathlib import Path

import pytest

from rlredteam.enterprise.transfer_completion import (
    TransferCompletionError,
    require,
    verify_transfer_completion,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CANONICAL_ROOT = REPO_ROOT / "results" / "advanced-rl-transfer-v1" / "test"


def test_transfer_completion_gate_fails_closed() -> None:
    with pytest.raises(TransferCompletionError, match="missing proof"):
        require(False, "missing proof")


@pytest.mark.skipif(
    not CANONICAL_ROOT.exists(),
    reason="canonical Phase 11 evidence is intentionally absent from a clean clone",
)
def test_canonical_transfer_evidence_passes_completion_gate() -> None:
    report = verify_transfer_completion(REPO_ROOT)
    assert report["complete"] is True
    assert set(report["checks"].values()) == {"pass", "not-requested"}
    assert report["evidence"]["runs"] == 20
    assert report["evidence"]["episodes"] == 400
    assert report["evidence"]["attack_paths"] == 400
    assert report["evidence"]["source_training_steps"] == 256_000
    assert report["evidence"]["target_training_steps"] == 256_000
    assert report["analysis"]["pairs"] == 10
