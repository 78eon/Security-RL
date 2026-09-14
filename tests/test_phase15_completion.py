from __future__ import annotations

import json

import pytest

from rlredteam.phase15_completion import Phase15VerificationError, verify_report
from tests.test_mitigation import make_mitigation_report


def test_phase15_verifier_accepts_complete_paired_report(tmp_path) -> None:
    path = tmp_path / "report.json"
    path.write_text(json.dumps(make_mitigation_report()))

    result = verify_report(path)

    assert result["status"] == "pass"
    assert result["policy_immutable"] == "pass"
    assert result["frozen_artifacts_immutable"] == "pass"


def test_phase15_verifier_rejects_seed_pairing_drift(tmp_path) -> None:
    report = make_mitigation_report()
    report["provenance"]["evaluation_seeds_mitigated"][-1] = 9999
    path = tmp_path / "report.json"
    path.write_text(json.dumps(report))

    with pytest.raises(Phase15VerificationError, match="invalid report|identity"):
        verify_report(path)
