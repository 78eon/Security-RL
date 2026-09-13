from __future__ import annotations

import json

import pytest

from rlredteam.attack_path_report import generate_report, write_report
from rlredteam.phase14_completion import Phase14VerificationError, verify_report
from tests.test_attack_path_report import simulation_document


def test_phase14_verifier_reconstructs_every_fact(tmp_path) -> None:
    source = tmp_path / "source.json"
    output = tmp_path / "report.json"
    source.write_text(json.dumps(simulation_document()))
    report = generate_report(
        source,
        mode="simulation_report",
        experiment_id="experiment-test",
        checkpoint_hashes={"sim-s42": "b" * 64, "sim-s43": "c" * 64},
    )
    write_report(report, output)

    result = verify_report(source, output)

    assert result["status"] == "pass"
    assert result["facts"] == len(report.facts)
    assert result["catalogue_only_mappings"] == "pass"
    assert result["hidden_topology_leakage"] == "none"


def test_phase14_verifier_rejects_a_fabricated_report_fact(tmp_path) -> None:
    source = tmp_path / "source.json"
    output = tmp_path / "report.json"
    source.write_text(json.dumps(simulation_document()))
    report = generate_report(
        source,
        mode="simulation_report",
        experiment_id="experiment-test",
    ).to_dict()
    report["facts"][0]["target"] = "invented-host"
    output.write_text(json.dumps(report))

    with pytest.raises(Phase14VerificationError, match="exactly derivable"):
        verify_report(source, output)
