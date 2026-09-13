from __future__ import annotations

import json

import pytest

from rlredteam.attack_path_report import (
    CRITICALITY_METHOD,
    EvidenceValidationError,
    ReportMode,
    build_report,
)


def simulation_document() -> list[dict]:
    return [
        {
            "run_name": "sim-s42",
            "evaluation_seed": 1001,
            "training_seed": 42,
            "reward_mode": "shaped",
            "policy_return": 4.0,
            "native_return": 10.0,
            "goal_reached": True,
            "events": [
                {
                    "step": 0,
                    "action": "discover:dmz",
                    "action_kind": "discover_network",
                    "target": "dmz",
                    "success": True,
                    "state_changed": True,
                    "newly_discovered": 1,
                    "outcomes": ["reachable:app"],
                },
                {
                    "step": 1,
                    "action": "exploit:unused",
                    "action_kind": "exploit",
                    "target": "unused",
                    "success": True,
                    "state_changed": True,
                    "cve_id": "CVE-2020-1111",
                    "cvss_base": 7.5,
                    "outcomes": ["access:unused"],
                },
                {
                    "step": 2,
                    "action": "exploit:app",
                    "action_kind": "exploit",
                    "target": "app",
                    "success": True,
                    "state_changed": True,
                    "cve_id": "CVE-2021-2222",
                    "cvss_base": 9.8,
                    "prerequisites": ["reachable:app"],
                    "outcomes": ["access:asset"],
                    "access_gained": 2,
                },
                {
                    "step": 3,
                    "action": "access:asset",
                    "action_kind": "access_asset",
                    "target": "asset",
                    "success": True,
                    "state_changed": True,
                    "is_crown_jewel": True,
                    "prerequisites": ["access:asset"],
                    "outcomes": ["objective:asset"],
                },
                {
                    "step": 4,
                    "action": "scan:after",
                    "action_kind": "enumerate_service",
                    "target": "asset",
                    "success": False,
                    "state_changed": False,
                },
            ],
        },
        {
            "run_name": "sim-s43",
            "evaluation_seed": 1001,
            "training_seed": 43,
            "reward_mode": "shaped",
            "policy_return": 3.0,
            "native_return": 8.0,
            "goal_reached": True,
            "events": [
                {
                    "step": 0,
                    "action": "exploit:crown",
                    "action_kind": "exploit",
                    "target": "crown",
                    "success": True,
                    "state_changed": True,
                    "is_crown_jewel": True,
                    "cve_id": "CVE-2022-3333",
                    "cvss_base": 8.8,
                }
            ],
        },
    ]


def make_report(document=None):
    return build_report(
        simulation_document() if document is None else document,
        source_trajectory_hash="a" * 64,
        mode=ReportMode.SIMULATION,
        experiment_id="experiment-test",
        checkpoint_hashes={"sim-s42": "b" * 64, "sim-s43": "c" * 64},
    )


def test_exact_timeline_uses_catalogue_mappings() -> None:
    report = make_report()
    first = report.facts[0]

    assert (first.framework, first.tactic_id, first.tactic_name) == (
        "attack-enterprise",
        "TA0007",
        "Discovery",
    )
    assert (first.technique_id, first.technique_name) == (
        "T1018",
        "Remote System Discovery",
    )
    assert [phase.tactic_name for phase in report.phases[:4]] == [
        "Discovery",
        "Lateral Movement",
        "Collection",
        "Discovery",
    ]
    assert report.facts[0].knowledge_delta == {"newly_discovered": 1}


def test_atlas_requires_explicit_ai_specific_behavior() -> None:
    ordinary = make_report()
    assert all(fact.framework != "atlas" for fact in ordinary.facts)

    document = simulation_document()[:1]
    document[0]["events"] = [
        {
            "step": 0,
            "action": "inspect:model",
            "action_kind": "enumerate_application",
            "ai_behavior": "discover_model_ontology",
            "target": "model-api",
            "success": True,
            "state_changed": True,
            "is_crown_jewel": True,
        }
    ]
    report = make_report(document)
    assert [fact.technique_id for fact in report.facts] == ["T1518", "AML.T0013"]


def test_observed_path_criticality_uses_causal_evidence_only() -> None:
    report = make_report()
    by_cve = {item.cve_id: item for item in report.observed_path_criticality}

    assert by_cve["CVE-2020-1111"].successful_trajectories_containing == 1
    assert by_cve["CVE-2020-1111"].observed_successful_paths_broken == 0
    assert by_cve["CVE-2021-2222"].observed_successful_paths_broken == 1
    assert by_cve["CVE-2021-2222"].observed_path_criticality_percentage == 50.0
    assert by_cve["CVE-2021-2222"].statement == (
        "CVE-2021-2222 was path-critical in 50.0% of observed successful trajectories."
    )
    assert report.methodology["criticality_method"] == CRITICALITY_METHOD
    assert "all attacks" in report.methodology["criticality_disclaimer"]


def test_cross_episode_metrics_are_trace_derived() -> None:
    report = make_report()
    analysis = report.cross_episode_analysis

    assert analysis["trajectory_count"] == 2
    assert analysis["successful_trajectory_count"] == 2
    assert analysis["path_length"] == {"minimum": 1, "maximum": 5, "mean": 3.0}
    assert analysis["path_diversity"]["proportion_unique"] == 1.0
    assert analysis["crown_jewel_reach_rate"] == 1.0
    assert analysis["failed_action_count"] == 1
    assert analysis["most_frequent_techniques"][0]["technique_id"] == "T1210"


def test_report_hashes_are_stable_and_strict_json() -> None:
    first, second = make_report(), make_report()

    assert first == second
    assert len(first.report_id) == 64
    assert len(first.provenance.facts_payload_sha256) == 64
    assert len(first.provenance.mitre_catalogue_sha256) == 64
    json.dumps(first.to_dict(), allow_nan=False)


def test_hidden_topology_is_rejected_from_normal_reports() -> None:
    document = simulation_document()
    document[0]["true_topology"] = {"secret-host": {}}

    with pytest.raises(EvidenceValidationError, match="hidden topology"):
        make_report(document)


def test_mapping_drift_cannot_bypass_catalogue() -> None:
    document = simulation_document()[:1]
    document[0]["events"][0]["framework_mappings"] = [
        {"framework": "attack-enterprise", "technique_id": "T9999"}
    ]

    with pytest.raises(EvidenceValidationError, match="MITRE mapping"):
        make_report(document)


def test_evidence_report_rejects_ppo_claims() -> None:
    with pytest.raises(EvidenceValidationError, match="PPO/simulator claims"):
        build_report(
            simulation_document(),
            source_trajectory_hash="a" * 64,
            mode=ReportMode.EVIDENCE,
            experiment_id="scanner-import",
        )


def test_evidence_report_contains_no_rl_or_reward_claim() -> None:
    document = [
        {
            "run_id": "scanner-import-1",
            "episode_id": "finding-set-1",
            "objective_reached": False,
            "events": [
                {
                    "step": 0,
                    "action": "observed-service",
                    "action_kind": "enumerate_service",
                    "target": "10.0.0.8:443",
                    "success": True,
                    "state_changed": True,
                    "cve_id": "CVE-2021-4444",
                    "cvss_score": 7.2,
                }
            ],
        }
    ]
    report = build_report(
        document,
        source_trajectory_hash="d" * 64,
        mode=ReportMode.EVIDENCE,
        experiment_id="scanner-import",
    )

    assert report.report_mode == "evidence_report"
    assert report.facts[0].rl_action_index is None
    assert report.facts[0].reward is None
    assert report.provenance.source_checkpoint_hashes == {}
    assert "does not claim PPO exploited" in report.methodology["mode_claim_boundary"]


def test_progress_only_source_does_not_invent_failed_action_count() -> None:
    document = simulation_document()[:1]
    document[0]["progress_steps"] = document[0].pop("events")[:1]
    report = make_report(document)

    assert report.cross_episode_analysis["failed_action_count"] is None
    assert (
        report.cross_episode_analysis["failed_action_observation"]
        == "unavailable_in_progress_only_source"
    )


def test_unknown_ai_behavior_is_rejected_instead_of_coerced_to_atlas() -> None:
    document = simulation_document()[:1]
    document[0]["events"][0]["ai_behavior"] = "pretend_model_attack"

    with pytest.raises(EvidenceValidationError, match="unsupported recorded AI behavior"):
        make_report(document)
