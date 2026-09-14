from __future__ import annotations

import json
from dataclasses import asdict

import numpy as np
import pytest

from rlredteam.assign import assign_cves
from rlredteam.catalogue import CVECatalogue
from rlredteam.mitigation import (
    CRITICALITY_LABEL,
    EFFECT_LABEL,
    INTERVENTION_VERSION,
    REPORT_SCHEMA_VERSION,
    CveMitigationOverlay,
    MitigationError,
    MitigationSpec,
    analyse_pairs,
    artifact_tree_digest,
    canonical_sha256,
    evaluate_checkpoint_mitigation,
    promote_phase14_cve,
    validate_report_identity,
)
from rlredteam.topology import REPO_ROOT, TopologyConfig, make_env
from tests.test_attack_path_report import make_report


def _action_index(env, name: str) -> int:
    return next(
        index
        for index in range(int(env.action_space.n))
        if env.action_space.get_action(index).name == name
    )


@pytest.mark.integration
def test_overlay_changes_only_selected_cve_and_not_observation_interface(monkeypatch) -> None:
    config = TopologyConfig.from_yaml()
    catalogue = CVECatalogue.open_default()
    base = make_env(config, topology_seed=42)
    assignment = assign_cves(
        list(base.scenario.exploits), list(base.scenario.privescs), catalogue, 42
    )
    selected = "CVE-2024-6387"
    spec = MitigationSpec.create(selected, catalogue)
    overlay = CveMitigationOverlay(base, catalogue=catalogue, topology_seed=42, spec=spec)

    assert overlay.assignment.mapping == assignment.mapping
    assert set(overlay.affected_actions) == {
        action for action, cve_id in assignment.mapping.items() if cve_id == selected
    }
    assert all(
        overlay.assignment.mapping[action] == cve_id
        for action, cve_id in assignment.mapping.items()
        if cve_id != selected
    )
    original = make_env(config, topology_seed=42)
    original_observation, _ = original.reset(seed=1001)
    mitigated_observation, _ = overlay.reset(seed=1001)
    assert overlay.observation_space == original.observation_space
    assert overlay.action_space == original.action_space
    np.testing.assert_array_equal(mitigated_observation, original_observation)

    delegated: list[int] = []
    original_step = base.step

    def record_step(action):
        delegated.append(int(action))
        return original_step(action)

    monkeypatch.setattr(base, "step", record_step)
    blocked = _action_index(overlay, overlay.affected_actions[0])
    next_observation, _, terminated, _, info = overlay.step(blocked)
    assert not terminated
    assert info["mitigation_blocked"] is True
    assert info["success"] is False
    np.testing.assert_array_equal(next_observation, mitigated_observation)
    assert delegated == []

    unaffected = next(
        index
        for index in range(int(overlay.action_space.n))
        if overlay.action_space.get_action(index).name not in overlay.affected_actions
    )
    overlay.step(unaffected)
    assert delegated == [unaffected]


def _condition(success: bool, steps: int, reward: float, cve: str = "CVE-2021-0001") -> dict:
    return {
        "success": success,
        "steps": steps,
        "reward": reward,
        "policy_reward": reward,
        "crown_jewel_reach": success,
        "terminal_reason": "goal" if success else "step_limit",
        "mitigation_blocked_attempts": 0,
        "mitre_techniques": [],
        "observed_path": [{"cve_id": cve}] if success else [],
    }


def make_pairs() -> list[dict]:
    rows = []
    for seed, original_success, mitigated_success in (
        (1001, True, False),
        (1002, True, True),
        (1003, False, False),
        (1004, True, False),
    ):
        original = _condition(original_success, 8, 20.0)
        mitigated = _condition(mitigated_success, 12, 10.0)
        rows.append(
            {
                "evaluation_seed": seed,
                "original": original,
                "mitigated": mitigated,
                "delta": {
                    "success": int(mitigated_success) - int(original_success),
                    "steps": 4,
                    "reward": -10.0,
                    "crown_jewel_reach": int(mitigated_success) - int(original_success),
                },
            }
        )
    return rows


def make_mitigation_report() -> dict:
    pairs = make_pairs()
    body = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "claim_boundaries": {
            "observed_path_criticality": CRITICALITY_LABEL,
            "mitigation_effect": EFFECT_LABEL,
            "causal_scope": "sample-specific",
        },
        "intervention": {
            **asdict(MitigationSpec("CVE-2024-6387")),
            "intervention_sha256": MitigationSpec("CVE-2024-6387").digest(),
            "affected_actions": ["e_srv_0_os_0"],
            "original_assignment_sha256": "a" * 64,
            "unselected_assignments_changed": 0,
            "canonical_catalogue_mutated": False,
        },
        "phase14_promotion": None,
        "pairs": pairs,
        "analysis": analyse_pairs(pairs),
        "provenance": {
            "checkpoint_sha256": "b" * 64,
            "policy_sha256_before": "c" * 64,
            "policy_sha256_after_original": "c" * 64,
            "policy_sha256_after_mitigated": "c" * 64,
            "gradient_updates": False,
            "evaluation_git_dirty": False,
            "evaluation_seeds_original": [1001, 1002, 1003, 1004],
            "evaluation_seeds_mitigated": [1001, 1002, 1003, 1004],
            "protected_artifacts_before": {"results/frozen": "d" * 64},
            "protected_artifacts_after": {"results/frozen": "d" * 64},
        },
    }
    return {"report_id": canonical_sha256(body), **body}


def test_paired_statistics_report_effect_and_confidence_intervals() -> None:
    analysis = analyse_pairs(make_pairs())

    assert analysis["original_success_rate"] == pytest.approx(0.75)
    assert analysis["mitigated_success_rate"] == pytest.approx(0.25)
    assert analysis["absolute_success_rate_change"] == pytest.approx(-0.5)
    assert analysis["relative_success_rate_change"] == pytest.approx(-2 / 3)
    assert analysis["mcnemar_exact"]["discordant_pairs"] == 2
    assert len(analysis["paired_success_change_ci95_bootstrap"]) == 2
    assert analysis["reward_change"]["paired_mean_change"] == -10.0


def test_phase14_criticality_can_be_promoted_without_changing_its_claim(tmp_path) -> None:
    source = tmp_path / "phase14.json"
    report = make_report().to_dict()
    source.write_text(json.dumps(report))
    cve = report["observed_path_criticality"][0]["cve_id"]

    promoted = promote_phase14_cve(source, cve)

    assert promoted["selected_cve"] == cve
    assert promoted["observed_path_criticality_label"] == CRITICALITY_LABEL
    assert "mitigation" not in promoted["observed_path_criticality_label"]


def test_phase14_noncritical_cve_cannot_be_promoted(tmp_path) -> None:
    source = tmp_path / "phase14.json"
    source.write_text(json.dumps(make_report().to_dict()))

    with pytest.raises(MitigationError, match="not observed path-critical"):
        promote_phase14_cve(source, "CVE-1999-9999")


def test_report_identity_detects_any_pair_tampering() -> None:
    report = make_mitigation_report()
    validate_report_identity(report)
    report["pairs"][0]["mitigated"]["success"] = True

    with pytest.raises(MitigationError, match="identity"):
        validate_report_identity(report)


def test_intervention_version_is_explicit() -> None:
    assert MitigationSpec("CVE-2024-6387").intervention_version == INTERVENTION_VERSION


@pytest.mark.slow
def test_real_frozen_checkpoint_is_paired_without_policy_or_artifact_mutation(
    tmp_path,
) -> None:
    run = REPO_ROOT / "runs/experiment_01-shaped-s42-t42"
    if not (run / "model.zip").is_file():
        pytest.skip("local frozen Essential checkpoint is not mounted")
    before = artifact_tree_digest(run)

    report = evaluate_checkpoint_mitigation(
        run,
        [1001],
        "CVE-2024-6387",
        tmp_path / "report.json",
    )

    provenance = report["provenance"]
    assert provenance["evaluation_seeds_original"] == [1001]
    assert provenance["evaluation_seeds_mitigated"] == [1001]
    assert provenance["gradient_updates"] is False
    assert len(
        {
            provenance["policy_sha256_before"],
            provenance["policy_sha256_after_original"],
            provenance["policy_sha256_after_mitigated"],
        }
    ) == 1
    assert artifact_tree_digest(run) == before
