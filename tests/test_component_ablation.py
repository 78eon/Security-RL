"""Phase 18 component isolation and matched-design tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from rlredteam.component_ablation import (
    ComponentExperimentConfig,
    analyse_metrics,
    protected_artifact_hash,
)
from rlredteam.events import AccessLevel, ActionKind, AttackEvent
from rlredteam.reward import RewardComponents, RewardConfig, RewardEngine

ROOT = Path(__file__).resolve().parents[1]
DESIGN = ROOT / "configs/experiments/reward_components_01.yaml"


def exploit_event(
    *, success: bool = True, crown: bool = False, access: AccessLevel = AccessLevel.USER
) -> AttackEvent:
    return AttackEvent(
        step=1,
        kind=ActionKind.EXPLOIT,
        action_name="exploit",
        target=(1, 0),
        success=success,
        cve_id="CVE-TEST",
        cvss_base=9.8,
        access_gained=access,
        is_crown_jewel=crown,
    )


def scan_event() -> AttackEvent:
    return AttackEvent(
        step=1,
        kind=ActionKind.SERVICE_SCAN,
        action_name="service_scan",
        target=(1, 0),
        success=True,
        newly_discovered=1,
    )


def test_legacy_reward_hash_is_unchanged() -> None:
    assert RewardConfig.from_yaml(ROOT / "configs/shaped.yaml").hash() == "8aa4aee9074f423d"


def test_explicit_zero_model_is_distinct_from_legacy_model() -> None:
    legacy = RewardConfig()
    explicit = RewardConfig(components=RewardComponents())
    assert legacy.hash() != explicit.hash()
    assert RewardEngine(explicit).score(exploit_event(crown=True)).total == 0.0


@pytest.mark.parametrize(
    ("components", "event", "field"),
    [
        (RewardComponents(cvss_weighting=True), exploit_event(), "cve"),
        (RewardComponents(mitre_tactic_shaping=True), exploit_event(), "tactic"),
        (
            RewardComponents(informative_success_shaping=True),
            scan_event(),
            "discovery",
        ),
        (
            RewardComponents(failure_penalty=True),
            exploit_event(success=False),
            "penalty",
        ),
        (
            RewardComponents(objective_reward=True),
            exploit_event(crown=True, access=AccessLevel.ROOT),
            "crown_jewel",
        ),
    ],
)
def test_each_component_can_be_enabled_in_isolation(
    components: RewardComponents, event, field: str
) -> None:
    result = RewardEngine(RewardConfig(components=components)).score(event)
    assert getattr(result, field) != 0.0
    terms = {
        "cve": result.cve,
        "tactic": result.tactic,
        "discovery": result.discovery,
        "penalty": result.penalty,
        "crown_jewel": result.crown_jewel,
    }
    assert [name for name, value in terms.items() if value] == [field]


def test_component_design_is_matched_and_changes_only_reward_switches() -> None:
    config = ComponentExperimentConfig.from_yaml(DESIGN)
    assert set(config.training_seeds).isdisjoint(config.evaluation_seeds)
    assert config.topology_seed == 42
    assert len(config.arms) == 7
    configs = [config.reward_config(arm) for arm in config.arms]
    for field in (
        "mode",
        "cve_scale",
        "tactic_bonuses",
        "crown_jewel",
        "failed_action",
        "weight",
        "first_success_only",
    ):
        assert len({repr(getattr(item, field)) for item in configs}) == 1
    RewardEngine(config.reward_config(config.full_arm)).assert_goal_dominance(8, 2)


def test_component_scoring_is_deterministic() -> None:
    config = RewardConfig(
        components=RewardComponents(
            cvss_weighting=True,
            mitre_tactic_shaping=True,
            informative_success_shaping=True,
            failure_penalty=True,
            objective_reward=True,
        )
    )
    events = [scan_event(), exploit_event(crown=True)]
    outputs = []
    for _ in range(2):
        engine = RewardEngine(config)
        outputs.append([engine.score(event).as_row() for event in events])
    assert outputs[0] == outputs[1]


def test_protected_artifact_hash_excludes_only_component_outputs() -> None:
    config = ComponentExperimentConfig.from_yaml(DESIGN)
    assert protected_artifact_hash(config) == protected_artifact_hash(config)


def test_analysis_declares_and_corrects_the_full_metric_family() -> None:
    config = ComponentExperimentConfig.from_yaml(DESIGN)
    rows = []
    for arm_index, arm in enumerate(config.arms):
        for seed in config.training_seeds:
            rows.append(
                {
                    "arm": arm,
                    "training_seed": seed,
                    "success_rate": 0.5 + arm_index / 100,
                    "steps_to_goal": 100.0 - arm_index,
                    "native_reward": -100.0 + arm_index,
                    "discovery_coverage": 0.5,
                    "failed_actions": 10.0,
                    "mitre_coverage": 3.0,
                    "attack_path_steps": 4.0,
                    "attack_path_unique_targets": 2.0,
                }
            )
    report = analyse_metrics(config, rows)
    assert report["complete"]
    assert len(report["comparisons"]) == 48
