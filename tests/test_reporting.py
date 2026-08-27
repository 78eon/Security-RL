"""Trace-derived attack paths and trajectory validation."""

from rlredteam.reporting import extract_attack_path, validate_trajectory


def episode(**overrides) -> dict:
    row = {
        "run_name": "experiment_01-shaped-s42-t42",
        "reward_mode": "shaped",
        "training_seed": "42",
        "evaluation_seed": "1001",
        "goal_reached": "True",
        "terminal_reason": "goal",
        "length": "3",
        "native_return": "10.0",
        "policy_return": "105.0",
    }
    row.update(overrides)
    return row


def step(index: int, **overrides) -> dict:
    row = {
        "step": index,
        "action": "service_scan_(1, 0)",
        "action_kind": "service_scan",
        "target": [1, 0],
        "success": True,
        "native_reward": 0.0,
        "policy_reward": 1.0,
        "newly_discovered": 1,
        "access_gained": 0,
        "cve_id": None,
        "cvss_base": None,
        "is_crown_jewel": False,
        "tactic": "recon",
        "technique_id": "T1046",
        "error": None,
    }
    row.update(overrides)
    return row


def test_attack_path_contains_only_observed_successful_progress() -> None:
    steps = [
        step(1),
        step(2, action="failed", success=False, newly_discovered=0, policy_reward=-5.0),
        step(
            3,
            action="exploit_(1, 0)",
            newly_discovered=0,
            access_gained=1,
            cve_id="CVE-2021-42013",
            cvss_base=9.8,
            tactic="exploit",
            technique_id="T1210",
            is_crown_jewel=True,
        ),
        step(4, action="no_progress", newly_discovered=0, access_gained=0),
    ]

    path = extract_attack_path(episode(), steps)

    assert [item.step for item in path.progress_steps] == [1, 3]
    assert path.progress_steps[-1].cve_id == "CVE-2021-42013"
    assert path.progress_steps[-1].is_crown_jewel is True


def test_trajectory_validation_flags_paid_errors_and_repeats() -> None:
    steps = [
        step(1),
        step(2, error="invalid", policy_reward=1.0),
        step(3, error="invalid", policy_reward=1.0),
    ]

    report = validate_trajectory(episode(goal_reached="False"), steps)

    assert report["longest_identical_action_streak"] == 3
    assert report["repeated_action_targets_with_positive_reward"] == 2
    assert report["invalid_or_impossible_actions_with_positive_reward"] == 2
    assert report["high_policy_return_without_goal"] is True
