from __future__ import annotations

from pathlib import Path

import pytest

from rlredteam.enterprise.multiagent import MultiAgentResearchConfig
from rlredteam.enterprise.multiagent_study import (
    aggregate_seed_metrics,
    analyse_seed_metrics,
    train_defender,
    train_red_arm,
    validate_paired_red_training_isolation,
)


def _episode(arm: str, seed: int, profile: str, topology_seed: int) -> dict:
    success = seed % 3 != 0
    length = 10 + seed % 5
    return {
        "arm": arm,
        "training_seed": seed,
        "profile": profile,
        "topology_seed": topology_seed,
        "evaluation_seed": 9601,
        "goal_reached": success,
        "steps_to_goal": length if success else None,
        "total_reward": float(100 - length),
        "native_total_reward": float(110 - length),
        "detected_actions": 2 + int(arm == "adaptive_defender_training"),
        "detectable_actions": length,
        "exploit_attempts": 2,
        "mitigated_exploits": int(arm == "adaptive_defender_training"),
        "discovery_coverage": 0.8,
        "failed_actions": 1,
    }


def test_seed_statistics_are_reconstructed_from_complete_evaluation_grid() -> None:
    config = MultiAgentResearchConfig.from_yaml()
    topology_seeds = (15001,)
    episodes = [
        _episode(arm, seed, profile.value, topology_seeds[0])
        for arm in config.arms
        for seed in config.training_seeds
        for profile in config.train_profiles
    ]
    rows = aggregate_seed_metrics(
        episodes, config, expected_topology_seeds=topology_seeds
    )
    report = analyse_seed_metrics(rows, config)
    assert len(rows) == 20
    assert report["complete"]
    assert {item["metric"] for item in report["comparisons"]} == set(
        config.primary_metrics + config.descriptive_metrics
    )
    primary = [
        item for item in report["comparisons"] if item["metric"] in config.primary_metrics
    ]
    assert all(item["n_pairs"] == 10 for item in primary)
    assert all(item["p_bonferroni"] is not None for item in primary)


@pytest.mark.slow
def test_short_alternating_training_updates_only_the_optimised_policy(
    tmp_path: Path,
) -> None:
    config = MultiAgentResearchConfig.from_yaml()
    static_dir = tmp_path / "static"
    static = train_red_arm(
        static_dir,
        arm=config.arms[0],
        training_seed=config.development_seed,
        defender_checkpoint=tmp_path / "unused.zip",
        config=config,
        timesteps=256,
        development=True,
        allow_dirty=True,
    )
    assert static["policy_sha256_before_training"] != static["policy_sha256"]

    defender_dir = tmp_path / "defender"
    defender = train_defender(
        defender_dir,
        bootstrap_red_checkpoint=static_dir / "model.zip",
        config=config,
        timesteps=256,
        allow_dirty=True,
    )
    assert defender["policy_sha256_before_training"] != defender["policy_sha256"]
    assert defender["bootstrap_red_policy_sha256_before"] == defender[
        "bootstrap_red_policy_sha256_after"
    ]

    adaptive = train_red_arm(
        tmp_path / "adaptive",
        arm=config.arms[1],
        training_seed=config.development_seed,
        defender_checkpoint=defender_dir / "model.zip",
        config=config,
        timesteps=256,
        development=True,
        allow_dirty=True,
    )
    assert adaptive["policy_sha256_before_training"] != adaptive["policy_sha256"]
    assert adaptive["defender_policy_sha256_before"] == adaptive[
        "defender_policy_sha256_after"
    ]
    validate_paired_red_training_isolation(static, adaptive)
