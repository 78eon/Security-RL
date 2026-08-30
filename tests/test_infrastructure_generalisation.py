"""Phase 4 cross-profile experiment and frozen-evaluation invariants."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from rlredteam.enterprise.generalisation import (
    GeneralisationError,
    reconstruct_attack_path,
    write_evaluation_package,
)
from rlredteam.enterprise.infrastructure_generalisation import (
    InfrastructureExperimentConfig,
    evaluate_infrastructure_policy,
    infrastructure_distribution_manifest,
)
from rlredteam.enterprise.onprem import OnPremGeneralisationSplit
from rlredteam.enterprise.profiles import DeploymentProfile, EnterpriseProfileConfig

PROFILES = (
    DeploymentProfile.LEGACY,
    DeploymentProfile.CLOUD,
    DeploymentProfile.HYBRID,
)


class HighestValidPolicy:
    def __init__(self) -> None:
        self.policy = torch.nn.Linear(1, 1, bias=False)

    def set_random_seed(self, seed: int) -> None:
        self.seed = seed

    def predict(self, observation, *, action_masks, deterministic: bool):
        del observation, deterministic
        return int(np.flatnonzero(action_masks)[-1]), None


def test_experiment_configuration_preregisters_every_profile() -> None:
    config = InfrastructureExperimentConfig.from_yaml()
    assert config.train_profiles == PROFILES
    assert config.total_timesteps == 50176
    assert config.total_timesteps % config.ppo["n_steps"] == 0
    assert len(config.digest()) == 64


def test_distribution_manifest_covers_profile_seed_product() -> None:
    split = OnPremGeneralisationSplit()
    manifest = infrastructure_distribution_manifest(
        split,
        EnterpriseProfileConfig.from_yaml(),
        PROFILES,
    )
    assert set(manifest) == {"train", "validation", "test"}
    for split_name in manifest:
        assert set(manifest[split_name]) == {profile.value for profile in PROFILES}
        assert all(
            len(digest) == 64
            for profile in manifest[split_name].values()
            for digest in profile.values()
        )


def test_cross_profile_evaluation_refuses_training_seed_overlap() -> None:
    with pytest.raises(GeneralisationError, match="overlap"):
        evaluate_infrastructure_policy(
            HighestValidPolicy(),
            profiles=PROFILES,
            topology_seeds=(1,),
            evaluation_episode_seeds=(9101,),
        )


def test_cross_profile_evaluation_is_frozen_masked_and_distinct() -> None:
    model = HighestValidPolicy()
    before = model.policy.weight.detach().clone()
    episodes, steps = evaluate_infrastructure_policy(
        model,
        profiles=PROFILES,
        topology_seeds=(1001,),
        evaluation_episode_seeds=(9101,),
    )
    assert len(episodes) == 3
    assert {row["profile"] for row in episodes} == {profile.value for profile in PROFILES}
    assert all(row["goal_reached"] for row in episodes)
    assert all(row["invalid_mask_selections"] == 0 for row in episodes)
    assert len({row["topology_hash"] for row in episodes}) == 3
    assert torch.equal(before, model.policy.weight)

    for profile in PROFILES:
        profile_steps = [row for row in steps if row["profile"] == profile.value]
        causal = reconstruct_attack_path(profile_steps)
        assert causal[-1]["action_kind"] == "access_asset"
        assert any(row["action_kind"] == "exploit" for row in causal)


def test_package_keeps_same_seed_profiles_separate(tmp_path) -> None:
    episodes, steps = evaluate_infrastructure_policy(
        HighestValidPolicy(),
        profiles=PROFILES,
        topology_seeds=(1002,),
        evaluation_episode_seeds=(9102,),
    )
    write_evaluation_package(
        tmp_path,
        episodes=episodes,
        steps=steps,
        metadata={"gradient_updates": False},
    )
    paths = json.loads((tmp_path / "attack_paths.json").read_text())
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert len(paths) == 3
    assert {path["profile"] for path in paths} == {profile.value for profile in PROFILES}
    assert summary["success_rate_by_profile"] == {
        "cloud": 1.0,
        "hybrid": 1.0,
        "legacy": 1.0,
    }

