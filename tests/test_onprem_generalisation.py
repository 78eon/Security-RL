"""Phase 3 mask-aware training/evaluation and provenance invariants."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from rlredteam.enterprise.generalisation import (
    GeneralisationError,
    OnPremExperimentConfig,
    distribution_manifest,
    evaluate_policy,
    reconstruct_attack_path,
    write_evaluation_package,
)
from rlredteam.enterprise.onprem import (
    OnPremCurriculumEnv,
    OnPremGeneralisationSplit,
    OnPremTopologyConfig,
)


class HighestValidPolicy:
    """Test policy: observes only the public observation/mask boundary."""

    def __init__(self) -> None:
        self.policy = torch.nn.Linear(1, 1, bias=False)

    def set_random_seed(self, seed: int) -> None:
        self.seed = seed

    def predict(self, observation, *, action_masks, deterministic: bool):
        del observation, deterministic
        return int(np.flatnonzero(action_masks)[-1]), None


def test_action_masks_are_boolean_and_do_not_read_true_topology() -> None:
    env = OnPremCurriculumEnv((1,))
    env.reset(seed=77, options={"topology_seed": 1})
    expected = env.action_masks().copy()

    class ForbiddenTruth:
        def __getattribute__(self, name):
            raise AssertionError(f"action mask read ground truth: {name}")

    env._env.true_topology = ForbiddenTruth()
    actual = env.action_masks()
    assert actual.dtype == np.bool_
    assert np.array_equal(actual, expected)


def test_experiment_configuration_is_complete_and_hashed() -> None:
    config = OnPremExperimentConfig.from_yaml()
    assert config.total_timesteps > 0
    assert len(config.digest()) == 64
    assert config.evaluation_episode_seeds == (9001,)


def test_distribution_manifest_covers_every_disjoint_seed() -> None:
    split = OnPremGeneralisationSplit()
    manifest = distribution_manifest(split, OnPremTopologyConfig.from_yaml())
    assert set(map(int, manifest["train"])) == set(split.train)
    assert set(map(int, manifest["validation"])) == set(split.validation)
    assert set(map(int, manifest["test"])) == set(split.test)
    assert all(len(value) == 64 for group in manifest.values() for value in group.values())


def test_evaluation_refuses_training_topology_overlap() -> None:
    with pytest.raises(GeneralisationError, match="overlap"):
        evaluate_policy(
            HighestValidPolicy(),
            topology_seeds=(1,),
            evaluation_episode_seeds=(9001,),
        )


def test_unseen_evaluation_is_masked_frozen_and_reconstructable() -> None:
    model = HighestValidPolicy()
    before = model.policy.weight.detach().clone()
    episodes, steps = evaluate_policy(
        model,
        topology_seeds=(1001,),
        evaluation_episode_seeds=(9001,),
    )
    assert len(episodes) == 1
    assert episodes[0]["goal_reached"]
    assert episodes[0]["invalid_mask_selections"] == 0
    assert steps[-1]["goal_reached"]
    assert all(step["target_entity"] for step in steps)
    assert any(step["prerequisites"] for step in steps)
    assert any(step["outcomes"] for step in steps)
    causal = reconstruct_attack_path(steps)
    assert causal[-1]["action_kind"] == "access_asset"
    assert any(step["action_kind"] == "exploit" for step in causal)
    assert len(causal) < len(steps)
    assert torch.equal(before, model.policy.weight)


def test_evaluation_package_contains_raw_trajectory_and_summary(tmp_path) -> None:
    episodes, steps = evaluate_policy(
        HighestValidPolicy(),
        topology_seeds=(1002,),
        evaluation_episode_seeds=(9002,),
    )
    write_evaluation_package(
        tmp_path,
        episodes=episodes,
        steps=steps,
        metadata={"gradient_updates": False},
    )
    assert (tmp_path / "episodes.csv").is_file()
    assert (tmp_path / "attack_paths.json").is_file()
    assert len((tmp_path / "trajectories.jsonl").read_text().splitlines()) == len(steps)
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["success_rate"] == 1.0
    assert summary["invalid_mask_selections"] == 0
