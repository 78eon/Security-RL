"""Frozen-policy evaluation and policy-boundary invariants."""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest

from rlredteam.evaluation import evaluate_checkpoint, evaluate_policy, write_bundle
from rlredteam.reward import RewardConfig, RewardMode
from rlredteam.topology import TopologyConfig


class ObservationOnlyPolicy:
    """A spy policy whose API cannot receive an environment or topology."""

    def __init__(self) -> None:
        self.observations = []
        self._next = 0

    def set_random_seed(self, seed: int) -> None:
        self._next = seed % 7

    def predict(self, observation, deterministic: bool = False):
        assert isinstance(observation, np.ndarray)
        assert observation.shape == (234,)
        self.observations.append(observation.copy())
        action = self._next
        self._next = (self._next + 1) % 120
        return action, None


def short_topology() -> TopologyConfig:
    return replace(TopologyConfig.from_yaml(), step_limit=8)


def test_policy_receives_only_partial_observations() -> None:
    policy = ObservationOnlyPolicy()
    bundle = evaluate_policy(
        policy,
        run_name="sparse-s42-t42",
        reward_mode="sparse",
        training_seed=42,
        evaluation_seeds=[1001],
        topology_seed=42,
        topology_config=short_topology(),
        reward_config=RewardConfig(mode=RewardMode.SPARSE),
    )

    initial = policy.observations[0].reshape(9, 26)
    assert np.count_nonzero(initial[1:8]) == 0
    assert bundle.episodes[0].terminal_reason in {"goal", "step_limit"}
    assert bundle.episodes[0].evaluation_seed == 1001


def test_evaluation_is_reproducible_for_same_policy_and_seed() -> None:
    kwargs = dict(
        run_name="sparse-s42-t42",
        reward_mode="sparse",
        training_seed=42,
        evaluation_seeds=[1001],
        topology_seed=42,
        topology_config=short_topology(),
        reward_config=RewardConfig(mode=RewardMode.SPARSE),
    )
    first = evaluate_policy(ObservationOnlyPolicy(), **kwargs)
    second = evaluate_policy(ObservationOnlyPolicy(), **kwargs)
    assert first == second


def test_evaluation_artifacts_are_machine_readable(tmp_path) -> None:
    bundle = evaluate_policy(
        ObservationOnlyPolicy(),
        run_name="shaped-s42-t42",
        reward_mode="shaped",
        training_seed=42,
        evaluation_seeds=[1001],
        topology_seed=42,
        topology_config=short_topology(),
        reward_config=RewardConfig(),
    )
    write_bundle(bundle, tmp_path, {"gradient_updates": False})

    assert (tmp_path / "evaluation.csv").read_text().startswith("run_name,")
    steps = [json.loads(line) for line in (tmp_path / "steps.jsonl").read_text().splitlines()]
    assert steps and {"action", "native_reward", "policy_reward"} <= steps[0].keys()
    assert json.loads((tmp_path / "evaluation_metadata.json").read_text()) == {
        "gradient_updates": False
    }


@pytest.mark.integration
@pytest.mark.slow
def test_short_checkpoint_evaluates_without_parameter_updates(tmp_path, monkeypatch) -> None:
    """Smoke the actual train -> freeze -> load -> evaluate boundary."""
    from rlredteam import train as train_module

    runs = tmp_path / "runs"
    monkeypatch.setattr(train_module, "RUNS_DIR", runs)
    monkeypatch.setenv("RLREDTEAM_GIT_DIRTY", "0")
    train_module.train(train_module.parse_args(["--seed", "42", "--timesteps", "2048"]))
    run_dir = runs / "shaped-s42-t42"
    out = tmp_path / "evaluation"

    bundle = evaluate_checkpoint(run_dir, [1001], out)
    metadata = json.loads((out / "evaluation_metadata.json").read_text())

    assert len(bundle.episodes) == 1
    assert metadata["gradient_updates"] is False
    assert metadata["policy_sha256_before"] == metadata["policy_sha256_after"]
