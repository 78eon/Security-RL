"""Controlled experiment configuration and evaluation-only analysis tests."""

from __future__ import annotations

import csv
from dataclasses import asdict
from pathlib import Path

import pytest

from rlredteam import experiment
from rlredteam.evaluation import EvaluationBundle, EvaluationEpisode, write_bundle
from rlredteam.experiment import ExperimentConfig, ExperimentError
from rlredteam.reward import RewardConfig

REPO_ROOT = Path(__file__).resolve().parents[1]


def config(tmp_path: Path, **overrides) -> ExperimentConfig:
    values = dict(
        experiment_id="test_experiment",
        description="test",
        arms=("sparse", "shaped"),
        topology_seed=42,
        training_seeds=(42,),
        evaluation_seeds=(1001,),
        training_timesteps=2048,
        postgres=False,
        log_steps=False,
        frozen_inputs=tmp_path / "frozen.json",
        metrics=REPO_ROOT / "configs" / "metrics.yaml",
    )
    values.update(overrides)
    return ExperimentConfig(**values)


def test_committed_experiment_pins_separate_train_and_evaluation_seeds() -> None:
    loaded = ExperimentConfig.from_yaml(
        REPO_ROOT / "configs" / "experiments" / "experiment_01.yaml"
    )
    assert loaded.training_seeds == tuple(range(42, 52))
    assert loaded.evaluation_seeds == tuple(range(1001, 1011))
    assert not set(loaded.training_seeds) & set(loaded.evaluation_seeds)


def test_overlapping_training_and_evaluation_seeds_are_rejected(tmp_path) -> None:
    with pytest.raises(ExperimentError, match="disjoint"):
        config(tmp_path, evaluation_seeds=(42,)).validate()


def test_reward_arms_differ_only_in_mode() -> None:
    sparse = asdict(RewardConfig.from_yaml(REPO_ROOT / "configs" / "sparse.yaml"))
    shaped = asdict(RewardConfig.from_yaml(REPO_ROOT / "configs" / "shaped.yaml"))
    assert sparse.pop("mode") != shaped.pop("mode")
    assert sparse == shaped


def _write_training(path: Path, native_return: float) -> None:
    path.parent.mkdir(parents=True)
    row = {
        "native_return": native_return,
        "shaped_return": native_return,
        "length": 1000,
        "goal_reached": False,
        "mean_cvss_exploited": "",
        "max_cvss_exploited": "",
        "hosts_compromised": 0,
    }
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)


def _write_evaluation(path: Path, run_name: str, arm: str, native_return: float) -> None:
    episode = EvaluationEpisode(
        run_name=run_name,
        reward_mode=arm,
        training_seed=42,
        evaluation_seed=1001,
        topology_seed=42,
        goal_reached=True,
        terminal_reason="goal",
        length=12,
        native_return=native_return,
        policy_return=1.0,
        hosts_compromised=2,
        discovered_hosts=3,
        mean_cvss_exploited=8.0,
        max_cvss_exploited=9.8,
        high_value_target_hit=True,
        distinct_actions=8,
        repeated_actions=1,
        tactics=("exploit", "recon"),
        techniques=("T1046", "T1210"),
    )
    write_bundle(EvaluationBundle((episode,), ()), path, {"gradient_updates": False})


def test_analysis_reads_evaluation_not_training_returns(tmp_path, monkeypatch) -> None:
    settings = config(tmp_path)
    runs = tmp_path / "runs"
    results = tmp_path / "results"
    monkeypatch.setattr(experiment, "RUNS_DIR", runs)
    for arm, value in (("sparse", 10.0), ("shaped", 20.0)):
        name = settings.run_name(arm, 42)
        _write_training(runs / name / "episodes.csv", -9999.0)
        _write_evaluation(results / "raw" / name, name, arm, value)

    metrics = experiment.collect_metrics(settings, results)

    assert [run.native_return for run in metrics] == [10.0, 20.0]
    assert all(not run.converged for run in metrics)
