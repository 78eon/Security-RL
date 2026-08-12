"""Training entry point -- smoke tests.

Marked `integration`: needs nasim and stable-baselines3. Kept to a very small
number of timesteps; these check the wiring, not that the agent learns.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("nasim")
pytest.importorskip("stable_baselines3")

from rlredteam.catalogue import CVECatalogue  # noqa: E402
from rlredteam.reward import RewardConfig, RewardMode  # noqa: E402
from rlredteam.topology import TopologyConfig  # noqa: E402
from rlredteam.train import (  # noqa: E402
    EpisodeCollector,
    build_env,
    config_digest,
    parse_args,
    set_all_seeds,
    summarise,
)

pytestmark = pytest.mark.integration

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def test_defaults_point_at_the_shaped_arm() -> None:
    args = parse_args([])
    assert args.seed == 42
    assert args.reward_config.name == "shaped.yaml"
    assert args.topology_seed is None  # falls back to --seed


def test_topology_seed_is_separable_from_training_seed() -> None:
    """They answer different questions and must be independently settable."""
    args = parse_args(["--seed", "7", "--topology-seed", "99"])
    assert (args.seed, args.topology_seed) == (7, 99)


def test_config_digest_is_stable_and_differs_per_file() -> None:
    shaped = config_digest(CONFIGS / "shaped.yaml")
    assert shaped == config_digest(CONFIGS / "shaped.yaml")
    assert shaped != config_digest(CONFIGS / "sparse.yaml")


def test_build_env_produces_a_steppable_vecenv() -> None:
    env = build_env(
        TopologyConfig.from_yaml(),
        RewardConfig.from_yaml(CONFIGS / "shaped.yaml"),
        CVECatalogue.open_default(),
        topology_seed=42,
        seed=42,
    )
    obs = env.reset()
    assert obs.shape == (1, *env.observation_space.shape)

    # SB3 hands down numpy ints; NASim asserts a Python int. Guards the coercion
    # in RewardWrapper.step that this whole entry point depends on.
    import numpy as np

    _, rewards, dones, infos = env.step(np.array([5], dtype=np.int64))
    assert len(rewards) == 1 and len(dones) == 1
    assert "native_reward" in infos[0]
    assert "reward_breakdown" in infos[0]


def test_set_all_seeds_makes_torch_deterministic() -> None:
    import torch

    set_all_seeds(42)
    first = torch.randn(8)
    set_all_seeds(42)
    assert torch.equal(first, torch.randn(8))
    assert torch.get_num_threads() == 1


def test_summarise_handles_no_episodes() -> None:
    report = summarise([], {"run_name": "x"})
    assert report["episodes"] == 0


def test_summarise_computes_success_rate_and_native_return() -> None:
    episodes = [
        {"goal_reached": True, "native_return": 10.0, "length": 100,
         "mean_cvss_exploited": 8.0},
        {"goal_reached": False, "native_return": -50.0, "length": 1000,
         "mean_cvss_exploited": 7.0},
    ]
    report = summarise(episodes, {})
    assert report["episodes"] == 2
    assert report["success_rate_overall"] == pytest.approx(0.5)
    assert report["mean_native_return_overall"] == pytest.approx(-20.0)


def test_collector_separates_goal_from_step_limit() -> None:
    """NASim terminates on goal and truncates on step limit; SB3 flattens both
    into `done`, so conflating them would corrupt the success-rate metric."""
    from rlredteam.events import AccessLevel, ActionKind, AttackEvent
    from rlredteam.reward import RewardBreakdown

    collector = EpisodeCollector(seed=42, topology_seed=42)
    collector.locals = {}

    event = AttackEvent(
        step=0, kind=ActionKind.EXPLOIT, action_name="e", target=(1, 0),
        success=True, access_gained=AccessLevel.USER, cvss_base=9.8,
        native_reward=5.0,
    )
    breakdown = RewardBreakdown(total=7.0, native=5.0)

    for truncated in (False, True):
        collector.locals = {
            "infos": [
                {
                    "attack_event": event,
                    "reward_breakdown": breakdown,
                    "native_reward": 5.0,
                    "TimeLimit.truncated": truncated,
                }
            ],
            "dones": [True],
        }
        collector.model = None
        collector.num_timesteps = 1
        collector._on_step()

    assert [e["terminal_state"] for e in collector.episodes] == ["goal", "step_limit"]
    assert [e["goal_reached"] for e in collector.episodes] == [True, False]


@pytest.mark.slow
def test_short_training_run_completes(tmp_path: Path, monkeypatch) -> None:
    """End-to-end: a tiny run trains, writes artefacts and reports."""
    from rlredteam import train as train_module

    monkeypatch.setattr(train_module, "RUNS_DIR", tmp_path)
    args = parse_args(["--seed", "42", "--timesteps", "2048"])
    report = train_module.train(args)

    out = tmp_path / "shaped-s42-t42"
    assert (out / "config.snapshot.json").exists()
    assert (out / "summary.json").exists()
    assert (out / "episodes.csv").exists()
    assert (out / "model.zip").exists()

    provenance = report["provenance"]
    assert provenance["reward_mode"] == str(RewardMode.SHAPED)
    assert len(provenance["cve_manifest_sha256"]) == 64
    assert provenance["topology_config_hash"]
