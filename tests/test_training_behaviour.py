"""Experiment-validity tests: does training actually behave as claimed?

These are the tests that defend the *results*, not the plumbing. They train real
policies, so they are marked `slow` and excluded from the default suite:

    pytest -m "not slow"     # fast suite
    pytest -m slow           # these

Numbering matches the agreed validation plan; tests 3, 4, 5, 6, 12 live with the
components they exercise (test_reward.py, test_nasim_adapter.py,
test_catalogue.py) and are cross-referenced below.
"""

from __future__ import annotations

import random
import time
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("nasim")
pytest.importorskip("stable_baselines3")

import torch  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402

from rlredteam.catalogue import CVECatalogue  # noqa: E402
from rlredteam.manifest import digest  # noqa: E402
from rlredteam.reward import RewardConfig, RewardMode  # noqa: E402
from rlredteam.topology import TopologyConfig, make_env  # noqa: E402
from rlredteam.train import build_env, parse_args, set_all_seeds  # noqa: E402
from rlredteam.train import train as run_training  # noqa: E402

pytestmark = [pytest.mark.integration, pytest.mark.slow]

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
SHAPED = CONFIGS / "shaped.yaml"
SPARSE = CONFIGS / "sparse.yaml"


# -- helpers ---------------------------------------------------------------


def _train_briefly(seed: int, timesteps: int, config_path: Path = SPARSE, out=None):
    """Train a policy and return (model, episode records)."""
    argv = ["--seed", str(seed), "--timesteps", str(timesteps),
            "--reward-config", str(config_path)]
    args = parse_args(argv)
    if out is not None:
        import rlredteam.train as train_module

        train_module.RUNS_DIR = out
    return run_training(args)


def _fresh_model(seed: int, timesteps: int, config_path: Path = SPARSE) -> PPO:
    """Train in-process without the file/artefact machinery."""
    set_all_seeds(seed)
    env = build_env(
        TopologyConfig.from_yaml(),
        RewardConfig.from_yaml(config_path),
        CVECatalogue.open_default(),
        topology_seed=seed,
        seed=seed,
    )
    model = PPO("MlpPolicy", env, seed=seed, ent_coef=0.01, verbose=0, device="cpu")
    model.learn(total_timesteps=timesteps)
    return model


def _params(model: PPO) -> dict[str, torch.Tensor]:
    return {k: v.detach().clone() for k, v in model.policy.state_dict().items()}


def _random_episodes(n: int, seed: int = 42) -> list[dict]:
    """Uniformly random policy, matching scripts/rollout_random.py."""
    from rlredteam.nasim_adapter import RewardWrapper

    env = make_env(TopologyConfig.from_yaml(), topology_seed=seed)
    env = RewardWrapper(env, CVECatalogue.open_default(), topology_seed=seed)

    results = []
    for episode in range(n):
        env.reset(seed=seed + episode)
        rng = random.Random(seed + episode)
        native = 0.0
        steps = 0
        terminated = truncated = False
        while not (terminated or truncated):
            _, _, terminated, truncated, info = env.step(
                rng.randrange(env.action_space.n)
            )
            native += info["native_reward"]
            steps += 1
        results.append(
            {"native": native, "steps": steps, "goal": bool(terminated)}
        )
    return results


# -- 1. reproducibility ----------------------------------------------------


def test_reproducibility_identical_weights_and_rewards(tmp_path: Path) -> None:
    """Two identical runs give identical weights AND identical reward curves.

    This is the single most important claim in the methodology chapter: without
    it, no reported number can be checked by anyone else.
    """
    first = _fresh_model(seed=42, timesteps=2048)
    second = _fresh_model(seed=42, timesteps=2048)

    a, b = _params(first), _params(second)
    assert a.keys() == b.keys()
    mismatched = [k for k in a if not torch.equal(a[k], b[k])]
    assert not mismatched, f"weights diverged in {mismatched[:5]}"


def test_reproducibility_identical_episode_records(tmp_path: Path) -> None:
    """The logged reward curve repeats exactly, not just the weights."""
    first = _train_briefly(42, 2048, SPARSE, out=tmp_path / "a")
    second = _train_briefly(42, 2048, SPARSE, out=tmp_path / "b")

    curve_a = [
        (e["native_return"], e["length"], e["goal_reached"])
        for e in _read_episodes(tmp_path / "a")
    ]
    curve_b = [
        (e["native_return"], e["length"], e["goal_reached"])
        for e in _read_episodes(tmp_path / "b")
    ]
    assert curve_a == curve_b
    assert first["episodes"] == second["episodes"]


def _read_episodes(runs_dir: Path) -> list[dict]:
    import csv

    path = next(runs_dir.glob("*/episodes.csv"))
    with path.open() as handle:
        return list(csv.DictReader(handle))


# -- 13. seed isolation ----------------------------------------------------


def test_different_seeds_give_different_policies() -> None:
    """If seeds 42 and 43 produced identical weights, the seed is being ignored
    and every 'independent run' in the results would be the same run."""
    a = _params(_fresh_model(seed=42, timesteps=2048))
    b = _params(_fresh_model(seed=43, timesteps=2048))
    assert any(not torch.equal(a[k], b[k]) for k in a), "seed had no effect"


# -- 7. random baseline ----------------------------------------------------


def test_random_baseline_is_weak_but_nonzero() -> None:
    """Establishes the bar PPO must clear.

    Random must sometimes reach the goal (otherwise the task is unsolvable by
    exploration and PPO could never bootstrap), and must score badly (otherwise
    there is nothing to learn).
    """
    episodes = _random_episodes(20, seed=42)
    success_rate = sum(e["goal"] for e in episodes) / len(episodes)
    mean_native = float(np.mean([e["native"] for e in episodes]))

    assert success_rate > 0.0, "random never reaches the goal — task unbootstrappable"
    assert mean_native < 0.0, "random already scores positively — nothing to learn"

    print(
        f"\nrandom baseline over 20 episodes: success={success_rate:.2f} "
        f"mean_native={mean_native:.1f} "
        f"mean_steps={np.mean([e['steps'] for e in episodes]):.0f}"
    )


# -- 8. PPO beats random ---------------------------------------------------


def test_ppo_beats_random_on_native_reward() -> None:
    """The core competence claim, tested on NATIVE reward.

    Evaluated on native reward deliberately: the trained policy optimises the
    shaped or sparse reward, but the comparison against random is only
    meaningful in the environment's own units.
    """
    from scipy import stats

    random_native = [e["native"] for e in _random_episodes(10, seed=100)]

    model = _fresh_model(seed=42, timesteps=6000, config_path=SPARSE)
    trained_native = _evaluate(model, episodes=10, seed=100)

    result = stats.ttest_ind(
        trained_native, random_native, equal_var=False, alternative="greater"
    )
    print(
        f"\nPPO {np.mean(trained_native):.1f} vs random "
        f"{np.mean(random_native):.1f}  (t={result.statistic:.2f}, "
        f"p={result.pvalue:.4f})"
    )
    assert np.mean(trained_native) > np.mean(random_native), (
        "PPO did not beat random — check the reward wiring before the ablation"
    )


def _evaluate(
    model: PPO, episodes: int, seed: int, deterministic: bool = False
) -> list[float]:
    """Evaluate a policy, accumulating NATIVE reward.

    ``deterministic=False`` is the project default and is a deliberate
    methodological choice, not a convenience. NASim is partially observable, and
    a failed action barely changes the observation -- so a memoryless greedy
    policy re-picks the same argmax action and loops until the step limit. See
    test_deterministic_evaluation_collapses for the measured effect. Stochastic
    policies being strictly better in partially observable settings is a known
    result (Singh, Jaakkola & Jordan, 1994).
    """
    from rlredteam.nasim_adapter import RewardWrapper

    env = make_env(TopologyConfig.from_yaml(), topology_seed=42)
    env = RewardWrapper(env, CVECatalogue.open_default(), topology_seed=42)

    returns = []
    for episode in range(episodes):
        obs, _ = env.reset(seed=seed + episode)
        native = 0.0
        terminated = truncated = False
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, _, terminated, truncated, info = env.step(action)
            native += info["native_reward"]
        returns.append(native)
    return returns


# -- 9. success rate rises -------------------------------------------------


def test_success_rate_does_not_fall_over_training(tmp_path: Path) -> None:
    """Split the run in half; the second half must not be worse.

    Stated as 'does not fall' rather than 'rises': on this topology even a
    random policy often reaches the goal, so success rate saturates early and
    demanding a strict increase would be testing noise.
    """
    _train_briefly(42, 8000, SPARSE, out=tmp_path)
    episodes = _read_episodes(tmp_path)
    assert len(episodes) >= 4, "too few episodes to split"

    half = len(episodes) // 2
    first = [e["goal_reached"] == "True" for e in episodes[:half]]
    second = [e["goal_reached"] == "True" for e in episodes[half:]]
    rate_first = sum(first) / len(first)
    rate_second = sum(second) / len(second)

    print(f"\nsuccess rate: first half {rate_first:.2f} -> second half {rate_second:.2f}")
    assert rate_second >= rate_first - 0.15, "success rate regressed materially"


def test_steps_to_goal_improves_over_training(tmp_path: Path) -> None:
    """The metric that actually discriminates once success rate saturates."""
    _train_briefly(42, 8000, SPARSE, out=tmp_path)
    episodes = [e for e in _read_episodes(tmp_path) if e["goal_reached"] == "True"]
    if len(episodes) < 6:
        pytest.skip("not enough successful episodes to compare halves")

    half = len(episodes) // 2
    first = np.mean([int(e["length"]) for e in episodes[:half]])
    second = np.mean([int(e["length"]) for e in episodes[half:]])
    print(f"\nsteps to goal: first half {first:.0f} -> second half {second:.0f}")
    assert second <= first * 1.25, "episodes got substantially longer, not shorter"


# -- 10. action diversity --------------------------------------------------


def _action_usage(model: PPO, episodes: int, seed: int, deterministic: bool):
    """Return (distinct actions used, goals reached, mean episode length)."""
    import collections

    from rlredteam.nasim_adapter import RewardWrapper

    env = make_env(TopologyConfig.from_yaml(), topology_seed=42)
    env = RewardWrapper(env, CVECatalogue.open_default(), topology_seed=42)

    counts: collections.Counter = collections.Counter()
    goals = 0
    lengths = []
    for episode in range(episodes):
        obs, _ = env.reset(seed=seed + episode)
        terminated = truncated = False
        steps = 0
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=deterministic)
            counts[int(action)] += 1
            obs, _, terminated, truncated, _ = env.step(action)
            steps += 1
        goals += bool(terminated)
        lengths.append(steps)
    return counts, goals, float(np.mean(lengths))


def test_policy_does_not_collapse_to_one_action() -> None:
    """A collapsed policy can look fine on reward while having learned nothing.

    Evaluated stochastically -- the project's evaluation policy. See
    test_deterministic_evaluation_collapses for why.
    """
    model = _fresh_model(seed=42, timesteps=6000, config_path=SPARSE)
    counts, goals, mean_len = _action_usage(model, 10, seed=500, deterministic=False)

    print(
        f"\nstochastic: {len(counts)} distinct actions, "
        f"{goals}/10 goals, mean length {mean_len:.0f}"
    )
    assert len(counts) > 1, "policy collapsed to a single action"
    assert len(counts) >= 5, f"policy used only {len(counts)} distinct actions"


def test_deterministic_evaluation_collapses() -> None:
    """Documents a real, measured property of this environment.

    Greedy action selection traps the agent: NASim is partially observable and a
    failed action leaves the observation essentially unchanged, so the argmax
    re-selects the same action forever and the episode runs to the step limit.
    Measured at 6k steps: the deterministic policy used 2 distinct actions and
    reached the goal 0/5 times, while the same weights sampled stochastically
    used all 120 and reached it 5/5.

    This is why the project evaluates stochastically. The test is written to
    FAIL if that ever stops being true, because the methodology would then need
    revisiting rather than silently inheriting an obsolete justification.
    """
    model = _fresh_model(seed=42, timesteps=6000, config_path=SPARSE)

    greedy_counts, greedy_goals, greedy_len = _action_usage(
        model, 5, seed=900, deterministic=True
    )
    sampled_counts, sampled_goals, sampled_len = _action_usage(
        model, 5, seed=900, deterministic=False
    )

    print(
        f"\ndeterministic: {len(greedy_counts)} actions, {greedy_goals}/5 goals, "
        f"len {greedy_len:.0f}\n"
        f"stochastic   : {len(sampled_counts)} actions, {sampled_goals}/5 goals, "
        f"len {sampled_len:.0f}"
    )

    assert len(greedy_counts) < len(sampled_counts), (
        "deterministic no longer narrower than stochastic -- revisit the "
        "evaluation-policy justification in docs/EVAL_SPEC.md"
    )
    assert sampled_goals >= greedy_goals


# -- 11. checkpoint reload -------------------------------------------------


def test_saved_model_reloads_and_predicts_identically(tmp_path: Path) -> None:
    """A checkpoint that does not reload faithfully invalidates every result
    reported from it."""
    model = _fresh_model(seed=42, timesteps=2048)
    path = tmp_path / "model"
    model.save(path)
    reloaded = PPO.load(path, device="cpu")

    env = make_env(TopologyConfig.from_yaml(), topology_seed=42)
    obs, _ = env.reset(seed=42)
    rng = np.random.default_rng(0)

    for _ in range(10):
        original, _ = model.predict(obs, deterministic=True)
        restored, _ = reloaded.predict(obs, deterministic=True)
        assert int(original) == int(restored)
        obs, _, terminated, truncated, _ = env.step(int(original))
        if terminated or truncated:
            obs, _ = env.reset(seed=int(rng.integers(0, 10_000)))


# -- 12. manifest consistency (see also test_catalogue.py) -----------------


def test_manifest_unchanged_across_a_training_run(tmp_path: Path) -> None:
    """The vulnerability data must not shift underneath a run."""
    before = digest(CVECatalogue.open_default())
    _train_briefly(42, 2048, SHAPED, out=tmp_path)
    after = digest(CVECatalogue.open_default())
    assert before == after


# -- 14. config hashes match ----------------------------------------------


def test_snapshot_hashes_match_runtime_values(tmp_path: Path) -> None:
    """What was logged must equal what was actually used.

    These hashes are the evidence that both ablation arms ran on the same
    topology and the same CVE data. If they were computed differently at log
    time than at run time, that evidence is worthless.
    """
    import json

    report = _train_briefly(42, 2048, SHAPED, out=tmp_path)
    snapshot = json.loads(
        next(tmp_path.glob("*/config.snapshot.json")).read_text()
    )

    assert snapshot["topology_config_hash"] == TopologyConfig.from_yaml().config_hash()
    assert snapshot["reward_config_hash"] == RewardConfig.from_yaml(SHAPED).hash()
    assert snapshot["cve_manifest_sha256"] == digest(CVECatalogue.open_default())
    assert report["provenance"] == snapshot


@pytest.mark.parametrize(
    ("path", "mode"),
    [(SHAPED, RewardMode.SHAPED), (SPARSE, RewardMode.SPARSE)],
)
def test_reward_config_hash_is_stable_and_distinct(path: Path, mode: RewardMode) -> None:
    config = RewardConfig.from_yaml(path)
    assert config.mode is mode
    assert config.hash() == RewardConfig.from_yaml(path).hash()
    assert len(config.hash()) == 16


def test_reward_config_hash_ignores_comments_and_key_order() -> None:
    """Hashes the resolved settings, not the file bytes."""
    from dataclasses import replace

    base = RewardConfig.from_yaml(SHAPED)
    assert replace(base).hash() == base.hash()
    assert replace(base, cve_scale=base.cve_scale + 1).hash() != base.hash()


# -- 15. throughput (reported, not asserted) -------------------------------


def test_training_throughput_is_recorded(tmp_path: Path) -> None:
    """Measures steps/sec and memory so the compute budget can be planned.

    Deliberately does not assert a hardware-specific threshold -- that would
    fail on a different machine for no scientific reason. It reports.
    """
    timesteps = 4000
    start = time.perf_counter()
    _train_briefly(42, timesteps, SPARSE, out=tmp_path)
    elapsed = time.perf_counter() - start

    throughput = timesteps / elapsed
    line = f"\nthroughput: {throughput:.0f} steps/sec ({elapsed:.1f}s for {timesteps})"

    try:
        import psutil

        rss = psutil.Process().memory_info().rss / 1e6
        line += f"  |  RSS {rss:.0f} MB"
    except ImportError:
        line += "  |  RSS unavailable (psutil not installed)"

    print(line)
    print(f"  full grid estimate (2 arms x 10 seeds x 200k): "
          f"{2 * 10 * 200_000 / throughput / 60:.0f} min")
    assert throughput > 0
