"""Convergence decisions are reproducible protocol decisions, not plot judgements."""

import copy
import subprocess

import pytest

from rlredteam.convergence import (
    BASE_CONFIG,
    ROOT,
    SEEDS,
    ConvergenceError,
    assess_all,
    assess_seed,
    diagnostic_advice,
    diagnostic_row,
    digest,
    load_config,
    load_criterion,
    validate_child,
    validate_config,
    write_new,
)


@pytest.fixture
def baseline():
    return load_config(BASE_CONFIG)


@pytest.fixture
def criterion():
    return load_criterion(ROOT / "configs/convergence_criterion_v1.yaml")


def evidence(collapse=False):
    episodes = []
    for i in range(1, 301):
        value = min(100.0, i * 2.0)
        if collapse and i > 275:
            value = -100.0
        episodes.append({"timesteps": i * 100, "shaped_return": value})
    rows = [{"timesteps": i * 1000, "explained_variance": 0.2 + i / 100} for i in range(1, 31)]
    return episodes, rows


def test_exact_baseline_and_normalization(baseline):
    from rlredteam.train import PPO_DEFAULTS

    assert baseline["ppo"] == PPO_DEFAULTS | {"normalize_advantage": True}
    assert baseline["ppo"]["batch_size"] == 64
    assert baseline["ppo"]["n_epochs"] == 10
    normalized = load_config(
        ROOT / "configs/experiments/experiment_01_convergence_normalized_1m.yaml"
    )
    validate_child(baseline, normalized)
    assert normalized["ppo"] == baseline["ppo"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("reward", "sparse"),
        ("training_seeds", [42]),
        ("topology_seed", 43),
        ("total_timesteps", 40000),
        ("id", "../../experiment_01"),
    ],
)
def test_invalid_protocol_fails(baseline, field, value):
    baseline[field] = value
    with pytest.raises(ConvergenceError):
        validate_config(baseline)


def test_baseline_hyperparameter_drift_rejected(baseline):
    baseline["ppo"]["learning_rate"] = 1e-4
    with pytest.raises(ConvergenceError):
        validate_config(baseline)


def test_unknown_fields_rejected(baseline):
    baseline["automatic_sweep"] = True
    with pytest.raises(ConvergenceError):
        validate_config(baseline)


def test_single_factor_and_order():
    parent = load_config(ROOT / "configs/experiments/experiment_01_convergence_normalized_1m.yaml")
    child = copy.deepcopy(parent)
    child.update(
        id="lr_candidate", stage="tuning", factor="learning_rate", rationale="KL and clipping high"
    )
    child["ppo"]["learning_rate"] = 1e-4
    child["learning_rate_schedule"] = "linear_to_zero"
    validate_child(parent, child)
    child["ppo"]["ent_coef"] = 0.02
    with pytest.raises(ConvergenceError, match="multi-factor"):
        validate_child(parent, child)
    child = copy.deepcopy(parent)
    child.update(stage="tuning", factor="ent_coef", rationale="entropy low")
    child["ppo"]["ent_coef"] = 0.02
    with pytest.raises(ConvergenceError, match="order"):
        validate_child(parent, child)


def test_diagnostics_explicit_missing_and_raw_reward():
    row = diagnostic_row(
        {"train/approx_kl": 0.02, "train/explained_variance": float("nan")},
        update=2,
        timestep=4096,
        episodes=[{"shaped_return": 100, "length": 8}],
    )
    assert row["approx_kl"] == 0.02
    assert row["explained_variance"] is None
    assert row["clip_fraction"] is None
    assert row["mean_episode_reward"] == 100
    assert row["mean_episode_length"] == 8


def test_assessment_deterministic_and_complete(criterion):
    episodes, rows = evidence()
    first = assess_seed(episodes, rows, actual_steps=30000, budget=30000, criterion=criterion)
    assert first["passed"], first
    assert first == assess_seed(
        episodes, rows, actual_steps=30000, budget=30000, criterion=criterion
    )
    result = assess_all({str(s): first for s in SEEDS}, criterion)
    assert result["passed"]
    assert result["criterion_sha256"] == digest(criterion)


@pytest.mark.parametrize(
    "case", ["collapse", "missing_ev", "short_budget", "no_rise", "negative_ev"]
)
def test_failed_criteria(criterion, case):
    episodes, rows = evidence(case == "collapse")
    if case == "missing_ev":
        rows = []
    elif case == "no_rise":
        for row in episodes:
            row["shaped_return"] = 10.0
    elif case == "negative_ev":
        for row in rows:
            row["explained_variance"] = -1.0
    result = assess_seed(
        episodes,
        rows,
        actual_steps=30000,
        budget=1000000 if case == "short_budget" else 30000,
        criterion=criterion,
    )
    assert not result["passed"]
    assert result["reasons"]


def test_missing_seed_and_inconsistent_seed_rewards_fail(criterion):
    episodes, rows = evidence()
    result = assess_seed(episodes, rows, actual_steps=30000, budget=30000, criterion=criterion)
    assert not assess_all({"42": result}, criterion)["passed"]
    all_results = {str(s): copy.deepcopy(result) for s in SEEDS}
    all_results["51"]["final_third"]["mean"] = 1000
    assert not assess_all(all_results, criterion)["passed"]


def test_flat_final_third_cannot_hide_a_late_drop_from_middle_third(criterion):
    episodes, rows = evidence()
    for row in episodes:
        if 10000 <= row["timesteps"] < 20000:
            row["shaped_return"] = 200.0
    result = assess_seed(episodes, rows, actual_steps=30000, budget=30000, criterion=criterion)
    assert not result["passed"]
    assert "late collapse" in result["reasons"]


def test_advice_is_not_a_sweep(criterion):
    rows = [
        {"approx_kl": 0.1, "clip_fraction": 0.5, "explained_variance": 0.01, "entropy_loss": -1}
        for _ in range(30)
    ]
    advice = diagnostic_advice(rows, criterion)
    assert any("LR stage" in line for line in advice)
    assert any("normalization" in line for line in advice)


def test_write_new_never_overwrites(tmp_path):
    path = tmp_path / "frozen.json"
    write_new(path, {"original": True})
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        write_new(path, {"original": False})
    assert path.read_bytes() == before


def test_old_config_bytes_equal_committed_baseline():
    # Pin the pre-convergence commit: HEAD would bless accidental edits once committed.
    baseline_commit = "bf6035a"
    paths = subprocess.run(
        ["git", "ls-tree", "-r", "--name-only", baseline_commit, "configs"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.splitlines()
    for name in paths:
        committed = subprocess.run(
            ["git", "show", f"{baseline_commit}:{name}"], cwd=ROOT, capture_output=True, check=True
        ).stdout
        assert (ROOT / name).read_bytes() == committed, name
