"""Short wiring tests only: no 1M run, no invented convergence evidence."""

import copy

import numpy as np
import pytest
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize

from rlredteam.catalogue import CVECatalogue
from rlredteam.convergence import BASE_CONFIG, ROOT, ConvergenceError, csv_rows, load_config, sha
from rlredteam.convergence_runner import (
    DiagnosticPPO,
    FrozenDeterministicPolicy,
    _train_one,
    normalized_environment,
    require_frozen,
    run,
    study_lock,
    verify,
    verify_run,
)
from rlredteam.evaluation import policy_digest
from rlredteam.reward import RewardConfig
from rlredteam.topology import TopologyConfig
from rlredteam.train import EpisodeCollector, build_env


def small_config(normalized=False):
    c = copy.deepcopy(load_config(BASE_CONFIG))
    c["total_timesteps"] = 32
    c["ppo"].update(n_steps=16, batch_size=8, n_epochs=2)
    c["normalization"]["enabled"] = normalized
    return c


def make_env():
    return build_env(
        TopologyConfig.from_yaml(),
        RewardConfig.from_yaml(ROOT / "configs/shaped.yaml"),
        CVECatalogue.open_default(),
        42,
        42,
    )


@pytest.mark.parametrize("normalized", [False, True])
def test_real_short_ppo_updates_saved_without_missing_final_update(tmp_path, normalized):
    c = small_config(normalized)
    out = tmp_path / "seed42"
    result = _train_one(c, 42, out, postgres=False)
    assert result["actual_timesteps"] == 32
    assert result["gradient_updates"] == 4
    assert result["status"] == "complete" and result["elapsed_seconds"] > 0
    rows = csv_rows(out / "diagnostics.csv")
    assert [int(row["timesteps"]) for row in rows] == [16, 32]
    assert all(row["approx_kl"] and row["learning_rate"] for row in rows)
    assert verify_run(out, c, 42) == result
    model = PPO.load(out / "model.zip", device="cpu")
    assert policy_digest(model) == result["policy_sha256"]
    if normalized:
        env = VecNormalize.load(out / "vecnormalize.pkl", make_env())
        assert env.ret_rms.count > 32
        assert env.norm_obs is False
        env.training = False
        state = (env.ret_rms.mean.copy(), env.ret_rms.var.copy(), env.ret_rms.count)
        env.reset()
        env.step(np.array([0]))
        assert state == (env.ret_rms.mean, env.ret_rms.var, env.ret_rms.count)
        env.close()
    before = sha(out / "model.zip")
    with pytest.raises(FileExistsError):
        _train_one(c, 42, out, postgres=False)
    assert sha(out / "model.zip") == before


def test_normalization_never_changes_partial_observation_or_event_info():
    plain, inner = make_env(), make_env()
    norm = normalized_environment(inner, small_config(True))
    initial, normalized_initial = plain.reset(), norm.reset()
    assert np.array_equal(initial, normalized_initial)
    assert np.count_nonzero(normalized_initial.reshape(9, 26)[1:8]) == 0
    for action in [0, 5, 10, 2]:
        obs, rewards, dones, infos = plain.step(np.array([action]))
        nobs, nrewards, ndones, ninfos = norm.step(np.array([action]))
        assert np.array_equal(obs, nobs)
        assert np.array_equal(dones, ndones)
        assert ninfos[0]["reward_breakdown"] == infos[0]["reward_breakdown"]
        assert ninfos[0]["native_reward"] == infos[0]["native_reward"]
        assert np.allclose(norm.get_original_reward(), rewards)
    plain.close()
    norm.close()


def test_policy_gets_only_partial_observation_and_deterministic_flag():
    class Spy:
        def predict(self, observation, deterministic):
            assert isinstance(observation, np.ndarray)
            assert deterministic is True
            return 0, None

    env = make_env()
    wrapper = FrozenDeterministicPolicy(Spy())
    wrapper.predict(env.reset(), deterministic=False)
    env.close()


def test_no_policy_updates_in_real_short_evaluation(tmp_path):
    from rlredteam.evaluation import evaluate_policy

    c = small_config()
    _train_one(c, 42, tmp_path / "s42", postgres=False)
    model = PPO.load(tmp_path / "s42/model.zip", device="cpu")
    before, updates = policy_digest(model), model._n_updates
    bundle = evaluate_policy(
        FrozenDeterministicPolicy(model),
        run_name="test-only",
        reward_mode="shaped",
        training_seed=42,
        evaluation_seeds=[42],
        topology_seed=42,
    )
    assert len(bundle.episodes) == 1
    assert policy_digest(model) == before and model._n_updates == updates


def test_sparse_and_final_evaluation_blocked_without_shaped_gate(tmp_path, monkeypatch):
    from rlredteam import convergence_runner as runner

    with pytest.raises(ConvergenceError, match="blocked"):
        require_frozen(BASE_CONFIG, tmp_path)
    with pytest.raises(ConvergenceError, match="blocked"):
        runner.evaluate(BASE_CONFIG, tmp_path)
    monkeypatch.setattr(runner, "require_clean", lambda: None)
    monkeypatch.setattr(
        runner, "verify_registration", lambda *_: ({"config": load_config(BASE_CONFIG)}, tmp_path)
    )
    with pytest.raises(ConvergenceError, match="blocked"):
        run(BASE_CONFIG, tmp_path, sparse=True)


def test_verifier_does_not_claim_unexecuted_convergence(tmp_path):
    result = verify(BASE_CONFIG, tmp_path)
    assert result["configuration_valid"]
    assert not result["convergence_established"]
    assert result["training"] == "not executed"


def test_study_lock_prevents_concurrent_mutations(tmp_path):
    with study_lock(tmp_path):
        with pytest.raises(ConvergenceError, match="lock"):
            with study_lock(tmp_path):
                pytest.fail("concurrent study mutation permitted")


def test_checkpoint_tampering_detected(tmp_path):
    c = small_config()
    out = tmp_path / "test-only"
    _train_one(c, 42, out, postgres=False)
    with (out / "model.zip").open("ab") as handle:
        handle.write(b"tampering")
    with pytest.raises(ConvergenceError, match="changed"):
        verify_run(out, c, 42)


def test_diagnostic_ppo_does_not_change_optimizer_algorithm():
    # Delegation to SB3 is the training seam, not a new policy/algorithm.
    assert issubclass(DiagnosticPPO, PPO)
    assert issubclass(EpisodeCollector, object)


def test_short_evaluation_pipeline_reconstructs_matched_evidence(tmp_path, monkeypatch):
    from dataclasses import replace

    from rlredteam import convergence_runner as runner
    from rlredteam.convergence import read_json, write_new

    # Disposable 32-step fixture, not a converged policy or research result.
    topology = replace(TopologyConfig.from_yaml(), step_limit=8)
    monkeypatch.setattr(runner.TopologyConfig, "from_yaml", lambda: topology)
    c = small_config(True)
    c.update(training_seeds=[42], evaluation_seeds=[42, 43])
    out = tmp_path / c["id"]
    out.mkdir()
    for arm in ("shaped", "sparse"):
        _train_one(c, 42, out / f"{arm}-42", postgres=False, reward_mode=arm)
    write_new(tmp_path / "first_stable.json", {"test_fixture_only": True})
    monkeypatch.setattr(runner, "require_frozen", lambda *_: None)
    monkeypatch.setattr(runner, "verify_registration", lambda *_: ({"config": c}, out))
    monkeypatch.setattr(runner, "persist_evaluation", lambda *_: {"test_fixture_only": True})
    runner.evaluate(BASE_CONFIG, tmp_path)
    runner.verify_evaluation(out, tmp_path, c)
    path = out / "final-evaluation/matched_outcomes.json"
    evidence = read_json(path)
    assert len(evidence["rows"]) == 4
    evidence["rows"][0]["policy_return"] += 100
    import json

    path.write_text(json.dumps(evidence))
    with pytest.raises(ConvergenceError, match="reconstruct"):
        runner.verify_evaluation(out, tmp_path, c)


def test_freeze_binds_metadata_and_blocks_tampering(tmp_path, monkeypatch):
    from rlredteam import convergence_runner as runner
    from rlredteam.convergence import SEEDS, read_json, write_new

    c = load_config(BASE_CONFIG)
    out = tmp_path / c["id"]
    out.mkdir()
    reg = {
        "config": c,
        "criterion": {"test_only": True},
        "order": 0,
        "inputs": {"git_commit": "test", "topology_sha256": "t", "cve_catalogue_sha256": "c"},
    }
    write_new(out / "registration.json", reg)
    write_new(out / "assessment.json", {"passed": True})
    for seed in SEEDS:
        write_new(out / f"shaped-{seed}/complete.json", {"policy_sha256": str(seed)})
    monkeypatch.setattr(runner, "require_clean", lambda: None)
    monkeypatch.setattr(runner, "verify_registration", lambda *_: (reg, out))
    monkeypatch.setattr(runner, "verify_assessment", lambda *_: {"passed": True})
    frozen = runner.freeze_candidate(BASE_CONFIG, tmp_path)
    assert frozen["config"] == c
    assert frozen["assessment_sha256"] == sha(out / "assessment.json")
    assert frozen["registration_sha256"] == sha(out / "registration.json")
    assert set(frozen["checkpoints"]) == {str(s) for s in SEEDS}
    assert runner.require_frozen(BASE_CONFIG, tmp_path) == frozen
    path = tmp_path / "first_stable.json"
    original = read_json(path)
    path.write_text("{}")  # Corruption only in disposable test evidence.
    with pytest.raises(ConvergenceError, match="already frozen"):
        runner.require_frozen(BASE_CONFIG, tmp_path)
    assert original["inputs"] == reg["inputs"]


def test_assessment_cannot_be_forged(tmp_path, monkeypatch):
    from rlredteam import convergence_runner as runner
    from rlredteam.convergence import write_new

    write_new(tmp_path / "assessment.json", {"passed": True})
    monkeypatch.setattr(runner, "compute_assessment", lambda *_: {"passed": False})
    with pytest.raises(ConvergenceError, match="reconstruct"):
        runner.verify_assessment(tmp_path)


def test_input_provenance_computes_real_topology_and_catalogue_hashes():
    from rlredteam.convergence_runner import current_inputs
    from rlredteam.manifest import digest as catalogue_digest

    inputs = current_inputs(BASE_CONFIG)
    assert inputs["files"]["configs/topology.yaml"] == sha(ROOT / "configs/topology.yaml")
    assert inputs["cve_catalogue_sha256"] == catalogue_digest(CVECatalogue.open_default())
    assert len(inputs["topology_sha256"]) == 64
    assert inputs["topology_hash"] == "f875a25adf37ee34"


def test_registration_is_immutable_and_parent_cannot_run_early(tmp_path, monkeypatch):
    from rlredteam import convergence_runner as runner

    # A disposable test registration; no scientific training is launched.
    monkeypatch.setattr(runner, "require_clean", lambda: None)
    registered = runner.register(BASE_CONFIG, tmp_path)
    checked, out = runner.verify_registration(BASE_CONFIG, tmp_path)
    assert checked == registered
    original = sha(out / "registration.json")
    normalized = ROOT / "configs/experiments/experiment_01_convergence_normalized_1m_v2.yaml"
    with pytest.raises(ConvergenceError, match="assessment"):
        runner.register(normalized, tmp_path)
    with pytest.raises(ConvergenceError, match="assessment"):
        runner.register(BASE_CONFIG, tmp_path)
    assert sha(out / "registration.json") == original


def test_dirty_tree_gate_is_not_bypassed(monkeypatch):
    from rlredteam import convergence_runner as runner

    monkeypatch.setattr(runner.prov, "git_dirty", lambda: True)
    with pytest.raises(ConvergenceError, match="clean"):
        runner.require_clean()


def test_analysis_artifacts_byte_deterministic(tmp_path):
    import csv

    from rlredteam.convergence import DIAGNOSTICS, load_criterion
    from rlredteam.convergence_runner import _plots

    c = small_config()
    c["training_seeds"] = [42]
    criterion = load_criterion(ROOT / "configs/convergence_criterion_v2.yaml")
    for name in ("first", "second"):
        out = tmp_path / name
        out.mkdir()
        run_path = out / "shaped-42"
        # A small plotting fixture, explicitly not experimental evidence.
        run_path.mkdir()
        for filename, rows in (
            ("episodes.csv", [{"timesteps": i, "shaped_return": i} for i in range(1, 40)]),
            ("diagnostics.csv", [{"timesteps": 16, **{key: 0.1 for key in DIAGNOSTICS}}]),
        ):
            with (run_path / filename).open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
        _plots(out, {"config": c, "criterion": criterion}, {"per_seed": {"42": {"passed": False}}})
    first = {p.name: sha(p) for p in (tmp_path / "first/analysis").iterdir()}
    second = {p.name: sha(p) for p in (tmp_path / "second/analysis").iterdir()}
    assert first == second
    assert "seed-42-diagnostics.png" in first
    assert "aggregate-reward.png" in first and "aggregate-reward.json" in first


def test_frozen_protocol_forwards_stochastic_selection_and_seed():
    from rlredteam.convergence_runner import FrozenProtocolPolicy

    class Spy:
        def set_random_seed(self, seed):
            self.seed = seed

        def predict(self, obs, deterministic):
            assert self.seed == 1001
            assert deterministic is False
            return 0, None

    policy = FrozenProtocolPolicy(Spy(), "stochastic")
    policy.set_random_seed(1001)
    policy.predict(np.zeros(234))


def test_diagnostic_summaries_keep_missing_values_explicit():
    from rlredteam.convergence_runner import summarize_diagnostics

    rows = [{"timesteps": 100, "approx_kl": 0.1}, {"timesteps": 300, "approx_kl": 0.2}]
    summary = summarize_diagnostics(rows, 300)
    assert summary["approx_kl"]["final_third_mean"] == 0.2
    assert summary["value_loss"]["mean"] is None
    assert summary["value_loss"]["missing_updates"] == 2


def test_first_passing_candidate_cannot_be_skipped(tmp_path, monkeypatch):
    from rlredteam import convergence_runner as runner
    from rlredteam.convergence import write_new

    write_new(tmp_path / "first/registration.json", {"order": 0})
    monkeypatch.setattr(runner, "require_clean", lambda: None)
    monkeypatch.setattr(runner, "verify_assessment", lambda *_: {"passed": True})
    with pytest.raises(ConvergenceError, match="earlier passing"):
        runner.register(BASE_CONFIG, tmp_path)


@pytest.mark.postgres
def test_normalized_training_and_frozen_evaluation_persist_raw_evidence(tmp_path, monkeypatch):
    from dataclasses import replace

    import psycopg

    from rlredteam import convergence_runner as runner
    from rlredteam.evaluation import evaluate_policy
    from rlredteam.provenance import ExperimentManifest
    from rlredteam.storage.postgres_logger import connection_string

    try:
        conn = psycopg.connect(connection_string(), connect_timeout=3, autocommit=True)
    except psycopg.OperationalError:
        pytest.skip("PostgreSQL unavailable")
    topology = replace(TopologyConfig.from_yaml(), step_limit=8)
    monkeypatch.setattr(runner.TopologyConfig, "from_yaml", lambda: topology)
    c = small_config(True)
    c["id"] = "pytest_convergence_raw_rewards"
    out = tmp_path / "db-smoke"
    experiment_id = None
    evaluation_experiment_id = None
    try:
        _train_one(c, 42, out, postgres=True)
        manifest = ExperimentManifest.read(out / "manifest.json")
        experiment_id = manifest.database_experiment_id
        rows = csv_rows(out / "episodes.csv")
        stored = conn.execute(
            "SELECT total_reward, length FROM episodes WHERE run_id=%s ORDER BY episode_idx",
            (manifest.database_run_id,),
        ).fetchall()
        assert len(stored) == len(rows) > 0
        assert [r[0] for r in stored] == pytest.approx(
            [float(r["shaped_return"]) for r in rows], abs=1e-4
        )
        model = PPO.load(out / "model.zip", device="cpu")
        before = policy_digest(model)
        bundle = evaluate_policy(
            FrozenDeterministicPolicy(model),
            run_name=manifest.experiment_id,
            reward_mode="shaped",
            training_seed=42,
            evaluation_seeds=[42, 43],
            topology_seed=42,
            topology_config=topology,
        )
        persisted = runner.persist_evaluation(bundle, manifest, out / "model.zip")
        evaluation_experiment_id = persisted["database_experiment_id"]
        assert persisted["database_training_experiment_id"] == experiment_id
        assert evaluation_experiment_id != experiment_id
        evaluation_id = persisted["database_evaluation_run_id"]
        count = conn.execute(
            "SELECT count(*) FROM episodes WHERE run_id=%s", (evaluation_id,)
        ).fetchone()[0]
        step_count = conn.execute(
            "SELECT count(*) FROM steps s JOIN episodes e ON e.id=s.episode_id WHERE e.run_id=%s",
            (evaluation_id,),
        ).fetchone()[0]
        assert count == 2 and step_count == len(bundle.steps)
        assert policy_digest(model) == before
    finally:
        if evaluation_experiment_id is not None:
            conn.execute("DELETE FROM experiments WHERE id=%s", (evaluation_experiment_id,))
        if experiment_id is not None:
            conn.execute("DELETE FROM experiments WHERE id=%s", (experiment_id,))
            conn.commit()
        conn.close()
