"""Additive Podman convergence runner, immutable registrations and gated evaluation.

Historical train/evaluation modules are reused, never changed. A study-wide lock
serialises transitions; incomplete outputs are preserved, never silently resumed.
"""

from __future__ import annotations

import csv
import fcntl
import json
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import VecNormalize

from rlredteam import provenance as prov
from rlredteam.catalogue import CVECatalogue
from rlredteam.convergence import (
    DIAGNOSTICS,
    ROOT,
    SCHEMA,
    ConvergenceError,
    assess_all,
    assess_seed,
    csv_rows,
    diagnostic_advice,
    diagnostic_row,
    digest,
    load_config,
    load_criterion,
    read_json,
    sha,
    write_new,
)
from rlredteam.evaluation import evaluate_policy, policy_digest, write_bundle
from rlredteam.manifest import digest as catalogue_digest
from rlredteam.reward import RewardConfig
from rlredteam.topology import TopologyConfig, describe, make_env
from rlredteam.train import EpisodeCollector, build_env, linear_learning_rate, set_all_seeds

DEFAULT_OUTPUT = ROOT / "results/convergence_v2"


@contextmanager
def study_lock(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ConvergenceError("another convergence command holds the study lock") from exc
        yield


def current_inputs(config_path: Path) -> dict:
    c = load_config(config_path)
    tracked = subprocess.run(
        ["git", "ls-files", "src", "scripts", "configs", "pyproject.toml"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.splitlines()
    # Include the additive files even before their first commit, for diagnostic tests.
    paths = set(tracked) | {
        "src/rlredteam/convergence.py",
        "src/rlredteam/convergence_runner.py",
        "scripts/run_convergence.py",
        str(config_path.resolve().relative_to(ROOT)),
        c["criterion"],
    }
    env = make_env(TopologyConfig.from_yaml(), topology_seed=c["topology_seed"])
    try:
        topology = describe(env)
    finally:
        env.close()
    return {
        "files": {p: sha(ROOT / p) for p in sorted(paths)},
        "config_sha256": sha(config_path),
        "git_commit": prov.git_commit(),
        "topology_sha256": digest(topology),
        "topology_hash": prov.topology_hash(topology),
        "cve_catalogue_sha256": catalogue_digest(CVECatalogue.open_default()),
        "image_digest": prov.docker_image_digest(),
    }


def require_clean() -> None:
    if not prov.git_commit() or prov.git_dirty() is not False:
        raise ConvergenceError("scientific runs require a committed, clean working tree")


def register(config_path: Path, root: Path) -> dict:
    require_clean()
    c = load_config(config_path)
    if c["schema"] != SCHEMA:
        raise ConvergenceError("superseded execution protocol; use the approved v2 configuration")
    if (root / "first_stable.json").exists():
        raise ConvergenceError("first stable candidate already frozen; tuning is closed")
    registrations = sorted(root.glob("*/registration.json"))
    # No best-of-many selection: a later candidate cannot begin until every
    # earlier registered candidate has completed and failed this same criterion.
    for path in registrations:
        prior = verify_assessment(path.parent)
        if prior["passed"]:
            raise ConvergenceError("earlier passing candidate must be frozen first")
        if prior["criterion_sha256"] != digest(load_criterion(ROOT / c["criterion"])):
            raise ConvergenceError("cannot change criterion within an existing study")
    if c["parent"]:
        parent = load_config(ROOT / c["parent"])
        if not (root / parent["id"] / "assessment.json").exists():
            raise ConvergenceError("parent must complete and be assessed first")
    elif registrations:
        raise ConvergenceError("only one first under-training candidate per study")
    out = root / c["id"]
    out.mkdir(exist_ok=False)
    registration = {
        "schema": "security-rl-convergence-registration-v1",
        "config": c,
        "criterion": load_criterion(ROOT / c["criterion"]),
        "inputs": current_inputs(config_path),
        "order": len(registrations),
        "config_path": str(config_path.resolve().relative_to(ROOT)),
    }
    write_new(out / "registration.json", registration)
    return registration


def verify_registration(config_path: Path, root: Path) -> tuple[dict, Path]:
    c = load_config(config_path)
    out = root / c["id"]
    registration = read_json(out / "registration.json")
    if registration["config"] != c or registration["inputs"] != current_inputs(config_path):
        raise ConvergenceError("registered source/config/topology/catalogue/image changed")
    if registration["criterion"] != load_criterion(ROOT / c["criterion"]):
        raise ConvergenceError("registered criterion changed")
    return registration, out


class DiagnosticPPO(PPO):
    """Capture *after* each PPO update, including the final undumped SB3 update."""

    def _excluded_save_params(self):
        return super()._excluded_save_params() + [
            "episode_collector",
            "diagnostic_path",
            "diagnostic_update",
        ]

    def attach_diagnostics(self, path: Path, collector: EpisodeCollector) -> None:
        self.diagnostic_path = path
        self.episode_collector = collector
        self.diagnostic_update = 0
        with path.open("x", newline="") as handle:
            csv.DictWriter(handle, fieldnames=["update", "timesteps", *DIAGNOSTICS]).writeheader()

    def train(self) -> None:
        super().train()
        self.diagnostic_update += 1
        row = diagnostic_row(
            self.logger.name_to_value,
            update=self.diagnostic_update,
            timestep=self.num_timesteps,
            episodes=self.episode_collector.episodes,
        )
        with self.diagnostic_path.open("a", newline="") as handle:
            csv.DictWriter(handle, fieldnames=list(row)).writerow(row)


def normalized_environment(env, config: dict):
    n = config["normalization"]
    return (
        VecNormalize(
            env,
            norm_obs=False,
            norm_reward=True,
            gamma=n["gamma"],
            epsilon=n["epsilon"],
            clip_reward=n["clip_reward"],
        )
        if n["enabled"]
        else env
    )


def _train_one(c: dict, seed: int, out: Path, *, postgres: bool, reward_mode="shaped") -> dict:
    """Private small-run seam for tests; public run always enforces full protocol."""
    out.mkdir(exist_ok=False)
    started = time.monotonic()
    set_all_seeds(seed)
    reward = RewardConfig.from_yaml(ROOT / f"configs/{reward_mode}.yaml")
    catalogue = CVECatalogue.open_default()
    topology = TopologyConfig.from_yaml()
    env = build_env(topology, reward, catalogue, c["topology_seed"], seed)
    description = describe(env.envs[0].env)
    manifest = prov.ExperimentManifest(
        experiment_id=f"{c['id']}-{reward_mode}-s{seed}",
        git_commit=prov.git_commit(),
        git_dirty=prov.git_dirty(),
        python_version=prov.python_version(),
        dependency_lock_hash=prov.dependency_lock_hash(),
        docker_image_digest=prov.docker_image_digest(),
        training_seed=seed,
        topology_seed=c["topology_seed"],
        topology_hash=prov.topology_hash(description),
        topology_config_hash=topology.config_hash(),
        environment_config_hash=prov.environment_config_hash(description),
        cve_manifest_sha256=catalogue_digest(catalogue),
        reward_config_hash=reward.hash(),
        ppo_config_hash=digest(c["ppo"] | {"learning_rate_schedule": c["learning_rate_schedule"]}),
        dataset_version=str(len(catalogue)),
        training_budget=c["total_timesteps"],
        checkpoint_path=str(out / "model.zip"),
        ppo_config=c["ppo"],
        reward_mode=reward_mode,
    )
    logger = None
    collector = None
    env = normalized_environment(env, c)
    try:
        if postgres:
            from rlredteam.storage.postgres_logger import EpisodeLogger

            logger = EpisodeLogger.start(
                name=manifest.experiment_id,
                reward_mode=reward_mode,
                config_hash=manifest.reward_config_hash,
                topology_config_hash=manifest.topology_config_hash,
                topology_hash=manifest.topology_hash,
                cve_manifest_sha256=manifest.cve_manifest_sha256,
                seed_set=[seed],
                log_steps=True,
                condition=reward_mode,
                algorithm="PPO",
                hyperparameters=c["ppo"],
                designation="training",
                checkpoint_path=manifest.checkpoint_path,
            )
            manifest.database_experiment_id, manifest.database_run_id = (
                logger.experiment_id,
                logger.run_id,
            )
        manifest.write(out / "manifest.json")
        collector = EpisodeCollector(
            seed=seed,
            topology_seed=c["topology_seed"],
            episode_logger=logger,
            csv_path=out / "episodes.csv",
            record_steps=postgres,
        )
        params = dict(c["ppo"])
        if c["learning_rate_schedule"] == "linear_to_zero":
            params["learning_rate"] = linear_learning_rate(params["learning_rate"])
        model = DiagnosticPPO(env=env, seed=seed, **params)
        model.set_logger(configure(format_strings=[]))
        model.attach_diagnostics(out / "diagnostics.csv", collector)
        model.learn(total_timesteps=c["total_timesteps"], callback=collector)
        model.save(out / "model.zip")
        if c["normalization"]["enabled"]:
            env.save(out / "vecnormalize.pkl")
        result = {
            "seed": seed,
            "reward_mode": reward_mode,
            "actual_timesteps": model.num_timesteps,
            "policy_sha256": policy_digest(model),
            "gradient_updates": model._n_updates,
            "config_sha256": digest(c),
            "normalization": c["normalization"],
            "files": {p.name: sha(p) for p in sorted(out.iterdir()) if p.is_file()},
        }
        if c["schema"] == SCHEMA:
            result.update(status="complete", elapsed_seconds=time.monotonic() - started)
        if logger:
            logger.flush()
            logger.finish("complete")
        write_new(out / "complete.json", result)
        return result
    except Exception:
        if logger:
            logger.finish("failed")
        raise
    finally:
        env.close()
        if collector is not None:
            collector._on_training_end()
        if logger:
            logger._conn.close()


def verify_run(path: Path, config: dict, seed: int, reward_mode="shaped") -> dict:
    result = read_json(path / "complete.json")
    if (
        result["seed"] != seed
        or result["reward_mode"] != reward_mode
        or result["config_sha256"] != digest(config)
    ):
        raise ConvergenceError("run does not match registered seed/config/condition")
    expected_steps = (
        int(np.ceil(config["total_timesteps"] / config["ppo"]["n_steps"]))
        * config["ppo"]["n_steps"]
    )
    if result["actual_timesteps"] != expected_steps:
        raise ConvergenceError("run did not complete its registered rollout budget")
    required = {"model.zip", "manifest.json", "episodes.csv", "diagnostics.csv"}
    if config["normalization"]["enabled"]:
        required.add("vecnormalize.pkl")
    if set(result["files"]) != required:
        raise ConvergenceError("missing or unexpected run evidence")
    for name, expected in result["files"].items():
        if sha(path / name) != expected:
            raise ConvergenceError(f"run evidence changed: {name}")
    manifest = prov.ExperimentManifest.read(path / "manifest.json")
    if (
        manifest.training_seed != seed
        or manifest.topology_seed != config["topology_seed"]
        or manifest.training_budget != config["total_timesteps"]
        or manifest.ppo_config != config["ppo"]
        or manifest.reward_mode != reward_mode
    ):
        raise ConvergenceError("training manifest does not match configuration")
    rows = csv_rows(path / "diagnostics.csv")
    expected_updates = expected_steps // config["ppo"]["n_steps"]
    if len(rows) != expected_updates or any(
        int(r["update"]) != i or int(r["timesteps"]) != i * config["ppo"]["n_steps"]
        for i, r in enumerate(rows, 1)
    ):
        raise ConvergenceError("diagnostic update sequence incomplete")
    return result


def run(config_path: Path, root: Path, *, sparse=False) -> None:
    require_clean()
    reg, out = verify_registration(config_path, root)
    c = reg["config"]
    if sparse:
        require_frozen(config_path, root)
    elif (root / "first_stable.json").exists():
        raise ConvergenceError("tuning closed after first passing candidate")
    condition = "sparse" if sparse else "shaped"
    for seed in c["training_seeds"]:
        path = out / f"{condition}-{seed}"
        if path.exists():
            verify_run(path, c, seed, condition)
            print(f"verified existing {condition} seed {seed}; not retrained", flush=True)
            continue
        print(f"starting {condition} seed {seed}: {c['total_timesteps']} timesteps", flush=True)
        result = _train_one(c, seed, path, postgres=True, reward_mode=condition)
        print(
            f"completed {condition} seed {seed}: "
            f"{result.get('elapsed_seconds', 'unknown')} seconds",
            flush=True,
        )


def compute_assessment(out: Path) -> dict:
    reg = read_json(out / "registration.json")
    c, criterion = reg["config"], reg["criterion"]
    per_seed, evidence = {}, {}
    for seed in c["training_seeds"]:
        path = out / f"shaped-{seed}"
        result = verify_run(path, c, seed)
        rows = csv_rows(path / "diagnostics.csv")
        per_seed[str(seed)] = assess_seed(
            csv_rows(path / "episodes.csv"),
            rows,
            actual_steps=result["actual_timesteps"],
            budget=c["total_timesteps"],
            criterion=criterion,
        )
        per_seed[str(seed)]["diagnostic_advice"] = diagnostic_advice(rows, criterion)
        if c["schema"] == SCHEMA:
            per_seed[str(seed)]["diagnostic_summaries"] = summarize_diagnostics(
                rows, result["actual_timesteps"]
            )
        evidence[str(seed)] = sha(path / "complete.json")
    return assess_all(per_seed, criterion) | {
        "registration_sha256": sha(out / "registration.json"),
        "run_evidence": evidence,
        "inputs": reg["inputs"],
        "candidate": c["id"],
    }


def summarize_diagnostics(rows: list[dict], actual_steps: int) -> dict:
    summary = {}
    for key in DIAGNOSTICS:
        finite = [
            (int(r["timesteps"]), float(r[key]))
            for r in rows
            if r.get(key) not in (None, "") and np.isfinite(float(r[key]))
        ]
        values = [v for _, v in finite]
        tail = [v for t, v in finite if t >= 2 * actual_steps / 3]
        summary[key] = {
            "available_updates": len(values),
            "missing_updates": len(rows) - len(values),
            "mean": float(np.mean(values)) if values else None,
            "min": min(values) if values else None,
            "max": max(values) if values else None,
            "last": values[-1] if values else None,
            "final_third_mean": float(np.mean(tail)) if tail else None,
        }
    return summary


def verify_assessment(out: Path) -> dict:
    if not (out / "assessment.json").exists():
        raise ConvergenceError("earlier candidate has no completed assessment")
    stored = read_json(out / "assessment.json")
    if stored != compute_assessment(out):
        raise ConvergenceError("assessment does not reconstruct from raw evidence")
    return stored


def _plots(out: Path, reg: dict, report: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    destination = out / "analysis"
    destination.mkdir(exist_ok=False)
    for seed in reg["config"]["training_seeds"]:
        run_path = out / f"shaped-{seed}"
        rows = csv_rows(run_path / "episodes.csv")
        window = reg["criterion"]["smooth_episodes"]
        values = [float(r["shaped_return"]) for r in rows]
        curve = [
            {
                "timesteps": int(r["timesteps"]),
                "reward": values[i],
                "smoothed_reward": float(np.mean(values[max(0, i - window + 1) : i + 1])),
            }
            for i, r in enumerate(rows)
        ]
        write_new(destination / f"seed-{seed}-curve.json", curve)
        write_new(destination / f"seed-{seed}-final-third.json", report["per_seed"][str(seed)])
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(
            [r["timesteps"] for r in curve],
            values,
            alpha=0.3,
            label="raw unnormalised shaped return",
        )
        ax.plot(
            [r["timesteps"] for r in curve],
            [r["smoothed_reward"] for r in curve],
            label=f"trailing {window} episodes",
        )
        ax.set(
            xlabel="environment timesteps",
            ylabel="episode reward",
            title=f"Seed {seed}: descriptive curve, not a convergence verdict",
        )
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(
            destination / f"seed-{seed}-reward.png",
            metadata={"Software": "Security-RL convergence-v1"},
        )
        plt.close(fig)
        diagnostic_rows = csv_rows(run_path / "diagnostics.csv")
        fig, axes = plt.subplots(3, 3, figsize=(12, 9))
        for ax, key in zip(axes.flat, DIAGNOSTICS, strict=True):
            ax.plot(
                [int(r["timesteps"]) for r in diagnostic_rows],
                [float(r[key]) if r[key] else np.nan for r in diagnostic_rows],
            )
            ax.set(title=key, xlabel="timesteps")
        fig.tight_layout()
        fig.savefig(
            destination / f"seed-{seed}-diagnostics.png",
            metadata={"Software": "Security-RL convergence-v1"},
        )
        plt.close(fig)
    write_new(destination / "aggregate.json", report)
    if reg["config"]["schema"] == SCHEMA:
        aggregate_reward_curve(out, reg["config"])


def aggregate_reward_curve(out: Path, config: dict) -> dict:
    """Equal-seed mean on the recorded PPO-update grid; no interpolation."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by_seed = {
        str(seed): {
            int(r["timesteps"]): float(r["mean_episode_reward"])
            if r["mean_episode_reward"]
            else None
            for r in csv_rows(out / f"shaped-{seed}/diagnostics.csv")
        }
        for seed in config["training_seeds"]
    }
    points = []
    for step in sorted({step for rows in by_seed.values() for step in rows}):
        values = {seed: rows.get(step) for seed, rows in by_seed.items()}
        available = [v for v in values.values() if v is not None and np.isfinite(v)]
        complete = len(available) == len(by_seed)
        points.append(
            {
                "timesteps": step,
                "per_seed": values,
                "mean": float(np.mean(available)) if complete else None,
                "seed_std": float(np.std(available, ddof=1))
                if complete and len(available) > 1
                else None,
            }
        )
    payload = {
        "schema": "security-rl-convergence-aggregate-curve-v1",
        "source": "diagnostics.csv:mean_episode_reward (trailing 100 original-reward episodes)",
        "aggregation": "equal seed weights; exact recorded timestep; no interpolation",
        "band": "descriptive between-seed standard deviation; not a confidence interval",
        "points": points,
    }
    write_new(out / "analysis/aggregate-reward.json", payload)
    x = [r["timesteps"] for r in points]
    y = np.array([r["mean"] if r["mean"] is not None else np.nan for r in points])
    spread = np.array([r["seed_std"] if r["seed_std"] is not None else np.nan for r in points])
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(x, y, label="equal-seed mean of recorded trailing-100 episode returns")
    if len(by_seed) > 1:
        ax.fill_between(x, y - spread, y + spread, alpha=0.2, label="between-seed SD (not CI)")
    ax.set(
        xlabel="environment timesteps",
        ylabel="original shaped episode return",
        title="Descriptive aggregate reward curve — not a convergence verdict",
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(
        out / "analysis/aggregate-reward.png", metadata={"Software": "Security-RL convergence-v2"}
    )
    plt.close(fig)
    return payload


def analyze(config_path: Path, root: Path) -> dict:
    reg, out = verify_registration(config_path, root)
    report = compute_assessment(out)
    if (out / "assessment.json").exists():
        verify_assessment(out)
    else:
        _plots(out, reg, report)
        write_new(out / "assessment.json", report)
    if (
        reg["config"]["stage"] == "normalization"
        and not (out / "baseline-comparison.json").exists()
    ):
        parent = load_config(ROOT / reg["config"]["parent"])
        baseline = verify_assessment(root / parent["id"])
        write_new(
            out / "baseline-comparison.json",
            {
                "schema": "security-rl-convergence-comparison-v1",
                "baseline": baseline,
                "normalized": report,
                "reward_basis": "original unnormalised shaped reward",
                "note": "Training convergence comparison, not final evaluation "
                "or inferential evidence",
            },
        )
    if report["passed"]:
        freeze_candidate(config_path, root)
    return report


def freeze_candidate(config_path: Path, root: Path) -> dict:
    require_clean()
    reg, out = verify_registration(config_path, root)
    report = verify_assessment(out)
    if not report["passed"]:
        raise ConvergenceError("convergence gate failed")
    for other in root.glob("*/registration.json"):
        prior = read_json(other)
        if prior["order"] < reg["order"] and verify_assessment(other.parent)["passed"]:
            raise ConvergenceError("cannot select a later candidate over the first passing one")
    frozen = {
        "schema": "security-rl-first-stable-v1",
        "candidate": reg["config"]["id"],
        "registration_sha256": sha(out / "registration.json"),
        "assessment_sha256": sha(out / "assessment.json"),
        "inputs": reg["inputs"],
        "config": reg["config"],
        "criterion": reg["criterion"],
        "checkpoints": {
            str(s): read_json(out / f"shaped-{s}/complete.json")
            for s in reg["config"]["training_seeds"]
        },
    }
    path = root / "first_stable.json"
    if path.exists():
        if read_json(path) != frozen:
            raise ConvergenceError("another candidate is already frozen")
    else:
        write_new(path, frozen)
    return frozen


def require_frozen(config_path: Path, root: Path) -> dict:
    if not (root / "first_stable.json").is_file():
        raise ConvergenceError("final n=10 evaluation blocked: no passing frozen shaped candidate")
    return freeze_candidate(config_path, root)


class FrozenDeterministicPolicy:
    def __init__(self, model):
        self.model = model

    def predict(self, observation, deterministic=False):
        return self.model.predict(observation, deterministic=True)


class FrozenProtocolPolicy:
    """Frozen inference with the registered action-selection mode and episode seed."""

    def __init__(self, model, action_selection):
        self.model = model
        self.deterministic = action_selection == "deterministic"

    def set_random_seed(self, seed):
        self.model.set_random_seed(seed)

    def predict(self, observation, deterministic=False):
        return self.model.predict(observation, deterministic=self.deterministic)


def persist_evaluation(bundle, manifest, checkpoint: Path) -> dict:
    """Existing canonical tables, separate evaluation namespace for reused seeds.

    Legacy episodes are unique on (experiment, seed, episode_idx), not run_id.
    Preserve that schema and explicitly link this evaluation-only experiment to
    its training parent in PostgreSQL notes and canonical metadata.
    """
    from rlredteam.evaluation import _step_record
    from rlredteam.storage.postgres_logger import EpisodeLogger, EpisodeRecord

    if manifest.database_experiment_id is None:
        raise ConvergenceError("evaluation has no PostgreSQL training parent")
    logger = EpisodeLogger.start(
        name=manifest.experiment_id + "-evaluation",
        reward_mode=manifest.reward_mode,
        config_hash=manifest.reward_config_hash,
        topology_config_hash=manifest.topology_config_hash,
        topology_hash=manifest.topology_hash,
        cve_manifest_sha256=manifest.cve_manifest_sha256,
        seed_set=[manifest.training_seed],
        condition=manifest.reward_mode,
        algorithm="PPO",
        designation="evaluation",
        log_steps=True,
        evaluation_seeds=[e.evaluation_seed for e in bundle.episodes],
        checkpoint_path=str(checkpoint),
        notes=json.dumps(
            {
                "schema": "security-rl-convergence-evaluation-link-v1",
                "training_experiment_id": manifest.database_experiment_id,
                "training_run_id": manifest.database_run_id,
                "checkpoint_sha256": sha(checkpoint),
            },
            sort_keys=True,
        ),
    )
    try:
        for index, episode in enumerate(bundle.episodes):
            steps = [s for s in bundle.steps if s["evaluation_seed"] == episode.evaluation_seed]
            logger.log_episode(
                EpisodeRecord(
                    seed=episode.evaluation_seed,
                    topology_seed=episode.topology_seed,
                    episode_idx=index,
                    total_reward=episode.policy_return,
                    native_reward=episode.native_return,
                    length=episode.length,
                    terminal_state=episode.terminal_reason,
                    goal_reached=episode.goal_reached,
                    exploited_hosts=[
                        s["target"]
                        for s in steps
                        if s["success"] and s["access_gained"] > 0 and s["target"]
                    ],
                    mean_cvss_exploited=episode.mean_cvss_exploited,
                    max_cvss_exploited=episode.max_cvss_exploited,
                    hosts_compromised=episode.hosts_compromised,
                    steps=[_step_record(s) for s in steps],
                )
            )
        logger.flush()
        logger.finish("complete")
        return {
            "database_training_experiment_id": manifest.database_experiment_id,
            "database_training_run_id": manifest.database_run_id,
            "database_experiment_id": logger.experiment_id,
            "database_evaluation_run_id": logger.run_id,
        }
    except Exception:
        logger.finish("failed")
        raise
    finally:
        logger._conn.close()


def evaluate(config_path: Path, root: Path) -> None:
    require_frozen(config_path, root)
    reg, out = verify_registration(config_path, root)
    c = reg["config"]
    # Preflight both arms before any evaluation output is created.
    for condition in ("shaped", "sparse"):
        for seed in c["training_seeds"]:
            verify_run(out / f"{condition}-{seed}", c, seed, condition)
    destination = out / "final-evaluation"
    destination.mkdir(exist_ok=False)
    all_rows = []
    for condition in ("shaped", "sparse"):
        for seed in c["training_seeds"]:
            run_path = out / f"{condition}-{seed}"
            manifest = prov.ExperimentManifest.read(run_path / "manifest.json")
            model = PPO.load(run_path / "model.zip", device="cpu")
            model.policy.set_training_mode(False)
            before, updates = policy_digest(model), model._n_updates
            if c["normalization"]["enabled"]:
                # Reward-only normalization never transforms policy observations.
                # Verify/load saved state, freeze it, and report raw reward via the
                # existing evaluator. No RMS or observation adaptation at test time.
                vec = build_env(
                    TopologyConfig.from_yaml(),
                    RewardConfig.from_yaml(ROOT / f"configs/{condition}.yaml"),
                    CVECatalogue.open_default(),
                    c["topology_seed"],
                    seed,
                )
                norm = VecNormalize.load(run_path / "vecnormalize.pkl", vec)
                norm.training, norm.norm_reward = False, False
                if norm.norm_obs:
                    raise ConvergenceError("unexpected observation normalization")
                norm.close()
            import torch

            with torch.no_grad():
                bundle = evaluate_policy(
                    FrozenProtocolPolicy(model, c["evaluation"]["action_selection"]),
                    run_name=manifest.experiment_id,
                    reward_mode=condition,
                    training_seed=seed,
                    evaluation_seeds=c["evaluation_seeds"],
                    topology_seed=c["topology_seed"],
                )
            after = policy_digest(model)
            if before != after or updates != model._n_updates:
                raise ConvergenceError("policy updated during evaluation")
            if before != read_json(run_path / "complete.json")["policy_sha256"]:
                raise ConvergenceError("loaded policy differs from frozen training policy")
            database = persist_evaluation(bundle, manifest, run_path / "model.zip")
            metadata = {
                "policy_sha256_before": before,
                "policy_sha256_after": after,
                "gradient_updates": False,
                "action_selection": c["evaluation"]["action_selection"],
                "evaluation_seeds": c["evaluation_seeds"],
                "training_seed": seed,
                "checkpoint_sha256": sha(run_path / "model.zip"),
                "registration": reg,
                "normalization_updates": False,
                "reward_basis": "unnormalised",
                **database,
            }
            write_bundle(bundle, destination / f"{condition}-{seed}", metadata)
            all_rows.extend(
                e.to_row()
                | {
                    "crown_jewel_reach": any(
                        s["evaluation_seed"] == e.evaluation_seed
                        and s["is_crown_jewel"]
                        and s["success"]
                        and s["access_gained"] > 0
                        for s in bundle.steps
                    )
                }
                for e in bundle.episodes
            )
    write_new(
        destination / "matched_outcomes.json",
        {
            "unit": "training seed (n=10); 10 paired episodes per seed",
            "rows": all_rows,
            "frozen_sha256": sha(root / "first_stable.json"),
            "files": {
                str(p.relative_to(destination)): sha(p)
                for p in sorted(destination.rglob("*"))
                if p.is_file()
            },
        },
    )


def verify_evaluation(out: Path, root: Path, config: dict) -> None:
    destination = out / "final-evaluation"
    evidence = read_json(destination / "matched_outcomes.json")
    if evidence["frozen_sha256"] != sha(root / "first_stable.json"):
        raise ConvergenceError("evaluation uses a different frozen candidate")
    expected_files = {
        f"{arm}-{seed}/{name}"
        for arm in ("shaped", "sparse")
        for seed in config["training_seeds"]
        for name in ("evaluation.csv", "steps.jsonl", "evaluation_metadata.json")
    }
    if set(evidence["files"]) != expected_files:
        raise ConvergenceError("incomplete evaluation file set")
    for name, value in evidence["files"].items():
        if sha(destination / name) != value:
            raise ConvergenceError("evaluation evidence changed")
    indexed = {
        (r["reward_mode"], r["training_seed"], r["evaluation_seed"]): r for r in evidence["rows"]
    }
    expected_count = 2 * len(config["training_seeds"]) * len(config["evaluation_seeds"])
    if len(indexed) != expected_count or len(evidence["rows"]) != expected_count:
        raise ConvergenceError("matched outcome count/uniqueness mismatch")
    for arm in ("shaped", "sparse"):
        for seed in config["training_seeds"]:
            run_path = out / f"{arm}-{seed}"
            trained = verify_run(run_path, config, seed, arm)
            metadata = read_json(destination / f"{arm}-{seed}/evaluation_metadata.json")
            if (
                metadata["gradient_updates"] is not False
                or metadata["normalization_updates"] is not False
                or metadata["policy_sha256_before"] != trained["policy_sha256"]
                or metadata["policy_sha256_after"] != trained["policy_sha256"]
                or metadata["checkpoint_sha256"] != sha(run_path / "model.zip")
                or metadata["evaluation_seeds"] != config["evaluation_seeds"]
                or metadata["action_selection"] != config["evaluation"]["action_selection"]
            ):
                raise ConvergenceError("evaluation policy/seed protocol mismatch")
            rows = csv_rows(destination / f"{arm}-{seed}/evaluation.csv")
            if [int(r["evaluation_seed"]) for r in rows] != config["evaluation_seeds"]:
                raise ConvergenceError("evaluation episodes are not paired")
            steps = [
                json.loads(line)
                for line in (destination / f"{arm}-{seed}/steps.jsonl").read_text().splitlines()
            ]
            for row in rows:
                key = (arm, seed, int(row["evaluation_seed"]))
                item = indexed.get(key)
                if (
                    item is None
                    or {k: "" if item.get(k) is None else str(item[k]) for k in row} != row
                ):
                    raise ConvergenceError("matched outcomes do not reconstruct from CSV")
                crown = any(
                    s["evaluation_seed"] == key[2]
                    and s["is_crown_jewel"]
                    and s["success"]
                    and s["access_gained"] > 0
                    for s in steps
                )
                if item["crown_jewel_reach"] is not crown:
                    raise ConvergenceError("crown-jewel outcome lacks step evidence")


def verify(config_path: Path, root: Path) -> dict:
    c = load_config(config_path)
    load_criterion(ROOT / c["criterion"])
    out = root / c["id"]
    status = {
        "config": c["id"],
        "configuration_valid": True,
        "convergence_established": False,
        "registration": "not executed",
        "training": "not executed",
        "assessment": "not executed",
    }
    if (out / "registration.json").exists():
        verify_registration(config_path, root)
        status["registration"] = "verified"
        completed = [s for s in c["training_seeds"] if (out / f"shaped-{s}").exists()]
        for seed in completed:
            verify_run(out / f"shaped-{seed}", c, seed)
        status["training"] = f"{len(completed)}/10 seeds verified"
    if (out / "assessment.json").exists():
        report = verify_assessment(out)
        status["assessment"] = "verified"
        status["convergence_established"] = report["passed"]
    if (root / "first_stable.json").exists():
        require_frozen(config_path, root)
        status["frozen_candidate"] = "verified"
    if (out / "final-evaluation").exists():
        verify_evaluation(out, root, c)
        status["final_evaluation"] = "verified"
    return status
