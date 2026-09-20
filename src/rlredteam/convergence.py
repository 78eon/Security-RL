"""Versioned, fail-closed convergence protocol; no historical experiment writes."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np
import yaml

SCHEMA = "security-rl-convergence-v2"
SEEDS = list(range(42, 52))
ROOT = Path(__file__).resolve().parents[2]
BASE_CONFIG = ROOT / "configs/experiments/experiment_01_convergence_1m_v2.yaml"
DIAGNOSTICS = (
    "approx_kl",
    "clip_fraction",
    "explained_variance",
    "entropy_loss",
    "policy_gradient_loss",
    "value_loss",
    "learning_rate",
    "mean_episode_reward",
    "mean_episode_length",
)


class ConvergenceError(ValueError):
    """Protocol invalid, incomplete or not eligible for the next stage."""


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path: Path):
    return json.loads(Path(path).read_text())


def write_new(path: Path, value) -> None:
    """Exclusive create: never overwrite a registration, result or checkpoint."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    with path.open("x") as handle:
        handle.write(payload)


def load_config(path: Path, _seen=()) -> dict:
    path = Path(path).resolve()
    if path in _seen:
        raise ConvergenceError("cyclic parent configuration")
    config = yaml.safe_load(path.read_text())
    validate_config(config)
    if config["parent"]:
        parent = load_config(ROOT / config["parent"], (*_seen, path))
        validate_child(parent, config)
    return config


def validate_config(c: dict) -> None:
    from rlredteam.train import PPO_DEFAULTS

    expected = {
        "schema",
        "id",
        "stage",
        "parent",
        "reward",
        "topology",
        "topology_seed",
        "training_seeds",
        "evaluation_seeds",
        "total_timesteps",
        "ppo",
        "learning_rate_schedule",
        "normalization",
        "criterion",
        "evaluation",
    }
    if set(c) - {"rationale", "factor"} != expected:
        raise ConvergenceError("missing or unknown configuration keys")
    if c["schema"] not in {SCHEMA, "security-rl-convergence-v1"} or not re.fullmatch(
        r"[a-z][a-z0-9_]{1,100}", c["id"]
    ):
        raise ConvergenceError("invalid schema or candidate id")
    evaluation_seeds = list(range(1001, 1011)) if c["schema"] == SCHEMA else SEEDS
    if (
        c["reward"] != "shaped"
        or c["training_seeds"] != SEEDS
        or c["evaluation_seeds"] != evaluation_seeds
    ):
        raise ConvergenceError("shaped-first protocol requires seeds 42–51")
    if c["topology"] != "configs/topology.yaml" or c["topology_seed"] != 42:
        raise ConvergenceError("fixed baseline topology required")
    if c["total_timesteps"] != 1_000_000:
        raise ConvergenceError("this protocol requires a 1M-step budget")
    if c["evaluation"] != {
        "action_selection": "stochastic" if c["schema"] == SCHEMA else "deterministic",
        "postgres": True,
        "policy_updates": False,
    }:
        raise ConvergenceError("evaluation must match the versioned PostgreSQL protocol")
    baseline = PPO_DEFAULTS | {"normalize_advantage": True}
    if set(c["ppo"]) != set(baseline) or c["ppo"]["normalize_advantage"] is not True:
        raise ConvergenceError("explicit complete PPO configuration required")
    for name, value in c["ppo"].items():
        if name not in {"policy", "device", "normalize_advantage"}:
            if not isinstance(value, float | int) or not math.isfinite(value) or value < 0:
                raise ConvergenceError(f"invalid PPO {name}")
    for name in ("n_steps", "batch_size", "n_epochs"):
        if type(c["ppo"][name]) is not int or c["ppo"][name] < 2:
            raise ConvergenceError(f"invalid PPO {name}")
    if c["ppo"]["n_steps"] % c["ppo"]["batch_size"]:
        raise ConvergenceError("batch size must divide rollout length")
    if c["learning_rate_schedule"] not in {"constant", "linear_to_zero"}:
        raise ConvergenceError("unsupported LR schedule")
    norm = c["normalization"]
    if norm != {
        "enabled": norm.get("enabled"),
        "norm_obs": False,
        "gamma": 0.99,
        "epsilon": 1e-8,
        "clip_reward": 10.0,
    }:
        raise ConvergenceError("only reward-only VecNormalize protocol is supported")
    if type(norm["enabled"]) is not bool:
        raise ConvergenceError("normalization enabled must be boolean")
    if c["stage"] == "under_training":
        if (
            c["parent"] is not None
            or c["ppo"] != baseline
            or norm["enabled"]
            or c["learning_rate_schedule"] != "constant"
        ):
            raise ConvergenceError("under-training changes only the budget")
    elif c["stage"] not in {"normalization", "tuning"} or not c["parent"]:
        raise ConvergenceError("invalid stage or missing parent")


def validate_child(parent: dict, child: dict) -> None:
    ignored = {"id", "stage", "parent", "rationale", "factor"}
    changes = {key for key in parent if key not in ignored and parent[key] != child[key]}
    if child["stage"] == "normalization":
        if (
            parent["stage"] != "under_training"
            or changes != {"normalization"}
            or child["normalization"] != parent["normalization"] | {"enabled": True}
        ):
            raise ConvergenceError("normalization must be the only intervention")
        return
    if parent["stage"] not in {"normalization", "tuning"}:
        raise ConvergenceError("tuning follows the normalization assessment")
    factor = child.get("factor")
    factors = {"learning_rate": 1, "n_steps": 2, "batch_size": 2, "ent_coef": 3}
    if factor not in factors or not child.get("rationale", "").strip():
        raise ConvergenceError("declare one factor and diagnostic-based rationale")
    previous = factors.get(parent.get("factor"), 0)
    if factors[factor] < previous or factors[factor] > previous + 1:
        raise ConvergenceError("tuning order is LR, rollout/batch, entropy")
    ppo_changes = {k for k in parent["ppo"] if parent["ppo"][k] != child["ppo"][k]}
    allowed = {"ppo", "learning_rate_schedule"} if factor == "learning_rate" else {"ppo"}
    if not changes or changes - allowed or ppo_changes - {factor}:
        raise ConvergenceError("accidental multi-factor intervention")
    if factor != "learning_rate" and ppo_changes != {factor}:
        raise ConvergenceError("declared factor did not change")


def load_criterion(path: Path) -> dict:
    c = yaml.safe_load(Path(path).read_text())
    if c["schema"] not in {
        "security-rl-convergence-criterion-v1",
        "security-rl-convergence-criterion-v2",
    }:
        raise ConvergenceError("unsupported criterion schema")
    if (
        c["window"] != "final_third_of_actual_timesteps"
        or c["reward_basis"] != "unnormalised_shaped_episode_return"
    ):
        raise ConvergenceError("unsupported convergence window/reward basis")
    if c["late_collapse_reference"] != "middle_and_final_third_blocks":
        raise ConvergenceError("unsupported late-collapse reference")
    for name, value in c.items():
        if isinstance(value, int | float) and not math.isfinite(value):
            raise ConvergenceError(f"non-finite criterion: {name}")
    if not 1 <= c["min_passing_seeds"] <= 10 or c["blocks"] < 2:
        raise ConvergenceError("invalid seed/block criterion")
    minimum = c.get("min_episodes_per_window", c.get("min_episodes_per_third"))
    if minimum < c["blocks"] or c["reward_scale_floor"] <= 0:
        raise ConvergenceError("invalid episode count or scale")
    if c["schema"].endswith("v2") and (
        c["initial_reference"] != "first_completed_episodes"
        or type(c["initial_episodes"]) is not int
        or c["initial_episodes"] < minimum
    ):
        raise ConvergenceError("invalid initial episode reference")
    if c["smooth_episodes"] < 1:
        raise ConvergenceError("invalid smoothing window")
    for name in (
        "min_initial_to_final_gain_fraction",
        "max_final_block_range_fraction",
        "max_abs_final_slope_fraction",
        "max_late_drop_fraction",
        "max_seed_final_mean_range_fraction",
    ):
        if c[name] < 0:
            raise ConvergenceError(f"negative threshold: {name}")
    return c


def diagnostic_row(values: dict, *, update: int, timestep: int, episodes: list[dict]) -> dict:
    row = {"update": update, "timesteps": timestep}
    for key in DIAGNOSTICS[:7]:
        value = values.get(f"train/{key}")
        row[key] = float(value) if value is not None and math.isfinite(float(value)) else None
    tail = episodes[-100:]
    row["mean_episode_reward"] = (
        float(np.mean([e["shaped_return"] for e in tail])) if tail else None
    )
    row["mean_episode_length"] = float(np.mean([e["length"] for e in tail])) if tail else None
    return row


def csv_rows(path: Path) -> list[dict]:
    with Path(path).open() as handle:
        return list(csv.DictReader(handle))


def assess_seed(
    episodes: list[dict],
    diagnostics: list[dict],
    *,
    actual_steps: int,
    budget: int,
    criterion: dict,
) -> dict:
    """Time-based thirds, block means, scaled OLS slopes; missing evidence fails."""
    c = criterion
    reasons = []
    times = np.asarray([float(e["timesteps"]) for e in episodes])
    rewards = np.asarray([float(e["shaped_return"]) for e in episodes])
    if (
        not len(times)
        or not np.isfinite(times).all()
        or not np.isfinite(rewards).all()
        or np.any(np.diff(times) <= 0)
        or times[-1] > actual_steps
    ):
        return {"passed": False, "reasons": ["invalid or empty episode evidence"]}
    early_reference = c["schema"].endswith("v2")
    first = (
        rewards[: c["initial_episodes"]] if early_reference else rewards[times <= actual_steps / 3]
    )
    if early_reference and (
        len(first) < c["initial_episodes"] or times[len(first) - 1] >= 2 * actual_steps / 3
    ):
        return {
            "passed": False,
            "reasons": ["initial reference overlaps final third or is incomplete"],
        }
    final_mask = times >= 2 * actual_steps / 3
    final = rewards[final_mask]
    if actual_steps < budget:
        reasons.append("training budget incomplete")
    if min(len(first), len(final)) < c.get(
        "min_episodes_per_window", c.get("min_episodes_per_third")
    ):
        return {"passed": False, "reasons": reasons + ["insufficient episodes in time thirds"]}
    scale = max(c["reward_scale_floor"], abs(float(np.mean(final))))
    x = (times[final_mask] - 2 * actual_steps / 3) / (actual_steps / 3)
    blocks = [
        final[(x >= low) & (x <= high if high == 1 else x < high)]
        for low, high in zip(
            np.linspace(0, 1, c["blocks"] + 1)[:-1],
            np.linspace(0, 1, c["blocks"] + 1)[1:],
            strict=True,
        )
    ]
    if any(len(block) == 0 for block in blocks):
        return {"passed": False, "reasons": reasons + ["empty final-third time block"]}
    means = [float(np.mean(block)) for block in blocks]
    slope = float(np.polyfit(x, final, 1)[0]) / scale
    gain = (float(np.mean(final)) - float(np.mean(first))) / scale
    # A lower but flat final third must not hide a collapse at its boundary.
    middle_mask = (times >= actual_steps / 3) & (times < 2 * actual_steps / 3)
    middle = rewards[middle_mask]
    middle_x = (times[middle_mask] - actual_steps / 3) / (actual_steps / 3)
    middle_blocks = [
        middle[(middle_x >= low) & (middle_x < high)]
        for low, high in zip(
            np.linspace(0, 1, c["blocks"] + 1)[:-1],
            np.linspace(0, 1, c["blocks"] + 1)[1:],
            strict=True,
        )
    ]
    if any(len(block) == 0 for block in middle_blocks):
        return {"passed": False, "reasons": reasons + ["empty middle-third time block"]}
    reference = max([float(np.mean(block)) for block in middle_blocks] + means[:-1])
    late_drop = max(0.0, reference - means[-1]) / scale
    checks = {
        "reward did not rise": gain >= c["min_initial_to_final_gain_fraction"],
        "final-third block range not plateaued": (max(means) - min(means)) / scale
        <= c["max_final_block_range_fraction"],
        "final-third slope not plateaued": abs(slope) <= c["max_abs_final_slope_fraction"],
        "late collapse": late_drop <= c["max_late_drop_fraction"],
    }
    ev = [
        (float(r["timesteps"]), float(r["explained_variance"]))
        for r in diagnostics
        if r.get("explained_variance") not in (None, "")
    ]
    ev = [(t, v) for t, v in ev if t >= 2 * actual_steps / 3 and math.isfinite(v)]
    ev_mean = ev_slope = None
    if len(ev) >= 2 and len({t for t, _ in ev}) == len(ev):
        ev_mean = float(np.mean([v for _, v in ev]))
        ev_slope = float(
            np.polyfit([t / (actual_steps / 3) for t, _ in ev], [v for _, v in ev], 1)[0]
        )
    checks["explained variance not positive/stable"] = (
        ev_mean is not None
        and ev_mean > c["min_final_explained_variance"]
        and ev_slope >= c["min_explained_variance_slope"]
    )
    reasons.extend(reason for reason, ok in checks.items() if not ok)
    result = {
        "passed": not reasons,
        "reasons": reasons,
        "actual_steps": actual_steps,
        "final_third": {
            "count": len(final),
            "mean": float(np.mean(final)),
            "std": float(np.std(final)),
            "min": float(np.min(final)),
            "max": float(np.max(final)),
            "block_means": means,
            "scale": scale,
            "slope_fraction": slope,
        },
        "initial_to_final_gain_fraction": gain,
        "explained_variance": {"final_mean": ev_mean, "final_slope": ev_slope},
        "late_collapse": {
            "drop_fraction": late_drop,
            "passed": checks["late collapse"],
            "reference_block_mean": reference,
        },
    }
    if early_reference:
        result["initial_reference"] = {
            "method": c["initial_reference"],
            "episodes": len(first),
            "last_timestep": float(times[len(first) - 1]),
            "mean": float(np.mean(first)),
        }
    return result


def assess_all(per_seed: dict, criterion: dict) -> dict:
    complete = set(per_seed) == {str(s) for s in SEEDS}
    passing = sum(row["passed"] for row in per_seed.values())
    means = [r["final_third"]["mean"] for r in per_seed.values() if "final_third" in r]
    spread = (
        (
            (max(means) - min(means))
            / max(criterion["reward_scale_floor"], abs(float(np.mean(means))))
        )
        if len(means) == 10
        else None
    )
    reasons = []
    if not complete:
        reasons.append("missing required seeds")
    if passing < criterion["min_passing_seeds"]:
        reasons.append("too few passing seeds")
    if spread is None or spread > criterion["max_seed_final_mean_range_fraction"]:
        reasons.append("final rewards inconsistent across seeds")
    return {
        "schema": "security-rl-convergence-assessment-v2"
        if criterion["schema"].endswith("v2")
        else "security-rl-convergence-assessment-v1",
        "passed": not reasons,
        "reasons": reasons,
        "passing_seeds": passing,
        "per_seed": per_seed,
        "seed_final_mean_range_fraction": spread,
        "criterion": criterion,
        "criterion_sha256": digest(criterion),
    }


def diagnostic_advice(rows: list[dict], criterion: dict) -> list[str]:
    """Explicit decision support only; never schedules training or a sweep."""

    def series(key):
        return [
            float(row[key])
            for row in rows
            if row.get(key) not in (None, "") and math.isfinite(float(row[key]))
        ]

    c = criterion["diagnostics"]
    advice = []
    kl, clip, ev, entropy = (
        series(k) for k in ("approx_kl", "clip_fraction", "explained_variance", "entropy_loss")
    )
    if (
        kl
        and clip
        and np.mean(kl[-10:]) > c["high_approx_kl"]
        and np.mean(clip[-10:]) > c["high_clip_fraction"]
    ):
        advice.append("LR stage: consider 3e-4 -> 1e-4, optionally with preregistered linear decay")
    if ev and abs(float(np.mean(ev[-10:]))) <= c["near_zero_explained_variance"]:
        advice.append(
            "Investigate reward/value normalization; only after LR stage consider rollout length"
        )
    early = entropy[: max(1, len(entropy) // 3)]
    if len(early) > 1 and abs(early[-1]) < abs(early[0]) * c["early_entropy_remaining_fraction"]:
        advice.append(
            "Early entropy collapse: only after LR and rollout/batch stages "
            "consider higher ent_coef"
        )
    return advice or ["No threshold-triggered tuning advice; inspect time series before deciding"]
