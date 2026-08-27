"""Reproducible figures and trace-derived trajectory evidence."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rlredteam.analyse import RunMetrics
    from rlredteam.experiment import ExperimentConfig


@dataclass(frozen=True, slots=True)
class AttackPathStep:
    step: int
    action: str
    action_kind: str
    target: tuple[int, int] | None
    newly_discovered: int
    access_gained: int
    cve_id: str | None
    cvss_base: float | None
    tactic: str | None
    technique_id: str | None
    is_crown_jewel: bool


@dataclass(frozen=True, slots=True)
class AttackPath:
    run_name: str
    reward_mode: str
    training_seed: int
    evaluation_seed: int
    goal_reached: bool
    terminal_reason: str
    native_return: float
    policy_return: float
    progress_steps: tuple[AttackPathStep, ...]

    def to_dict(self) -> dict:
        return asdict(self)


def extract_attack_path(episode: dict, steps: list[dict]) -> AttackPath:
    """Extract observed progress only; never infer a path from topology state."""
    progress = []
    for step in sorted(steps, key=lambda row: int(row["step"])):
        if not step["success"]:
            continue
        if int(step.get("newly_discovered", 0)) <= 0 and int(step.get("access_gained", 0)) <= 0:
            continue
        target = step.get("target")
        progress.append(
            AttackPathStep(
                step=int(step["step"]),
                action=str(step["action"]),
                action_kind=str(step["action_kind"]),
                target=tuple(target) if target else None,
                newly_discovered=int(step.get("newly_discovered", 0)),
                access_gained=int(step.get("access_gained", 0)),
                cve_id=step.get("cve_id"),
                cvss_base=step.get("cvss_base"),
                tactic=step.get("tactic"),
                technique_id=step.get("technique_id"),
                is_crown_jewel=bool(step.get("is_crown_jewel", False)),
            )
        )
    return AttackPath(
        run_name=str(episode["run_name"]),
        reward_mode=str(episode["reward_mode"]),
        training_seed=int(episode["training_seed"]),
        evaluation_seed=int(episode["evaluation_seed"]),
        goal_reached=_as_bool(episode["goal_reached"]),
        terminal_reason=str(episode["terminal_reason"]),
        native_return=float(episode["native_return"]),
        policy_return=float(episode["policy_return"]),
        progress_steps=tuple(progress),
    )


def validate_trajectory(episode: dict, steps: list[dict]) -> dict:
    actions = [(row["action"], tuple(row["target"] or ())) for row in steps]
    longest_streak = 0
    current_streak = 0
    previous = None
    seen: set[tuple[str, tuple]] = set()
    repeated_paid = 0
    positive_errors = 0
    for row, key in zip(steps, actions, strict=True):
        current_streak = current_streak + 1 if key == previous else 1
        longest_streak = max(longest_streak, current_streak)
        previous = key
        if key in seen and float(row["policy_reward"]) > 0:
            repeated_paid += 1
        seen.add(key)
        if row.get("error") and float(row["policy_reward"]) > 0:
            positive_errors += 1
    return {
        "run_name": episode["run_name"],
        "training_seed": int(episode["training_seed"]),
        "evaluation_seed": int(episode["evaluation_seed"]),
        "goal_reached": _as_bool(episode["goal_reached"]),
        "terminal_reason": episode["terminal_reason"],
        "length": int(episode["length"]),
        "distinct_action_targets": len(set(actions)),
        "repeated_action_targets": len(actions) - len(set(actions)),
        "longest_identical_action_streak": longest_streak,
        "repeated_action_targets_with_positive_reward": repeated_paid,
        "invalid_or_impossible_actions_with_positive_reward": positive_errors,
        "high_policy_return_without_goal": (
            not _as_bool(episode["goal_reached"]) and float(episode["policy_return"]) > 0
        ),
    }


def write_trajectory_evidence(config: ExperimentConfig, result_root: Path) -> dict:
    trajectories_dir = result_root / "trajectories"
    trajectories_dir.mkdir(parents=True, exist_ok=True)
    paths: list[AttackPath] = []
    validations: list[dict] = []

    for arm in config.arms:
        for seed in config.training_seeds:
            run_name = config.run_name(arm, seed)
            raw = result_root / "raw" / run_name
            episodes = _read_csv(raw / "evaluation.csv")
            steps = _read_jsonl(raw / "steps.jsonl")
            grouped: dict[int, list[dict]] = {}
            for step in steps:
                grouped.setdefault(int(step["evaluation_seed"]), []).append(step)
            for episode in episodes:
                episode_steps = grouped.get(int(episode["evaluation_seed"]), [])
                paths.append(extract_attack_path(episode, episode_steps))
                validations.append(validate_trajectory(episode, episode_steps))

    serialised = [path.to_dict() for path in paths]
    (trajectories_dir / "attack_paths.json").write_text(
        json.dumps(serialised, indent=2, sort_keys=True)
    )
    for arm in config.arms:
        candidates = [path for path in paths if path.reward_mode == arm]
        examples = _representative_paths(candidates)
        (trajectories_dir / f"{arm}_examples.json").write_text(
            json.dumps([path.to_dict() for path in examples], indent=2, sort_keys=True)
        )

    report = {
        "episodes_checked": len(validations),
        "positive_error_rewards": sum(
            row["invalid_or_impossible_actions_with_positive_reward"] for row in validations
        ),
        "repeated_paid_actions": sum(
            row["repeated_action_targets_with_positive_reward"] for row in validations
        ),
        "high_return_failures": sum(row["high_policy_return_without_goal"] for row in validations),
        "maximum_identical_action_streak": max(
            (row["longest_identical_action_streak"] for row in validations), default=0
        ),
        "episodes": validations,
    }
    (trajectories_dir / "validation.json").write_text(
        json.dumps(report, indent=2, sort_keys=True)
    )
    return report


def _representative_paths(paths: list[AttackPath]) -> list[AttackPath]:
    selected: list[AttackPath] = []
    success = next((path for path in paths if path.goal_reached), None)
    failure = next((path for path in paths if not path.goal_reached), None)
    highest = max(paths, key=lambda path: path.policy_return, default=None)
    for path in (success, failure, highest):
        if path is not None and path not in selected:
            selected.append(path)
    return selected


def write_figures(metrics: list[RunMetrics], out_dir: Path) -> None:
    """Create figures exclusively from per-checkpoint evaluation metrics."""
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    arms = ("sparse", "shaped")
    grouped = {arm: [row for row in metrics if row.reward_mode == arm] for arm in arms}
    plots = (
        ("success_rate.png", "Evaluation success rate", "Success rate", "success_rate"),
        (
            "steps_to_goal.png",
            "Evaluation steps to goal",
            "Steps (successful episodes)",
            "steps_to_goal",
        ),
        ("reward_distribution.png", "Native evaluation return", "Native return", "native_return"),
    )
    for filename, title, ylabel, field in plots:
        figure, axis = plt.subplots(figsize=(6.4, 4.2), constrained_layout=True)
        values = [
            [float(value) for row in grouped[arm] if (value := getattr(row, field)) is not None]
            for arm in arms
        ]
        axis.boxplot(values, tick_labels=arms, showmeans=True)
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", alpha=0.25)
        figure.savefig(out_dir / filename, dpi=160)
        plt.close(figure)


def _read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _read_jsonl(path: Path) -> list[dict]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _as_bool(value) -> bool:
    return value if isinstance(value, bool) else str(value).lower() == "true"
