"""CP-27/28/29 — metrics, convergence and statistics from raw run-level data.

Everything here is computed from the episode records a run actually produced.
Nothing is read from a summary that some earlier step wrote, because a summary
is a claim and this module exists to check claims.

The protocol -- which metrics are primary, which test, what correction -- lives
in configs/metrics.yaml and is fixed before results are inspected.
"""

from __future__ import annotations

import json
import math
import statistics
from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_METRICS_CONFIG = REPO_ROOT / "configs" / "metrics.yaml"


def load_protocol(path: Path = DEFAULT_METRICS_CONFIG) -> dict:
    protocol = yaml.safe_load(Path(path).read_text())
    validate_protocol(protocol)
    return protocol


def validate_protocol(protocol: dict) -> None:
    """Fail closed on fixed-budget designs that could enable post-hoc filtering."""
    if not isinstance(protocol, dict):
        raise ValueError("metrics protocol must be a mapping")
    design = protocol.get("design")
    if design is None:  # Historical protocols remain byte-for-byte interpretable.
        return
    if not isinstance(design, dict):
        raise ValueError("metrics design must be a mapping")
    required = {
        "protocol_version",
        "estimand",
        "unit_of_analysis",
        "analysis_population",
        "convergence_role",
        "training_budget_timesteps",
        "learning_rate_schedule",
        "topology_seed",
        "prior_lower_budget_exposure",
    }
    missing = sorted(required - set(design))
    if missing:
        raise ValueError(f"fixed-budget design missing fields: {', '.join(missing)}")
    expected = {
        "protocol_version": "fixed_budget_v1",
        "estimand": "frozen_policy_performance_after_fixed_training_budget",
        "unit_of_analysis": "training_seed",
        "analysis_population": "all_completed_planned_runs",
        "convergence_role": "diagnostic_only",
        "prior_lower_budget_exposure": "disclosed",
    }
    mismatches = [
        f"{key}={design.get(key)!r}; expected {value!r}"
        for key, value in expected.items()
        if design.get(key) != value
    ]
    if mismatches:
        raise ValueError("invalid fixed-budget design: " + "; ".join(mismatches))
    if not isinstance(design["training_budget_timesteps"], int) or design[
        "training_budget_timesteps"
    ] <= 0:
        raise ValueError("fixed-budget training_budget_timesteps must be positive")
    if design["learning_rate_schedule"] not in ("constant", "linear_to_zero"):
        raise ValueError("invalid fixed-budget learning_rate_schedule")
    if not isinstance(design["topology_seed"], int):
        raise ValueError("fixed-budget topology_seed must be an integer")


# -- per-run metrics --------------------------------------------------------


@dataclass
class RunMetrics:
    """Metrics for one run, computed from its episodes."""

    run_name: str
    reward_mode: str
    seed: int
    topology_seed: int
    episodes: int
    native_return: float
    native_return_std: float
    native_return_median: float
    success_rate: float
    steps_to_goal: float | None
    median_steps_to_goal: float | None
    mean_cvss_exploited: float | None
    hvt_hit_rate: float
    max_cvss_exploited: float | None
    hosts_compromised: float
    discovered_hosts: float
    tactic_coverage: float
    total_shaped_return: float
    failure_reasons: dict[str, int]
    converged: bool
    convergence_detail: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def _window(episodes: list[dict], fraction: float) -> list[dict]:
    size = max(1, int(len(episodes) * fraction))
    return episodes[-size:]


def assess_convergence(episodes: list[dict], protocol: dict) -> tuple[bool, dict]:
    """CP-29 — a testable stability criterion, not a last-window summary.

    Three conditions, all of which must hold over the final window: enough
    episodes to say anything, a relative spread below threshold, and no
    meaningful downward trend. A run still improving steeply has not plateaued
    either, so a large gain over the preceding window also fails.
    """
    rules = protocol["convergence"]
    method = rules.get("method", "raw_episode_v1")
    if method == "block_means_v2":
        return _assess_block_mean_stability(episodes, rules)
    if method != "raw_episode_v1":
        raise ValueError(f"unsupported convergence method: {method}")
    detail: dict = {"criterion": rules}

    if len(episodes) < rules["min_episodes"]:
        detail["reason"] = f"only {len(episodes)} episodes; need {rules['min_episodes']}"
        return False, detail

    tail = _window(episodes, rules["window"])
    if len(tail) < rules["min_episodes"]:
        tail = episodes[-rules["min_episodes"] :]

    values = [e["native_return"] for e in tail]
    mean = statistics.fmean(values)
    spread = statistics.pstdev(values)
    scale = abs(mean) if abs(mean) > 1e-9 else 1.0
    relative_std = spread / scale

    # Least-squares slope over the window, normalised by the mean so the
    # threshold is scale-free.
    n = len(values)
    xs = list(range(n))
    x_mean = statistics.fmean(xs)
    denominator = sum((x - x_mean) ** 2 for x in xs) or 1.0
    slope = sum((x - x_mean) * (y - mean) for x, y in zip(xs, values, strict=True))
    slope /= denominator
    normalised_slope = slope / scale

    previous = episodes[-2 * len(tail) : -len(tail)] or tail
    previous_mean = statistics.fmean([e["native_return"] for e in previous])
    improvement = abs(mean - previous_mean) / (abs(previous_mean) or 1.0)

    detail.update(
        {
            "window_episodes": len(tail),
            "window_mean": round(mean, 3),
            "relative_std": round(relative_std, 4),
            "normalised_slope": round(normalised_slope, 6),
            "improvement_vs_previous_window": round(improvement, 4),
        }
    )

    checks = {
        "spread_within_threshold": relative_std <= rules["max_relative_std"],
        "no_negative_trend": normalised_slope >= rules["max_negative_slope"] - 1e-9,
        "plateaued": improvement <= rules["max_improvement_ratio"],
    }
    detail["checks"] = checks
    if not all(checks.values()):
        detail["reason"] = ", ".join(k for k, ok in checks.items() if not ok)
    return all(checks.values()), detail


def _assess_block_mean_stability(
    episodes: list[dict], rules: dict
) -> tuple[bool, dict]:
    """Assess stability of aggregate return blocks, not raw episode variance.

    Episodic cyber simulations remain stochastic after a policy has stopped
    materially changing. Requiring the coefficient of variation of individual
    episode returns to become small therefore confuses environmental variance
    with learning stability. Version 2 preregisters consecutive block means as
    the unit of analysis and still reports raw spread as a diagnostic.
    """
    detail: dict = {"criterion": rules}
    minimum = int(rules["min_episodes"])
    block_count = int(rules["blocks"])
    if block_count < 2:
        raise ValueError("block_means_v2 requires at least two blocks")
    if len(episodes) < minimum:
        detail["reason"] = f"only {len(episodes)} episodes; need {minimum}"
        return False, detail

    tail_size = max(minimum, int(len(episodes) * float(rules["window"])))
    tail = episodes[-tail_size:]
    usable = len(tail) - (len(tail) % block_count)
    tail = tail[-usable:]
    block_size = usable // block_count
    values = [float(episode["native_return"]) for episode in tail]
    block_means = [
        statistics.fmean(values[index : index + block_size])
        for index in range(0, usable, block_size)
    ]
    mean = statistics.fmean(values)
    scale = max(abs(mean), float(rules["return_scale_floor"]))
    relative_range = (max(block_means) - min(block_means)) / scale

    xs = list(range(block_count))
    x_mean = statistics.fmean(xs)
    block_mean = statistics.fmean(block_means)
    denominator = sum((x - x_mean) ** 2 for x in xs) or 1.0
    slope = sum(
        (x - x_mean) * (value - block_mean)
        for x, value in zip(xs, block_means, strict=True)
    ) / denominator
    normalised_slope = slope / scale
    midpoint = block_count // 2
    previous_mean = statistics.fmean(block_means[:midpoint])
    final_mean = statistics.fmean(block_means[midpoint:])
    half_change = abs(final_mean - previous_mean) / max(
        abs(previous_mean), float(rules["return_scale_floor"])
    )
    raw_relative_std = statistics.pstdev(values) / scale

    detail.update(
        {
            "window_episodes": len(values),
            "block_size": block_size,
            "block_means": [round(value, 3) for value in block_means],
            "window_mean": round(mean, 3),
            "raw_relative_std_diagnostic": round(raw_relative_std, 4),
            "block_mean_relative_range": round(relative_range, 4),
            "normalised_block_slope": round(normalised_slope, 6),
            "half_window_change": round(half_change, 4),
        }
    )
    checks = {
        "block_range_within_threshold": relative_range
        <= float(rules["max_block_mean_relative_range"]),
        "block_trend_within_threshold": abs(normalised_slope)
        <= float(rules["max_abs_normalised_block_slope"]),
        "halves_equivalent": half_change <= float(rules["max_half_window_change"]),
    }
    detail["checks"] = checks
    if not all(checks.values()):
        detail["reason"] = ", ".join(key for key, passed in checks.items() if not passed)
    return all(checks.values()), detail


def metrics_for_run(
    run_name: str,
    reward_mode: str,
    seed: int,
    topology_seed: int,
    episodes: list[dict],
    protocol: dict,
    training_episodes: list[dict] | None = None,
) -> RunMetrics:
    """Compute every metric for one run from its raw episodes."""
    if not episodes:
        raise ValueError(f"{run_name}: no episodes to analyse")

    threshold = protocol["high_value_target"]["cvss_threshold"]
    successes = [e for e in episodes if e["goal_reached"]]
    cvss = [e["mean_cvss_exploited"] for e in episodes if e.get("mean_cvss_exploited") is not None]
    maxima = [e["max_cvss_exploited"] for e in episodes if e.get("max_cvss_exploited") is not None]
    convergence_rows = training_episodes if training_episodes is not None else episodes
    converged, detail = assess_convergence(convergence_rows, protocol)
    native_returns = [e["native_return"] for e in episodes]
    successful_lengths = [e["length"] for e in successes]
    failure_reasons: dict[str, int] = {}
    for episode in episodes:
        if episode["goal_reached"]:
            continue
        reason = str(episode.get("terminal_reason", "unknown"))
        failure_reasons[reason] = failure_reasons.get(reason, 0) + 1
    supported_tactics = {"recon", "exploit", "privesc"}
    observed_tactics = {
        tactic
        for episode in episodes
        for tactic in episode.get("tactics", [])
        if tactic in supported_tactics
    }

    return RunMetrics(
        run_name=run_name,
        reward_mode=reward_mode,
        seed=seed,
        topology_seed=topology_seed,
        episodes=len(episodes),
        native_return=statistics.fmean(native_returns),
        native_return_std=(statistics.stdev(native_returns) if len(native_returns) > 1 else 0.0),
        native_return_median=statistics.median(native_returns),
        success_rate=len(successes) / len(episodes),
        # Conditional on success: averaging over failures rewards an arm that
        # fails fast, because a truncated episode has a fixed length.
        steps_to_goal=(statistics.fmean(successful_lengths) if successes else None),
        median_steps_to_goal=(statistics.median(successful_lengths) if successes else None),
        mean_cvss_exploited=statistics.fmean(cvss) if cvss else None,
        hvt_hit_rate=(sum(1 for m in maxima if m >= threshold) / len(episodes) if maxima else 0.0),
        max_cvss_exploited=max(maxima) if maxima else None,
        hosts_compromised=statistics.fmean([e.get("hosts_compromised", 0) for e in episodes]),
        discovered_hosts=statistics.fmean([e.get("discovered_hosts", 0) for e in episodes]),
        tactic_coverage=len(observed_tactics) / len(supported_tactics),
        total_shaped_return=statistics.fmean([e.get("shaped_return", 0.0) for e in episodes]),
        failure_reasons=failure_reasons,
        converged=converged,
        convergence_detail=detail,
    )


# -- statistics -------------------------------------------------------------


@dataclass
class Comparison:
    metric: str
    n_pairs: int
    mean_a: float
    mean_b: float
    sd_a: float
    sd_b: float
    median_a: float
    median_b: float
    difference: float
    t_statistic: float | None
    p_value: float | None
    p_bonferroni: float | None
    cohens_d: float | None
    ci_low: float | None
    ci_high: float | None
    significant: bool
    assumption_warning: str | None

    def to_dict(self) -> dict:
        return asdict(self)


def paired_comparison(
    metric: str,
    arm_a: dict[int, float],
    arm_b: dict[int, float],
    protocol: dict,
) -> Comparison:
    """Paired comparison across seeds present in BOTH arms.

    Pairing is by seed: each seed is one network seen by both arms, so the
    between-network variance cancels. Seeds missing from either arm are dropped
    rather than compared unpaired, which would silently change the test.
    """
    from scipy import stats

    shared = sorted(set(arm_a) & set(arm_b))
    a = [arm_a[s] for s in shared]
    b = [arm_b[s] for s in shared]
    rules = protocol["statistics"]

    if len(shared) < 2:
        return Comparison(
            metric=metric,
            n_pairs=len(shared),
            mean_a=statistics.fmean(a) if a else float("nan"),
            mean_b=statistics.fmean(b) if b else float("nan"),
            sd_a=statistics.stdev(a) if len(a) > 1 else 0.0,
            sd_b=statistics.stdev(b) if len(b) > 1 else 0.0,
            median_a=statistics.median(a) if a else float("nan"),
            median_b=statistics.median(b) if b else float("nan"),
            difference=float("nan"),
            t_statistic=None,
            p_value=None,
            p_bonferroni=None,
            cohens_d=None,
            ci_low=None,
            ci_high=None,
            significant=False,
            assumption_warning="fewer than two matched pairs",
        )

    result = stats.ttest_rel(b, a)
    differences = [x - y for x, y in zip(b, a, strict=True)]
    mean_difference = statistics.fmean(differences)
    spread = statistics.stdev(differences) if len(differences) > 1 else 0.0
    # Paired Cohen's d: mean difference over the SD of the differences.
    d = mean_difference / spread if spread > 1e-12 else None

    low, high = _bootstrap_ci(differences, rules["bootstrap_samples"], rules["confidence"])
    p_bonferroni = (
        min(1.0, float(result.pvalue) * rules["family_size"])
        if metric in protocol["primary_metrics"]
        else None
    )
    threshold = rules["alpha"]
    assumption_warning = None
    if len(differences) >= 3 and spread > 1e-12:
        normality_p = float(stats.shapiro(differences).pvalue)
        if normality_p < threshold:
            assumption_warning = (
                f"paired differences fail Shapiro-Wilk normality at alpha={threshold} "
                f"(p={normality_p:.4g}); interpret the paired t-test cautiously"
            )

    return Comparison(
        metric=metric,
        n_pairs=len(shared),
        mean_a=statistics.fmean(a),
        mean_b=statistics.fmean(b),
        sd_a=statistics.stdev(a),
        sd_b=statistics.stdev(b),
        median_a=statistics.median(a),
        median_b=statistics.median(b),
        difference=mean_difference,
        t_statistic=float(result.statistic),
        p_value=float(result.pvalue),
        p_bonferroni=p_bonferroni,
        cohens_d=d,
        ci_low=low,
        ci_high=high,
        significant=bool(
            (p_bonferroni if p_bonferroni is not None else float(result.pvalue)) < threshold
        ),
        assumption_warning=assumption_warning,
    )


def _bootstrap_ci(
    values: list[float], samples: int, confidence: float
) -> tuple[float | None, float | None]:
    if len(values) < 2:
        return None, None
    import random

    rng = random.Random(42)  # fixed, so a reported CI is reproducible
    means = []
    n = len(values)
    for _ in range(samples):
        means.append(statistics.fmean([values[rng.randrange(n)] for _ in range(n)]))
    means.sort()
    tail = (1.0 - confidence) / 2.0
    low = means[max(0, math.floor(tail * samples))]
    high = means[min(samples - 1, math.ceil((1.0 - tail) * samples) - 1)]
    return low, high


# -- report -----------------------------------------------------------------


def analyse(
    runs: list[RunMetrics], protocol: dict, arm_a: str = "sparse", arm_b: str = "shaped"
) -> dict:
    """Full comparison of two arms across every metric in the protocol."""
    by_arm: dict[str, dict[int, RunMetrics]] = {}
    for run in runs:
        by_arm.setdefault(run.reward_mode, {})[run.seed] = run

    comparisons = []
    metrics = protocol["primary_metrics"] + protocol.get("descriptive_metrics", [])
    for metric in metrics:
        a = {
            seed: getattr(run, metric)
            for seed, run in by_arm.get(arm_a, {}).items()
            if getattr(run, metric, None) is not None
        }
        b = {
            seed: getattr(run, metric)
            for seed, run in by_arm.get(arm_b, {}).items()
            if getattr(run, metric, None) is not None
        }
        if a and b:
            comparisons.append(paired_comparison(metric, a, b, protocol).to_dict())

    not_converged = [r.run_name for r in runs if not r.converged]
    expected_seeds = set(protocol["evaluation"]["seeds"])
    complete = all(
        expected_seeds <= set(by_arm.get(arm, {})) for arm in (arm_a, arm_b)
    )
    report = {
        "protocol": protocol,
        "arms": {arm_a: len(by_arm.get(arm_a, {})), arm_b: len(by_arm.get(arm_b, {}))},
        "runs": [r.to_dict() for r in runs],
        "comparisons": comparisons,
        "not_converged": not_converged,
        # CP-26: completeness is reported, so a missing seed is visible rather
        # than quietly reducing n.
        "expected_seeds": protocol["evaluation"]["seeds"],
        "complete": complete,
    }
    design = protocol.get("design")
    if design:
        report["fixed_budget_interpretation"] = {
            "protocol_version": design["protocol_version"],
            "estimand": design["estimand"],
            "unit_of_analysis": design["unit_of_analysis"],
            "analysis_population": design["analysis_population"],
            "convergence_role": design["convergence_role"],
            "all_planned_pairs_present": complete,
            "training_stability_warning": bool(not_converged),
            "status": (
                "complete_with_training_stability_warning"
                if complete and not_converged
                else "complete"
                if complete
                else "incomplete"
            ),
        }
    return report


def format_table(report: dict) -> str:
    """The results table, in native units, ready to paste into the write-up."""
    lines = [
        f"{'metric':<24}{'sparse':>12}{'shaped':>12}{'Δ':>12}"
        f"{'t':>9}{'p':>10}{'p_bonf':>10}{'d':>8}",
        "-" * 97,
    ]
    for c in report["comparisons"]:

        def fmt(value, spec=">12.3f"):
            return (
                "—".rjust(int(spec.split(">")[1].split(".")[0]))
                if value is None
                else format(value, spec)
            )

        lines.append(
            f"{c['metric']:<24}{fmt(c['mean_a'])}{fmt(c['mean_b'])}"
            f"{fmt(c['difference'])}{fmt(c['t_statistic'], '>9.2f')}"
            f"{fmt(c['p_value'], '>10.4f')}{fmt(c['p_bonferroni'], '>10.4f')}"
            f"{fmt(c['cohens_d'], '>8.2f')}"
        )
    if report["not_converged"]:
        lines.append("")
        interpretation = report.get("fixed_budget_interpretation", {})
        if interpretation.get("convergence_role") == "diagnostic_only":
            lines.append(
                "TRAINING STABILITY WARNING (diagnostic; runs retained by the "
                "preregistered fixed-budget estimand): "
                + ", ".join(report["not_converged"])
            )
        else:
            lines.append(f"NOT CONVERGED: {', '.join(report['not_converged'])}")
    if not report["complete"]:
        lines.append("")
        lines.append("INCOMPLETE: not every expected seed is present in both arms")
    return "\n".join(lines)


def write_report(report: dict, out_dir: Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "analysis.json").write_text(json.dumps(report, indent=2, default=str))
    (out_dir / "results_table.txt").write_text(format_table(report) + "\n")
