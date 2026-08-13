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
    return yaml.safe_load(Path(path).read_text())


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
    success_rate: float
    steps_to_goal: float | None
    mean_cvss_exploited: float | None
    hvt_hit_rate: float
    max_cvss_exploited: float | None
    hosts_compromised: float
    total_shaped_return: float
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
    detail: dict = {"criterion": rules}

    if len(episodes) < rules["min_episodes"]:
        detail["reason"] = (
            f"only {len(episodes)} episodes; need {rules['min_episodes']}"
        )
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


def metrics_for_run(
    run_name: str,
    reward_mode: str,
    seed: int,
    topology_seed: int,
    episodes: list[dict],
    protocol: dict,
) -> RunMetrics:
    """Compute every metric for one run from its raw episodes."""
    if not episodes:
        raise ValueError(f"{run_name}: no episodes to analyse")

    threshold = protocol["high_value_target"]["cvss_threshold"]
    successes = [e for e in episodes if e["goal_reached"]]
    cvss = [
        e["mean_cvss_exploited"] for e in episodes
        if e.get("mean_cvss_exploited") is not None
    ]
    maxima = [
        e["max_cvss_exploited"] for e in episodes
        if e.get("max_cvss_exploited") is not None
    ]
    converged, detail = assess_convergence(episodes, protocol)

    return RunMetrics(
        run_name=run_name,
        reward_mode=reward_mode,
        seed=seed,
        topology_seed=topology_seed,
        episodes=len(episodes),
        native_return=statistics.fmean([e["native_return"] for e in episodes]),
        success_rate=len(successes) / len(episodes),
        # Conditional on success: averaging over failures rewards an arm that
        # fails fast, because a truncated episode has a fixed length.
        steps_to_goal=(
            statistics.fmean([e["length"] for e in successes]) if successes else None
        ),
        mean_cvss_exploited=statistics.fmean(cvss) if cvss else None,
        hvt_hit_rate=(
            sum(1 for m in maxima if m >= threshold) / len(episodes) if maxima else 0.0
        ),
        max_cvss_exploited=max(maxima) if maxima else None,
        hosts_compromised=statistics.fmean(
            [e.get("hosts_compromised", 0) for e in episodes]
        ),
        total_shaped_return=statistics.fmean(
            [e.get("shaped_return", 0.0) for e in episodes]
        ),
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
    difference: float
    t_statistic: float | None
    p_value: float | None
    p_bonferroni: float | None
    cohens_d: float | None
    ci_low: float | None
    ci_high: float | None
    significant: bool

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
            metric=metric, n_pairs=len(shared),
            mean_a=statistics.fmean(a) if a else float("nan"),
            mean_b=statistics.fmean(b) if b else float("nan"),
            difference=float("nan"), t_statistic=None, p_value=None,
            p_bonferroni=None, cohens_d=None, ci_low=None, ci_high=None,
            significant=False,
        )

    result = stats.ttest_rel(b, a)
    differences = [x - y for x, y in zip(b, a, strict=True)]
    mean_difference = statistics.fmean(differences)
    spread = statistics.stdev(differences) if len(differences) > 1 else 0.0
    # Paired Cohen's d: mean difference over the SD of the differences.
    d = mean_difference / spread if spread > 1e-12 else None

    low, high = _bootstrap_ci(
        differences, rules["bootstrap_samples"], rules["confidence"]
    )
    p_bonferroni = (
        min(1.0, float(result.pvalue) * rules["family_size"])
        if metric in protocol["primary_metrics"] else None
    )
    threshold = rules["alpha"]

    return Comparison(
        metric=metric,
        n_pairs=len(shared),
        mean_a=statistics.fmean(a),
        mean_b=statistics.fmean(b),
        difference=mean_difference,
        t_statistic=float(result.statistic),
        p_value=float(result.pvalue),
        p_bonferroni=p_bonferroni,
        cohens_d=d,
        ci_low=low,
        ci_high=high,
        significant=bool(
            (p_bonferroni if p_bonferroni is not None else float(result.pvalue))
            < threshold
        ),
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
    return {
        "protocol": protocol,
        "arms": {arm_a: len(by_arm.get(arm_a, {})), arm_b: len(by_arm.get(arm_b, {}))},
        "runs": [r.to_dict() for r in runs],
        "comparisons": comparisons,
        "not_converged": not_converged,
        # CP-26: completeness is reported, so a missing seed is visible rather
        # than quietly reducing n.
        "expected_seeds": protocol["evaluation"]["seeds"],
        "complete": all(
            set(protocol["evaluation"]["seeds"]) <= set(by_arm.get(arm, {}))
            for arm in (arm_a, arm_b)
        ),
    }


def format_table(report: dict) -> str:
    """The results table, in native units, ready to paste into the write-up."""
    lines = [
        f"{'metric':<24}{'sparse':>12}{'shaped':>12}{'Δ':>12}"
        f"{'t':>9}{'p':>10}{'p_bonf':>10}{'d':>8}",
        "-" * 97,
    ]
    for c in report["comparisons"]:
        def fmt(value, spec=">12.3f"):
            return "—".rjust(int(spec.split(">")[1].split(".")[0])) if value is None \
                else format(value, spec)

        lines.append(
            f"{c['metric']:<24}{fmt(c['mean_a'])}{fmt(c['mean_b'])}"
            f"{fmt(c['difference'])}{fmt(c['t_statistic'], '>9.2f')}"
            f"{fmt(c['p_value'], '>10.4f')}{fmt(c['p_bonferroni'], '>10.4f')}"
            f"{fmt(c['cohens_d'], '>8.2f')}"
        )
    if report["not_converged"]:
        lines.append("")
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
