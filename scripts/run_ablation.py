#!/usr/bin/env python3
"""CP-24/25 — the sparse-vs-shaped ablation runner.

Executes both arms across seeds 42-51 on ONE fixed topology, guarded by a hash
check that aborts when a run's inputs differ from the frozen experiment.

    python scripts/run_ablation.py --freeze          # write the frozen manifest
    python scripts/run_ablation.py --dry-run         # show the grid and the gate
    python scripts/run_ablation.py --timesteps 200000

The guardrail is the point. Two arms that differ in anything but the reward are
not an ablation, and the cheapest moment to discover that is before the compute
is spent rather than after the results are written up.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from rlredteam import provenance as prov  # noqa: E402
from rlredteam.analyse import (  # noqa: E402
    RunMetrics,
    analyse,
    format_table,
    load_protocol,
    metrics_for_run,
    write_report,
)
from rlredteam.catalogue import CVECatalogue  # noqa: E402
from rlredteam.manifest import digest  # noqa: E402
from rlredteam.reward import RewardConfig  # noqa: E402
from rlredteam.topology import TopologyConfig, describe, make_env  # noqa: E402
from rlredteam.train import DEFAULT_TOPOLOGY_SEED  # noqa: E402

ARMS = ("sparse", "shaped")
FROZEN_PATH = REPO_ROOT / "configs" / "frozen_experiment.json"


@dataclass(frozen=True)
class Run:
    arm: str
    seed: int
    topology_seed: int

    @property
    def name(self) -> str:
        return f"{self.arm}-s{self.seed}-t{self.topology_seed}"

    @property
    def config(self) -> Path:
        return REPO_ROOT / "configs" / f"{self.arm}.yaml"


# -- freezing ---------------------------------------------------------------


def current_hashes(topology_seed: int) -> dict:
    """The hashes that define this experiment, computed from current inputs."""
    topology_config = TopologyConfig.from_yaml()
    catalogue = CVECatalogue.open_default()
    described = describe(make_env(topology_config, topology_seed=topology_seed))

    from rlredteam.train import PPO_DEFAULTS

    return {
        "topology_seed": topology_seed,
        "topology_hash": prov.topology_hash(described),
        "topology_config_hash": topology_config.config_hash(),
        "environment_config_hash": prov.environment_config_hash(described),
        "cve_database_hash": digest(catalogue),
        "ppo_config_hash": prov._digest(PPO_DEFAULTS),
        "reward_config_hash": {
            arm: RewardConfig.from_yaml(REPO_ROOT / "configs" / f"{arm}.yaml").hash()
            for arm in ARMS
        },
        "dependency_lock_hash": prov.dependency_lock_hash(),
    }


def freeze(topology_seed: int) -> dict:
    frozen = current_hashes(topology_seed)
    frozen["frozen_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    frozen["git_commit"] = prov.git_commit()
    FROZEN_PATH.write_text(json.dumps(frozen, indent=2, sort_keys=True))
    return frozen


def check_against_frozen(topology_seed: int) -> tuple[bool, list[str]]:
    """CP-25 — abort when anything drifted from the frozen experiment."""
    if not FROZEN_PATH.exists():
        return False, [
            f"no frozen experiment at {FROZEN_PATH.relative_to(REPO_ROOT)}; "
            "run with --freeze first"
        ]
    frozen = json.loads(FROZEN_PATH.read_text())
    now = current_hashes(topology_seed)

    problems = []
    for key in (
        "topology_seed", "topology_hash", "topology_config_hash",
        "environment_config_hash", "cve_database_hash", "ppo_config_hash",
        "dependency_lock_hash",
    ):
        if frozen.get(key) != now.get(key):
            problems.append(f"{key}: frozen {frozen.get(key)} != current {now.get(key)}")
    for arm in ARMS:
        expected = frozen.get("reward_config_hash", {}).get(arm)
        actual = now["reward_config_hash"][arm]
        if expected != actual:
            problems.append(f"reward_config_hash[{arm}]: {expected} != {actual}")

    # The arms must differ from each other, or this compares a condition with
    # itself and no amount of statistics will reveal it.
    if now["reward_config_hash"]["sparse"] == now["reward_config_hash"]["shaped"]:
        problems.append("both arms have the same reward configuration")
    return not problems, problems


# -- execution --------------------------------------------------------------


def build_grid(seeds: list[int], topology_seed: int) -> list[Run]:
    """Both arms on every seed, all on ONE topology.

    The topology seed is constant across the grid by design (CP-11): varying it
    with the training seed would confound reward mode with network.
    """
    return [Run(arm, seed, topology_seed) for arm in ARMS for seed in seeds]


def execute(run: Run, timesteps: int, extra: list[str]) -> int:
    argv = [
        sys.executable, "-m", "rlredteam.train",
        "--seed", str(run.seed),
        "--topology-seed", str(run.topology_seed),
        "--timesteps", str(timesteps),
        "--reward-config", str(run.config),
        *extra,
    ]
    print(f"\n=== {run.name} ===", flush=True)
    return subprocess.run(argv, cwd=REPO_ROOT, check=False).returncode


def collect(runs: list[Run], protocol: dict) -> list[RunMetrics]:
    """Read every run's raw episodes back and compute its metrics."""
    import csv

    out: list[RunMetrics] = []
    for run in runs:
        path = REPO_ROOT / "runs" / run.name / "episodes.csv"
        if not path.exists():
            print(f"  missing episodes for {run.name} — excluded", file=sys.stderr)
            continue
        with path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        episodes = [
            {
                "native_return": float(r["native_return"]),
                "shaped_return": float(r["shaped_return"]),
                "length": int(r["length"]),
                "goal_reached": r["goal_reached"] == "True",
                "mean_cvss_exploited": (
                    float(r["mean_cvss_exploited"]) if r["mean_cvss_exploited"] else None
                ),
                "max_cvss_exploited": (
                    float(r["max_cvss_exploited"]) if r["max_cvss_exploited"] else None
                ),
                "hosts_compromised": int(r.get("hosts_compromised") or 0),
            }
            for r in rows
        ]
        if not episodes:
            continue
        out.append(
            metrics_for_run(
                run.name, run.arm, run.seed, run.topology_seed, episodes, protocol
            )
        )
    return out


def main() -> int:
    protocol = load_protocol()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+",
                        default=protocol["evaluation"]["seeds"])
    parser.add_argument("--topology-seed", type=int, default=DEFAULT_TOPOLOGY_SEED)
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--freeze", action="store_true",
                        help="write the frozen experiment manifest and exit")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--analyse-only", action="store_true",
                        help="skip training; analyse existing runs")
    parser.add_argument("--postgres", action="store_true", default=True)
    parser.add_argument("--log-steps", action="store_true", default=True)
    args = parser.parse_args()

    if args.freeze:
        frozen = freeze(args.topology_seed)
        print(f"froze experiment to {FROZEN_PATH.relative_to(REPO_ROOT)}")
        print(json.dumps(frozen, indent=2, sort_keys=True))
        return 0

    ok, problems = check_against_frozen(args.topology_seed)
    print("experiment guardrail:")
    if ok:
        print("  [PASS] every input matches the frozen experiment")
    else:
        for problem in problems:
            print(f"  [FAIL] {problem}")
        if not args.analyse_only:
            print("\nSTOP. Inputs differ from the frozen experiment; not training.")
            return 2

    grid = build_grid(args.seeds, args.topology_seed)
    print(f"\ngrid: {len(grid)} runs — {len(ARMS)} arms x {len(args.seeds)} seeds")
    print(f"topology seed held at {args.topology_seed} for every run")
    if args.dry_run:
        for run in grid:
            print(f"  {run.name}")
        return 0

    failures = []
    if not args.analyse_only:
        extra = []
        if args.postgres:
            extra.append("--postgres")
        if args.log_steps:
            extra.append("--log-steps")
        started = time.time()
        for index, run in enumerate(grid, start=1):
            print(f"[{index}/{len(grid)}] {run.name}", flush=True)
            if execute(run, args.timesteps, extra) != 0:
                failures.append(run.name)
                print(f"  FAILED: {run.name}", file=sys.stderr)
        print(f"\nwall clock: {(time.time() - started) / 60:.1f} min")

    metrics = collect(grid, protocol)
    if not metrics:
        print("no runs to analyse", file=sys.stderr)
        return 1

    report = analyse(metrics, protocol)
    report["failed_runs"] = failures
    write_report(report, REPO_ROOT / "runs" / "_analysis")
    print("\n" + format_table(report))
    print("\nwrote runs/_analysis/analysis.json and results_table.txt")
    if failures:
        print(f"FAILED RUNS (reported, not dropped): {failures}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
