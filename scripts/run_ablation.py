#!/usr/bin/env python3
"""Shaped-vs-sparse ablation harness -- STUB.

Enumerates the run grid and asserts the two arms are genuinely comparable:
identical topology config, identical CVE catalogue, identical seeds, differing
in nothing but the reward. It deliberately does NOT train -- it calls
into `train.py` once that exists (see docs/PPO_BRIEF.md).

    python scripts/run_ablation.py --dry-run     # print the grid and checks
"""

from __future__ import annotations

import argparse
import itertools
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from rlredteam.catalogue import CVECatalogue  # noqa: E402
from rlredteam.manifest import digest  # noqa: E402
from rlredteam.reward import RewardConfig  # noqa: E402
from rlredteam.topology import TopologyConfig  # noqa: E402

SEEDS = list(range(42, 52))
ARMS = {
    "shaped": REPO_ROOT / "configs" / "shaped.yaml",
    "sparse": REPO_ROOT / "configs" / "sparse.yaml",
}


@dataclass(frozen=True)
class Run:
    arm: str
    seed: int
    topology_seed: int
    reward_config_path: Path

    @property
    def name(self) -> str:
        return f"{self.arm}-s{self.seed}-t{self.topology_seed}"


def build_grid(vary_topology: bool) -> list[Run]:
    """One run per (arm, seed).

    ``vary_topology`` decides what the experiment measures. Fixed topology seed
    isolates training variance on one network; varying it measures
    generalisation across networks. These answer different questions -- pick one
    and state it in the write-up.
    """
    runs = []
    for arm, seed in itertools.product(ARMS, SEEDS):
        runs.append(
            Run(
                arm=arm,
                seed=seed,
                topology_seed=seed if vary_topology else 42,
                reward_config_path=ARMS[arm],
            )
        )
    return runs


def assert_arms_comparable() -> dict[str, str]:
    """Fail loudly if the arms differ in anything but the reward mode.

    The ablation's entire validity rests on this. Checked before any compute is
    spent rather than discovered afterwards.
    """
    topology = TopologyConfig.from_yaml()
    catalogue = CVECatalogue.open_default()
    configs = {arm: RewardConfig.from_yaml(path) for arm, path in ARMS.items()}

    shaped, sparse = configs["shaped"], configs["sparse"]
    differences = []
    for field in ("cve_scale", "tactic_bonuses", "crown_jewel", "failed_action", "weight"):
        if getattr(shaped, field) != getattr(sparse, field):
            differences.append(field)
    if differences:
        raise AssertionError(
            f"arms differ in {differences}; they must differ only in reward mode"
        )
    if shaped.mode == sparse.mode:
        raise AssertionError("both arms have the same reward mode")

    return {
        "topology_config_hash": topology.config_hash(),
        "cve_manifest_sha256": digest(catalogue),
        "shaped_mode": str(shaped.mode),
        "sparse_mode": str(sparse.mode),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", default=True)
    parser.add_argument(
        "--vary-topology",
        action="store_true",
        help="use a different topology per seed (generalisation) rather than one fixed",
    )
    args = parser.parse_args()

    shared = assert_arms_comparable()
    print("arms are comparable:")
    for key, value in shared.items():
        print(f"  {key:<24} {value}")

    runs = build_grid(args.vary_topology)
    print(f"\ngrid: {len(runs)} runs ({len(ARMS)} arms x {len(SEEDS)} seeds)")
    print(f"topology mode: {'varying per seed' if args.vary_topology else 'fixed at 42'}")
    for run in runs:
        print(f"  {run.name:<24} reward={run.reward_config_path.name}")

    print(
        "\nSTUB: no training performed. "
        "Wire train.py in here once written (docs/PPO_BRIEF.md)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
