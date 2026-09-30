"""Descriptive corrected-vs-frozen comparison after verified evaluation.

Never reads discovery-derived metrics, trains policies or edits frozen files.
This command is deliberately unavailable until all registered runs evaluate.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

from rlredteam.convergence import ROOT, ConvergenceError, load_config, read_json, sha, write_new
from rlredteam.convergence_runner import verify_evaluation, verify_registration

CONFIG = ROOT / "configs/experiments/experiment_01_discovery_corrected_v3.yaml"
OUTPUT = ROOT / "results/confirmatory_discovery_v3"
BASELINE = ROOT / "results/convergence_v2/final_baseline_evaluation/report"
METRICS = ("success_rate", "steps_to_success", "native_return", "failed_actions",
           "mean_cvss_exploited")


def _checked_baseline_csv(name: str, manifest: dict) -> list[dict]:
    path = BASELINE / name
    if sha(path) != manifest["files"][name]:
        raise ConvergenceError(f"reviewed baseline file changed: {name}")
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _summary(rows: list[dict]) -> dict:
    successes = [row for row in rows if row["goal_reached"]]
    cvss = [float(row["mean_cvss_exploited"]) for row in rows
            if row["mean_cvss_exploited"] not in (None, "")]
    return {
        "success_rate": len(successes) / len(rows),
        "steps_to_success": mean(float(row["length"]) for row in successes)
        if successes else None,
        "native_return": mean(float(row["native_return"]) for row in rows),
        "failed_actions": mean(float(row["failed_actions"]) for row in rows),
        "mean_cvss_exploited": mean(cvss) if cvss else None,
    }


def compare(config_path: Path = CONFIG, root: Path = OUTPUT) -> dict:
    config = load_config(config_path)
    registration, out = verify_registration(config_path, root)
    verify_evaluation(out, root, config)
    manifest = read_json(BASELINE / "reviewed_manifest.json")
    if sha(BASELINE / "reviewed_results.json") != manifest["files"]["reviewed_results.json"]:
        raise ConvergenceError("reviewed baseline result changed")
    baseline = {}
    corrected = {}
    expected = {(seed, episode) for seed in config["training_seeds"]
                for episode in config["evaluation_seeds"]}
    for arm in ("shaped", "sparse"):
        raw = _checked_baseline_csv(f"{arm}_evaluation.csv", manifest)
        if len(raw) != len(expected) or {
            (int(row["training_seed"]), int(row["evaluation_seed"])) for row in raw
        } != expected:
            raise ConvergenceError("frozen evaluation rows do not match the protocol")
        baseline[arm] = [
            {
                "training_seed": int(row["training_seed"]),
                "evaluation_seed": int(row["evaluation_seed"]),
                "goal_reached": row["goal_reached"] == "True",
                "length": int(row["length"]),
                "native_return": float(row["native_return"]),
                "failed_actions": int(row["failed_actions"]),
                "mean_cvss_exploited": row["mean_cvss_exploited"],
            }
            for row in raw
        ]
        rows = []
        for seed in config["training_seeds"]:
            directory = out / f"final-evaluation/{arm}-{seed}"
            with (directory / "evaluation.csv").open(newline="") as handle:
                evaluations = list(csv.DictReader(handle))
            failed = defaultdict(int)
            for line in (directory / "steps.jsonl").read_text().splitlines():
                step = json.loads(line)
                if not step["success"]:
                    failed[int(step["evaluation_seed"])] += 1
            for row in evaluations:
                row["goal_reached"] = row["goal_reached"] == "True"
                row["failed_actions"] = failed[int(row["evaluation_seed"])]
                rows.append(row)
        if len(rows) != len(expected) or {
            (int(row["training_seed"]), int(row["evaluation_seed"])) for row in rows
        } != expected:
            raise ConvergenceError("corrected evaluation rows do not match the protocol")
        corrected[arm] = rows
    table = []
    for arm in ("shaped", "sparse"):
        old, new = _summary(baseline[arm]), _summary(corrected[arm])
        for metric in METRICS:
            table.append({
                "arm": arm,
                "metric": metric,
                "frozen": old[metric],
                "corrected": new[metric],
                "corrected_minus_frozen": new[metric] - old[metric]
                if new[metric] is not None and old[metric] is not None else None,
            })
    return {
        "schema": "security-rl-discovery-confirmatory-comparison-v1",
        "basis": (
            "Descriptive native outcomes; matched training/evaluation seeds; "
            "no statistical test"
        ),
        "discovery_derived_metrics": "excluded from frozen baseline claims",
        "table": table,
        "provenance": {
            "registration_sha256": sha(out / "registration.json"),
            "baseline_reviewed_manifest_sha256": sha(BASELINE / "reviewed_manifest.json"),
            "baseline_reviewed_results_sha256": sha(BASELINE / "reviewed_results.json"),
            "corrected_matched_outcomes_sha256": sha(
                out / "final-evaluation/matched_outcomes.json"
            ),
            "code_commit": registration["inputs"]["git_commit"],
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--output-root", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if args.output_root.resolve() != OUTPUT.resolve():
        parser.error("comparison must use the separate confirmatory output root")
    report = compare(args.config, args.output_root)
    path = args.output_root / load_config(args.config)["id"] / "comparison.json"
    write_new(path, report)
    print(json.dumps({"path": str(path), "table": report["table"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
