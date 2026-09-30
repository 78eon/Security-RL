"""Read-only PostgreSQL reconstruction check for completed confirmatory runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import psycopg

from rlredteam.convergence import ROOT, ConvergenceError, load_config, read_json
from rlredteam.convergence_runner import verify_evaluation, verify_registration, verify_run
from rlredteam.storage.postgres_logger import connection_string

CONFIG = ROOT / "configs/experiments/experiment_01_discovery_corrected_v3.yaml"
OUTPUT = ROOT / "results/confirmatory_discovery_v3"


def verify_database(config_path: Path = CONFIG, root: Path = OUTPUT) -> dict:
    config = load_config(config_path)
    _, out = verify_registration(config_path, root)
    verify_evaluation(out, root, config)
    totals = {"training_runs": 0, "evaluation_runs": 0,
              "training_episodes": 0, "evaluation_episodes": 0, "steps": 0}
    with psycopg.connect(connection_string(), connect_timeout=5) as connection:
        connection.execute("SET TRANSACTION READ ONLY")
        connection.execute("SET LOCAL statement_timeout = '15s'")
        for arm in ("shaped", "sparse"):
            for seed in config["training_seeds"]:
                path = out / f"{arm}-{seed}"
                verify_run(path, config, seed, arm)
                manifest = read_json(path / "manifest.json")
                metadata = read_json(
                    out / f"final-evaluation/{arm}-{seed}/evaluation_metadata.json"
                )
                for phase, experiment_id, run_id in (
                    ("training", manifest["database_experiment_id"],
                     manifest["database_run_id"]),
                    ("evaluation", metadata["database_experiment_id"],
                     metadata["database_evaluation_run_id"]),
                ):
                    row = connection.execute(
                        """SELECT e.name, r.designation, r.status, r.seed,
                                  r.evaluation_seeds, count(DISTINCT ep.id),
                                  count(s.id),
                                  (SELECT coalesce(sum(length), 0)
                                   FROM episodes WHERE run_id = r.id)
                           FROM experiments e
                           JOIN runs r ON r.experiment_id = e.id
                           LEFT JOIN episodes ep ON ep.run_id = r.id
                           LEFT JOIN steps s ON s.episode_id = ep.id
                           WHERE e.id = %s AND r.id = %s
                           GROUP BY e.name, r.id, r.designation, r.status, r.seed,
                                    r.evaluation_seeds""",
                        (experiment_id, run_id),
                    ).fetchone()
                    if row is None:
                        raise ConvergenceError(f"missing {phase} database run: {arm}-{seed}")
                    (
                        name, designation, status, recorded_seed,
                        eval_seeds, episodes, steps, lengths,
                    ) = row
                    expected_name = f"{config['id']}-{arm}-s{seed}"
                    if phase == "evaluation":
                        expected_name += "-evaluation"
                    if (
                        name != expected_name or designation != phase
                        or status != "complete" or recorded_seed != seed
                        or steps != lengths or episodes < 1
                        or (phase == "evaluation" and (
                            episodes != len(config["evaluation_seeds"])
                            or eval_seeds != config["evaluation_seeds"]
                        ))
                    ):
                        raise ConvergenceError(
                            f"{phase} database reconstruction mismatch: {arm}-{seed}"
                        )
                    totals[f"{phase}_runs"] += 1
                    totals[f"{phase}_episodes"] += episodes
                    totals["steps"] += steps
    return totals


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--output-root", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if args.output_root.resolve() != OUTPUT.resolve():
        parser.error("verifier must use the separate confirmatory output root")
    print(json.dumps(verify_database(args.config, args.output_root), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
