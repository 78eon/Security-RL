#!/usr/bin/env python3
"""Export real data as JSON fixtures for UI design and GUI development.

Two consumers:

* a designer, who needs true content in mockups -- real run names, real CVE
  ids, real negative rewards -- because placeholder text hides the layout
  problems that actually bite (decimal alignment, 64-character hashes);
* the GUI's own tests, which need deterministic data without a live database.

    python scripts/export_fixtures.py --out tests/fixtures
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import psycopg  # noqa: E402

from rlredteam.catalogue import CVECatalogue  # noqa: E402
from rlredteam.manifest import digest  # noqa: E402
from rlredteam.storage.postgres_logger import connection_string  # noqa: E402
from rlredteam.topology import TopologyConfig  # noqa: E402


def _rows(conn, sql: str, params: tuple = ()) -> list[dict]:
    cur = conn.execute(sql, params)
    columns = [d[0] for d in cur.description]
    return [dict(zip(columns, row, strict=True)) for row in cur.fetchall()]


def export(out_dir: Path) -> dict[str, int]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, int] = {}

    def dump(name: str, payload) -> None:
        path = out_dir / f"{name}.json"
        path.write_text(json.dumps(payload, indent=2, default=str))
        written[name] = len(payload) if isinstance(payload, list) else 1

    with psycopg.connect(connection_string()) as conn:
        # -- screen 1: run browser -------------------------------------
        experiments = _rows(
            conn,
            """
            SELECT e.id, e.name, e.reward_mode, e.config_hash,
                   e.topology_config_hash, e.cve_manifest_sha256, e.git_sha,
                   e.seed_set, e.created_at,
                   count(ep.id)                        AS episode_count,
                   avg(ep.native_reward)               AS mean_native_reward,
                   avg(ep.goal_reached::int)           AS success_rate,
                   avg(ep.length)                      AS mean_length
            FROM experiments e
            LEFT JOIN episodes ep ON ep.experiment_id = e.id
            GROUP BY e.id ORDER BY e.id
            """,
        )
        dump("experiments", experiments)

        # -- screen 2: charts ------------------------------------------
        dump(
            "episodes",
            _rows(
                conn,
                """
                SELECT ep.experiment_id, e.name AS run_name, e.reward_mode,
                       ep.seed, ep.topology_seed, ep.episode_idx,
                       ep.total_reward, ep.native_reward, ep.length,
                       ep.terminal_state, ep.goal_reached,
                       ep.exploited_hosts, ep.mean_cvss_exploited
                FROM episodes ep
                JOIN experiments e ON e.id = ep.experiment_id
                ORDER BY ep.experiment_id, ep.episode_idx
                """,
            ),
        )

        # -- screen 3: replay ------------------------------------------
        # One complete successful episode: the shortest goal-reaching one, so
        # the fixture stays small while still containing a full attack path.
        target = _rows(
            conn,
            """
            SELECT ep.id, ep.length, e.name AS run_name
            FROM episodes ep
            JOIN experiments e ON e.id = ep.experiment_id
            WHERE ep.goal_reached
              AND EXISTS (SELECT 1 FROM steps s WHERE s.episode_id = ep.id)
            ORDER BY ep.length ASC LIMIT 1
            """,
        )
        if target:
            episode_id = target[0]["id"]
            dump(
                "replay_episode",
                {
                    "episode": _rows(
                        conn, "SELECT * FROM episodes WHERE id = %s", (episode_id,)
                    )[0],
                    "steps": _rows(
                        conn,
                        "SELECT step_idx, action_name, action_kind, tactic, "
                        "technique_id, target_subnet, target_host, success, "
                        "reward, native_reward, cve_id, cvss_base "
                        "FROM steps WHERE episode_id = %s ORDER BY step_idx",
                        (episode_id,),
                    ),
                },
            )

    # -- static reference data -----------------------------------------
    catalogue = CVECatalogue.open_default()
    dump(
        "cve_catalogue",
        [
            {
                "cve_id": r.cve_id,
                "kind": r.kind,
                "base_score": r.base_score,
                "base_severity": r.base_severity,
                "vector": r.vector,
                "cwe": r.cwe,
                "note": r.note,
            }
            for r in catalogue.all_records()
        ],
    )

    topology = TopologyConfig.from_yaml()
    dump(
        "topology",
        {
            "config_hash": topology.config_hash(),
            "cve_manifest_sha256": digest(catalogue),
            "num_hosts": topology.num_hosts,
            "num_services": topology.num_services,
            "num_exploits": topology.num_exploits,
            "num_privescs": topology.num_privescs,
            "step_limit": topology.step_limit,
            "crown_jewel_value": topology.r_sensitive,
        },
    )
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "tests" / "fixtures")
    args = parser.parse_args()

    written = export(args.out)
    print(f"wrote {len(written)} fixtures to {args.out}")
    for name, count in written.items():
        print(f"  {name + '.json':<24} {count} record(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
