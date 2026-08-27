# Trajectory Validation

Status: **complete for experiment_01 evaluation traces**.

The canonical evaluation contains 200 held-out episodes and 30,842 step rows
from 20 frozen checkpoints. Every checkpoint policy hash is unchanged before
and after evaluation. The policy receives only NASim's partial observation;
trace metadata is consumed after each decision.

## Automated findings

| Check | Result |
|---|---:|
| invalid/impossible actions with positive reward | 0 |
| repeated paid action-targets | 0 |
| high policy return without goal completion | 0 |
| maximum identical-action streak | 4 |
| evaluation failures / step-limit loops | 0 / 200 |
| non-increasing crown-jewel payments | 0 |

There are 493 crown-jewel payments across 400 episode/host pairs. Ninety-three
pairs pay twice, and every one is a legitimate strict access increase
`NONE → USER → ROOT`; no later action at the same access level is paid.

The first generated experiment was rejected by this audit because it exposed
repeated crown-jewel payment on already-compromised hosts. Its files and
database rows are retained under `invalid_*_pre_reward_fix` labels and are not
part of the canonical result package. The corrected code adds an explicit
`reward_paid` trace field and a regression test for the invariant.

## Evidence locations

- all trace-derived paths: `results/experiment_01/trajectories/attack_paths.json`;
- representative sparse and shaped paths: `sparse_examples.json` and
  `shaped_examples.json`;
- per-episode checks: `validation.json`;
- source steps: `results/experiment_01/raw/*/steps.jsonl`.

Attack paths contain only observed successful discovery/access progress. They
are not inferred from hidden topology state. All evaluation episodes succeeded,
so the representative files contain successes and highest-return examples but
no fabricated failure example.
