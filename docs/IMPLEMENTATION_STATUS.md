# Essential Research Implementation Status

Audit date: 2026-08-27
Audited commit: `74b730913fbcf387ea18e2fce7faf10599e419c9`
Scope: fixed-topology NASim PPO, sparse versus shaped reward only.

## Baseline evidence

The untouched full suite was run with `make test` before audit changes:

```text
312 passed, 10 skipped, 2 failed, 2 warnings in 79.13s
```

Both failures were research-provenance failures: training emitted
`cve_database_hash`, while the tested and database-facing contract requires
`cve_manifest_sha256`.

A focused PostgreSQL run (`pytest tests/test_postgres_logger.py`) produced
`6 passed, 1 failed`. The end-to-end failure showed that training persisted the
realised topology hash (`f875a25adf37ee34`) in the column named
`topology_config_hash`, whose correct value is `2aa12404d40a23d0`.

## Essential audit

| Component | File(s) | Status | Evidence | Missing work |
|---|---|---|---|---|
| Fixed topology | `configs/topology.yaml`, `topology.py` | PASS | topology seed defaults to 42; realised hash `f875a25adf37ee34`; config hash `2aa12404d40a23d0` | none for fixed experiment |
| Policy observation boundary | `topology.py`, `nasim_adapter.py` | PASS, TEST GAP | NASim is `fully_obs=False`; initial observation is `(234,)` with one known host and zeroed undiscovered rows | add an invariant test and policy-input spy |
| PPO training | `train.py` | FUNCTIONAL, PROVENANCE FAIL | short PPO training completes and saves CSV, summary, manifest, checkpoint | repair CVE manifest key and DB identifiers |
| Sparse reward | `reward.py`, `configs/sparse.yaml` | PASS | only 0 or 1 crown-jewel signal; reward unit tests pass | document formula |
| Shaped reward | `reward.py`, `configs/shaped.yaml` | PASS | CVSS, tactic, goal, failure and anti-farming tests pass | document exact formula and limitations |
| Reward isolation | `run_ablation.py`, reward configs | PASS, TEST GAP | resolved configs differ in `mode`; topology/CVE/PPO inputs are shared by construction | add exact resolved-config comparison test; enforce frozen input in every child run |
| CVE snapshot | `data/cve_catalogue.sqlite`, manifest | PASS | 16 frozen records; canonical digest `c14575707519311e90a571e1b27d0bed16b2d10e41757f60a135ab8756137205` | standardise field name |
| PostgreSQL logging | `storage/schema.sql`, `postgres_logger.py` | PARTIAL / E2E FAIL | synthetic experiment/episode/step reconstruction passes | separate topology hashes; add run identity/status and evaluation designation; persist reward components |
| Training/evaluation separation | `run_ablation.py` | FAIL | current analysis reads training `episodes.csv` | frozen-policy evaluation loop and raw evaluation output |
| Paired statistics | `analyse.py`, `metrics.yaml` | PARTIAL | paired t-test, paired d, bootstrap CI and Bonferroni are implemented | feed evaluation outcomes; emit SD/median and assumption warnings |
| Figures | — | FAIL | no reproducible figure generator | generate figures from raw evaluation data |
| Attack-path extraction | — | FAIL | GUI derives display rows directly from DB steps | typed path model and JSON export from evaluation traces |
| Trajectory validation | — | FAIL | no systematic repeated-action/farming/loop report | generate checks and document inspected traces |
| Results package | `runs/_analysis` | FAIL | existing artifacts trace to older commits and training outcomes | create `results/experiment_01/` from current clean commit |
| Checkpoint gating | `.gitignore`, `release_weights.py` | PASS | weights under ignored `runs/`; explicit approval required for packaging | none |
| GUI | `gui/` | OUT OF ESSENTIAL PATH | read-only backend snapshot | must not block experiment work |
| Hybrid/live extensions | `enterprise/`, lab scripts | ISOLATED / DEFERRED | separate entry points and lab image | do not use for Essential experiment |

## One episode trace

1. `train.build_env()` creates a seeded NASim environment and wraps it in
   `RewardWrapper`; PPO receives the wrapper's observation and action spaces.
2. `reset(seed)` resets NASim state and the reward engine's per-episode
   anti-farming history, returning only NASim's partial observation.
3. `PPO` supplies an action index to `RewardWrapper.step()`.
4. The wrapper range-checks the index, executes `env.step()`, and converts the
   action result into an `AttackEvent`.
5. `RewardEngine.score()` returns sparse, shaped, or native reward plus an
   attributed `RewardBreakdown`.
6. PPO receives only `(observation, configured reward, done flags, info)`; the
   callback consumes `info` after the policy decision.
7. `EpisodeCollector` accumulates native/configured returns and optional step
   records, distinguishes goal termination from step-limit truncation, writes
   CSV, and queues PostgreSQL records.
8. On completion, the checkpoint, manifest, snapshot, and training summary are
   written under ignored `runs/`.

## Existing artifacts are not current evidence

The stored 20-run grid largely records commit `455a3d3e...`, while the audited
code is `74b7309...`. It also analyzes training trajectories rather than a
frozen-policy evaluation phase. Those files may guide debugging, but they are
not dissertation evidence for the current system.

## Minimal blocker set

1. Repair the manifest and database topology/CVE provenance contracts.
2. Enforce the frozen experiment in every training child process.
3. Add dedicated frozen-policy evaluation and typed attack-path export.
4. Analyze evaluation results, generate figures and a traceable results tree.
5. Add targeted tests, rerun the full suite, then run the controlled grid from
   a clean commit.
