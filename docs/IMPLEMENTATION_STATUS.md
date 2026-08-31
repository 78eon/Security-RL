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
| Policy observation boundary | `topology.py`, `nasim_adapter.py`, `test_evaluation.py` | PASS | NASim is `fully_obs=False`; initial observation is `(234,)` with one known host and zeroed undiscovered rows; a spy policy receives observations only | none for fixed experiment |
| PPO training | `train.py` | FUNCTIONAL, CONVERGENCE FAIL | 20/20 checkpoints trained with exact hashes and database IDs; all fail the preregistered stability threshold | extend training under a preregistered amendment before confirmatory claims |
| Sparse reward | `reward.py`, `configs/sparse.yaml` | PASS | only 0 or 1 crown-jewel signal; reward unit tests pass | document formula |
| Shaped reward | `reward.py`, `configs/shaped.yaml` | PASS | CVSS, tactic, goal, failure and anti-farming tests pass | document exact formula and limitations |
| Reward isolation | `experiment.py`, reward configs | PASS | exact resolved-config test proves arms differ only in `mode`; every child receives the frozen manifest | final grid pending |
| CVE snapshot | `data/cve_catalogue.sqlite`, manifest | PASS | 16 frozen records; canonical digest `c14575707519311e90a571e1b27d0bed16b2d10e41757f60a135ab8756137205`; one field contract | none |
| PostgreSQL logging | `storage/schema.sql`, `postgres_logger.py` | PASS | canonical DB reconstructs 20 experiments, 40 complete runs, 3,055 episodes and 847,540 steps; evaluation subset is 20 runs/200 episodes/30,842 steps | none |
| Training/evaluation separation | `evaluation.py`, `experiment.py` | PASS | 20 frozen checkpoints, held-out seeds, no gradient updates, and all before/after policy hashes match | none |
| Paired statistics | `analyse.py`, `metrics.yaml` | PASS WITH WARNINGS | complete 10-pair evaluation analysis; Shapiro-Wilk warnings are emitted for non-normal differences | results remain provisional because training did not converge |
| Figures | `reporting.py` | PASS | three plots generated only from evaluation outcomes | none |
| Attack-path extraction | `reporting.py` | PASS | 200 typed paths contain observed successful progress only, never hidden topology | none |
| Trajectory validation | `reporting.py` | PASS | 0 paid errors, 0 repeated paid actions, 0 high-return failures; longest identical streak 4 | none for current traces |
| Results package | `results/experiment_01` | PASS, PROVISIONAL | 20 raw evaluation sets, hashes, checkpoint metadata, statistics, figures, paths and README trace to commit `7fcfde8` | convergence blocks confirmatory interpretation |
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

1. Do not present `experiment_01` as a converged confirmatory comparison: all
   20 runs fail the preregistered training-stability criterion.
2. Preregister a longer training budget or an independently justified stopping
   rule, then regenerate the grid before dissertation-level causal claims.

## Amendment outcome — 2026-08-31

Phase 6 tested that blocker prospectively without rewriting the historical
audit. Constant-rate pilots at 200k and 400k demonstrated learning but failed
the preregistered two-arm stability gate. A single preregistered
linear-to-zero optimizer pilot at 200k then produced one stable sparse policy
and one non-stable shaped policy. The final matched grid was therefore not run.

The authoritative decision evidence is in
`docs/PHASE_06_CONVERGENCE_AMENDMENT.md` and
`docs/PHASE_06B_OPTIMIZER_STABILITY.md`. The research status remains:
functional PPO and valid frozen evaluation infrastructure, but no converged
confirmatory sparse-versus-shaped comparison under the capped protocols.
