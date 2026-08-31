# Phase 3 completion audit — hidden on-prem topology generalisation

## Verdict

The research milestone is complete for its declared scope: a simulation-only,
partially observable, seeded on-premises topology distribution. A single
mask-aware PPO checkpoint was trained across topology seeds 1–60 and evaluated
without updates on disjoint validation seeds 1001–1020 and held-out test seeds
2001–2020. The canonical local evidence passes the repeatable completion gate,
including PostgreSQL reconstruction.

This verdict does **not** claim production deployment readiness, real-network
capability, or a statistically confirmed sparse-versus-shaped reward effect.
The fixed baseline is preserved and methodologically packaged, but all 20 of
its training runs are explicitly marked non-converged, so that comparison is
provisional.

## Repeatable gate

Run the complete artifact and database verification through Podman:

```bash
make onprem-verify
```

The command is read-only with respect to experiment evidence. It verifies the
checkpoint and files already present under the gated `runs/` and `results/`
directories, then queries the existing PostgreSQL rows. A clean clone without
the privately held checkpoint fails closed; this is intentional because policy
weights are not released through Git.

The implementation is split between:

- `src/rlredteam/enterprise/completion.py` — reusable evidence invariants;
- `scripts/verify_onprem_completion.py` — JSON-emitting command-line gate;
- `tests/test_onprem_completion.py` — fail-closed and canonical-evidence tests;
- `make onprem-verify` — Podman-only operator entry point.

## Requirement verdict

| Requirement | Verdict | Evidence checked by the gate |
|---|---|---|
| Fixed baseline preserved | PASS | 20 matched sparse/shaped runs, 200 frozen evaluation episodes, isolated reward hashes |
| Hidden random topology | PASS | regenerated and hash-matched 60 train, 20 validation and 20 test topologies |
| Explicit `AgentKnowledge` | PASS | training manifest and evaluation action-mask provenance |
| Zero topology leakage | PASS | architecture regression tests plus knowledge-derived mask invariant |
| Discovery updates knowledge | PASS | causal trajectories and environment tests |
| Interleaved discovery/exploitation/pivot | PASS | successful trace-derived paths from unseen topologies |
| Train across distribution | PASS | clean 50,176-step MaskablePPO checkpoint over seeds 1–60 |
| Frozen unseen-seed evaluation | PASS | unchanged policy hash before/after validation and test |
| Attack-path reconstruction | PASS | 20 validation and 20 test causal paths ending in successful asset access |
| PostgreSQL reconstruction | PASS | experiment → run → episode → ordered step records agree with files |
| Simulation-only boundary | PASS | simulator modules contain no network client imports; runtime network is internal |
| Weight release gated | PASS | checkpoint is untracked and `runs/` remains gitignored |

## Canonical identities

| Item | Value |
|---|---|
| Training commit | `c369d4c51c92da31a24b85eadbc2257b1c9d7186` |
| Actual training steps | 50,176 |
| Experiment config SHA-256 | `228d57a9bfd6eaf4bfbcfb39d973b75211592c3ae06eb09a32ab289900cb2540` |
| Topology config SHA-256 | `98fa63b38c6412cb84b7f334050f5b3164cd94a82c58cdd900ba6c1c88256411` |
| Synthetic vulnerability manifest SHA-256 | `586a16454e0daae4c8414f769376770f45432256ba2cad5a86692f0941529363` |
| Checkpoint SHA-256 | `a38a86c585e5532c5a0478db8f1f5745b9b8a8964b570e95b40ba497c4eb9626` |
| Policy SHA-256 | `545676ca4c1419c3bff8bdf5464e8876a8164dfd3f6ed5d9ab446da86a4594c4` |

## Evaluation and reconstruction evidence

| Split | Topologies | Success | Invalid actions | Causal paths | PostgreSQL identity | Ordered steps |
|---|---:|---:|---:|---:|---|---:|
| Validation | 20 | 20/20 | 0 | 20 | experiment 246 / run 130 | 1,714 |
| Held-out test | 20 | 20/20 | 0 | 20 | experiment 247 / run 131 | 1,848 |

Every stored path is non-empty, contains only successful state-changing events,
and ends with `access_asset` setting `goal_reached=true`. Every trajectory
topology hash matches the preregistered seed-to-hash distribution.

## Fixed-baseline interpretation

The fixed-topology `experiment_01` package remains a valid controlled pipeline:

- sparse and shaped arms use matched training seeds 42–51;
- topology, CVE snapshot, PPO configuration and budgets match within each pair;
- only reward mode/configuration differs between arms;
- each frozen checkpoint has ten dedicated evaluation episodes using seeds
  1001–1010;
- analysis consumes those evaluation outputs and applies paired statistics,
  effect sizes, bootstrap confidence intervals and multiple-test correction.

However, `analysis.json` lists all 20 runs as non-converged. Existing tables and
figures are therefore descriptive/provisional evidence, not a confirmatory
claim that reward shaping caused better generalisation.

## Production-readiness assessment

The secure runtime controls appropriate to this research simulator are in
place: pinned dependencies, unprivileged containers, dropped capabilities,
`no-new-privileges`, read-only code/configuration mounts, injected secrets,
loopback-only PostgreSQL exposure, an internal runtime network, bounded topology
and episode sizes, and provenance hashes.

The system is **not certified production-ready**. A production service would
still require deployment-specific SLOs, load/soak testing, backup-and-restore
evidence, vulnerability scanning/signing of the final images, operational
monitoring and alerting, and a tested rollback procedure. Those controls are
outside this dissertation simulation milestone and must not be inferred from a
passing research evidence gate.

## Validation record — 2026-08-31

- `make onprem-verify`: PASS, including live PostgreSQL reconstruction;
- focused completion verifier: 2 passed in 2.30 seconds;
- focused on-prem test group: 29 passed; measured runtime 7h40m52s;
- Ruff for the new verifier, CLI and tests: PASS after formatting correction;
- full clean Podman suite, GUI suite, Python compilation and Compose validation:
  recorded in the completion commit after the final gate.

The unusually long focused run included real training tests. Operators should
use `tests/test_onprem_completion.py` for the fast evidence-integrity check and
reserve the complete suite for a release gate.

## Next phase boundary

No real-network scanning or autonomous exploitation is authorized by this
completion. Cross-profile legacy/cloud/hybrid simulation and the backend-driven
desktop demonstration are documented separately in Phase 4 and Phase 5. Any
new phase must retain the fixed baseline, hidden-truth boundary, simulation-only
runtime and gated policy release.
