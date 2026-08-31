# Phase 6 — Fixed-baseline convergence amendment

## Status

Preregistered before the Phase 6 pilot. No Phase 6 training or evaluation
result was inspected when this protocol was written.

## Research requirement

The fixed `experiment_01` pipeline is complete, reproducible and correctly
evaluates frozen policies. Its reward comparison remains provisional because
all 20 policies were trained for only 40,000 timesteps and the historical
stability rule marked every run non-converged. The next research task is to
establish a defensible fixed-budget comparison with explicit learning/stability
evidence, not to add another application feature.

The proposal requires PPO learning evidence, matched seeds 42–51, dedicated
evaluation, t-tests/effect sizes and transparent handling of convergence risk.
It does not justify changing the sparse/shaped hypothesis after results are
known.

## Why this is an amendment

The historical `raw_episode_v1` rule gates on the coefficient of variation of
individual episode returns. In the 40k package, the final-window relative
spread ranges from 0.50 to 2.00. Longer training does not necessarily remove
this variance: stochastic action sampling and variable successful path lengths
remain random even under unchanged policy parameters.

Consequently, raw episode variance mixes two different phenomena:

1. policy drift while PPO is still learning; and
2. irreducible outcome variance under a stable stochastic policy.

Changing the old result in place would erase research history. Phase 6 leaves
`configs/metrics.yaml`, `experiment_01`, its hashes and its interpretation
unchanged. The new `block_means_v2` method and new experiment IDs form an
explicit prospective amendment.

## Version 2 stability rule

For each training run:

1. take the final 40% of completed training episodes, with at least 100;
2. retain an equal number of episodes in four consecutive blocks;
3. compute mean native return per block;
4. use `max(abs(final-window mean), 100)` as the normalising scale—the floor is
   one native crown-jewel reward and prevents ratios exploding near zero;
5. require all three conditions:
   - block-mean relative range at most 0.35;
   - absolute normalised block slope at most 0.10 per block;
   - relative change between the first and second half at most 0.15.

Raw episode relative spread remains in the evidence as a diagnostic but is not
a stability gate. Passing means “the aggregate training curve is stable under
this declared rule,” not mathematical proof of global policy convergence.

## Controlled pilot

The pilot is isolated from the confirmatory seed set:

| Field | Value |
|---|---|
| Experiment | `experiment_01_convergence_pilot` |
| Training seed | 32 (excluded from final seeds 42–51) |
| Evaluation seeds | 901–910 (excluded from final 1001–1010) |
| Topology seed | 42, unchanged |
| Arms | sparse and shaped |
| Budget | 200,000 timesteps per arm |
| PPO/CVE/topology/reward definitions | unchanged |
| PostgreSQL steps | enabled, matching the final operational path |

The Phase 6 pilot may start only after its code, configuration, tests and frozen
input manifest are committed from a clean tree.

## Pilot decision rule

Proceed automatically to the full amended grid only if:

- both arms finish without NaN/Inf, provenance mismatch or database failure;
- frozen evaluation completes with no gradient updates and unchanged policy
  hashes;
- both arms show measurable learning: final 20% mean native training return is
  at least 10% better than the first 20%, normalised by
  `max(abs(first-window mean), 100)`;
- both arms pass `block_means_v2` at 200k;
- measured end-to-end runtime projects the 20-run grid to at most two hours on
  the available host;
- sparse/shaped resolved inputs still differ only by reward mode/hash.

If both arms learn but either is not stable, do **not** reinterpret the rule.
Create and freeze a 400k pilot extension before running it. If either arm does
not show learning, stop the grid and investigate the PPO setup using only the
excluded development seed. Final seeds and evaluation seeds must not be used
for tuning.

## 200k pilot result and administrative correction

The two policies completed training and frozen evaluation. Packaging then
failed closed because the pilot configuration declared development training
seed 32 and evaluation seeds 901–910, while its metrics file still declared the
final seed sets 42–51 and 1001–1010. The resulting analysis correctly reported
`complete: false`. The mismatch did not affect PPO inputs, environment steps,
checkpoint contents or evaluation actions, but it makes that first package
administratively incomplete.

The original frozen manifest is retained as
`frozen_experiment_01_convergence_pilot_attempt1.json`. A configuration-contract
check now rejects any future mismatch before training. The corrected pilot
metrics file changes only the declared seed lists; all outcome definitions,
statistics and stability thresholds are byte-for-byte equivalent in parsed
content.

Observed preregistered decision fields from the 200k attempt:

| Arm | Episodes | First→last 20% gain | v2 stability | Frozen evaluation |
|---|---:|---:|---|---|
| Sparse | 1,547 | +85.59% | FAIL | PASS |
| Shaped | 2,471 | +142.64% | PASS | PASS |

Both arms demonstrated learning and the runtime projects below two hours, but
sparse failed all three v2 stability checks. Therefore the preregistered rule
forbids the final 200k grid. The next allowed action is a from-scratch 400k
pilot extension using the same excluded seeds and unchanged thresholds. It is
defined in `experiment_01_convergence_pilot_400k.yaml` and must be frozen from
a clean commit before execution.

## 400k extension result

The extension was frozen at commit `a77bddb` and executed from clean commit
`3894d9e`. Both training and evaluation runs completed, but the declared pilot
decision is **NO-GO**:

| Field | Sparse | Shaped |
|---|---:|---:|
| Training episodes | 3,941 | 7,317 |
| Overall success | 99.9% | 99.9% |
| Final-10% success | 100% | 100% |
| First→last 20% native-return gain | +117.83% | +142.27% |
| Final-40% block means | −20.277, 35.424, 38.835, 16.287 | 109.572, 89.168, 131.000, 126.748 |
| `block_means_v2` | FAIL | FAIL |
| Frozen evaluation success | 10/10 | 10/10 |
| Frozen evaluation mean native return | 46.2 | 116.9 |
| Policy hash unchanged | PASS | PASS |

Sparse failed block range, trend and half-window equivalence. Shaped passed the
trend check but failed range and half-window equivalence. The raw package is
complete and truthfully lists both runs in `not_converged`; thresholds were not
changed after observing the result.

Canonical identities and PostgreSQL reconstruction:

| Arm | Checkpoint SHA-256 | Training exp/run | Evaluation run | Training/evaluation steps |
|---|---|---|---:|---:|
| Sparse | `d938de0979bc747bb4183eabd3aa6c3f3151bbbe821c0d0da19929552d5f2139` | 270 / 158 | 159 | 401,408 / 687 |
| Shaped | `ec4f08aaed64a4051b33841aff15099ee1e1e379697c917436cee81ecbfb7bc8` | 271 / 160 | 161 | 401,397 / 379 |

The workstation slept during sparse training, so its database wall time is not
a compute measurement. The uninterrupted shaped arm took 487.8 seconds; a
sequential 20-run 400k grid would project to about 163 minutes and also exceed
the two-hour limit. The final amended grid is therefore forbidden by both the
stability and runtime gates.

No larger-budget grid will be attempted. The only methodologically permitted
next step is a separately preregistered optimizer-stability pilot on another
excluded development seed, applying one common training schedule to both arms.
See `PHASE_06B_OPTIMIZER_STABILITY.md` once that protocol is committed.

## Full amended experiment

If the pilot passes, run `experiment_01_amendment` at the preregistered 200k
budget:

- matched training seeds 42–51;
- evaluation seeds 1001–1010;
- fixed topology seed 42;
- identical PPO, topology and CVE inputs;
- sparse versus shaped reward as the only experimental difference;
- frozen-policy evaluation and paired analysis in native reward units;
- original `experiment_01` retained alongside the amended package.

The result must be reported neutrally even if shaping harms performance, has no
effect, or fails to produce a CVE-severity shift. Statistical significance does
not override failed assumptions or failed stability evidence.

## Security and production boundary

This phase runs only the synthetic NASim simulator inside the existing
unprivileged, no-egress Podman service. It adds no live discovery, exploitation,
cloud integration, external API, browser runtime or weight release. Checkpoints
remain local and gitignored.

This is a research validation phase, not a production release. Backup/restore,
SLOs, monitoring, image signing/scanning and rollback remain unverified for a
production service and are not implied by a passing experiment.

## Acceptance evidence

- [ ] v2 stability tests pass and v1 historical tests remain green.
- [ ] pilot and final configurations validate as controlled experiments.
- [ ] pilot frozen-input manifest is committed before execution.
- [ ] 200k paired pilot passes the declared go/no-go rule.
- [ ] full grid is run only after a passing pilot.
- [ ] each final policy is evaluated frozen on the declared seeds.
- [ ] paired statistics consume only evaluation outcomes.
- [ ] PostgreSQL reconstructs every final run and evaluation.
- [ ] canonical amended package contains raw data, metadata, summaries, tables,
  figures and trajectories.
- [ ] full Podman, GUI, lint, compilation and Compose gates pass.
- [ ] coherent commits are published to `Security-RL`.

The unchecked pilot/full-grid items remain deliberate failures, not missing
reporting. Phase 6A ends with an evidence-backed no-go and does not produce a
new canonical reward comparison.
