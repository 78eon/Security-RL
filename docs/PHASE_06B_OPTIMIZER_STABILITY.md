# Phase 6B — Optimizer-stability pilot

## Status

Complete — **NO-GO**. The protocol was preregistered before training and the
decision below uses the unchanged prospective gates. No final seed was run.

## Reason for this phase

The Phase 6A constant-learning-rate pilots learned effective policies but did
not pass the unchanged aggregate stability rule at either 200,000 or 400,000
timesteps. The 400,000-timestep design also exceeded the declared two-hour
projection for a 20-run grid. Running final seeds after those failures would
not produce a defensible confirmatory comparison.

A constant PPO learning rate can continue moving policy parameters late in a
run. This phase tests one prospective remedy allowed by the proposal's
development-stage hyperparameter process: linearly decay the common initial
rate of `3e-4` to zero over training. It is applied identically to sparse and
shaped PPO and is represented in the frozen PPO hash.

## Frozen pilot protocol

| Field | Preregistered value |
|---|---|
| Experiment | `experiment_01_optimizer_pilot` |
| Training seed | 33, excluded from final seeds 42–51 |
| Evaluation seeds | 911–920, excluded from final 1001–1010 |
| Topology seed | 42 |
| Budget | 200,000 timesteps per arm |
| Learning rate | linear `3e-4` to zero |
| Arms | sparse and shaped, unchanged |
| Stability method | unchanged `block_means_v2` |
| Database evidence | PostgreSQL episode and step logging |

The topology, CVE snapshot, PPO settings other than the schedule, reward
definitions, native evaluation scale, evaluation action selection and all
stability thresholds remain unchanged. Historical constant-rate configuration
digests remain bit-for-bit compatible.

## Go/no-go rule

Proceed to a separately frozen final 200,000-timestep matched grid only if both
arms satisfy every condition:

1. training and PostgreSQL logging complete without NaN, Inf or provenance
   failure;
2. final-20% mean native return improves by at least 10%, normalised by
   `max(abs(first-20% mean), 100)`;
3. the unchanged `block_means_v2` rule passes;
4. frozen evaluation completes on all ten declared episode seeds, performs no
   gradient updates and preserves the policy hash;
5. the results package is internally complete; and
6. measured uninterrupted runtime projects the 20-run final grid to at most
   two hours on this host.

If either arm fails, Phase 6B is a no-go. Thresholds will not be changed, final
seeds will not be inspected, and no further optimizer candidates will be tried
in this phase. The correct report will be that convergence was not established
under the capped protocol.

## Security and scope boundary

This remains a synthetic, offline NASim experiment in rootless Podman. It adds
no live network discovery, exploit execution, cloud credentials, browser
runtime, external API or model publication. The change is research provenance
and optimizer control only; it does not make the application production-ready.

## Pilot result

The pilot executed from clean commit `90996b4`, completed both arms and all
frozen evaluations, and generated a structurally complete local package at
`results/experiment_01_optimizer_pilot`. Total uninterrupted wall time was 487
seconds. Scaling the observed two runs to the 20-run final grid projects 4,870
seconds (81.2 minutes), so the runtime gate passed.

| Field | Sparse | Shaped |
|---|---:|---:|
| Training episodes | 1,597 | 2,424 |
| First→last 20% native-return gain | +95.60% | +123.97% |
| Final-40% block means | −16.063, −24.031, 5.176, −27.233 | 90.360, 81.335, 46.835, 48.083 |
| Block relative range | 0.3241 (PASS) | 0.4352 (FAIL) |
| Absolute normalised block slope | 0.004302 (PASS) | 0.161331 (FAIL) |
| Half-window change | 0.0902 (PASS) | 0.3839 (FAIL) |
| `block_means_v2` | PASS | **FAIL** |
| Frozen evaluation success | 10/10 | 10/10 |
| Evaluation mean native return | −56.8 | 5.4 |
| Evaluation mean steps to goal | 115.6 | 85.8 |

Both learning-gain checks passed, but shaped PPO failed every aggregate
stability threshold. The result is therefore a preregistered **NO-GO** even
though shaped performed better descriptively on this single development seed.
One pilot seed cannot support inferential reward-comparison claims, and these
development outcomes must not be presented as the final experiment.

Frozen-policy integrity and checkpoint identities:

| Arm | Checkpoint SHA-256 | Policy SHA-256 before/after evaluation |
|---|---|---|
| Sparse | `f487e4ce510d049ebed5dc3475d9d8ed56e765ff8a5fe85b908e7e9657c2b7b9` | `7c2ef7a5320855f6d8ede14489d0f066d3e248e0ab631e8b93630474173d1b76` |
| Shaped | `5df583286e84fec733e78147ac62d977f8ee2d339ea1928ca466087120084acc` | `a745f424851ac3d388fb41e1e2b85deed8c8f4f30cba4cda5dec77402ede1203` |

Both evaluation metadata records declare `gradient_updates: false`, and each
before/after policy hash is identical.

PostgreSQL reconstruction is complete:

| Arm/phase | Experiment/run | Episodes | Steps | Status |
|---|---:|---:|---:|---|
| Sparse training | 272 / 162 | 1,597 | 200,702 | complete |
| Sparse evaluation | 272 / 163 | 10 | 1,156 | complete |
| Shaped training | 273 / 164 | 2,424 | 200,695 | complete |
| Shaped evaluation | 273 / 165 | 10 | 858 | complete |

## Decision and next research boundary

The final matched grid is not authorised. Thresholds were not reinterpreted,
seeds 42–51 and 1001–1010 remain uninspected under this amendment, and no
additional optimizer candidate will be introduced. The defensible conclusion
is that PPO learning was demonstrated, but two-arm convergence was not
established under the capped fixed-baseline protocol.

This is a legitimate negative research finding rather than a software defect.
Further reward-effect claims require a new, supervisor-approved protocol with
new development and confirmatory seed sets; they cannot be manufactured by
continuing adaptive tuning in this phase.

## Acceptance record

- [x] schedule implementation and historical-hash regression tests pass;
- [x] protocol and inputs are frozen from a clean committed tree;
- [ ] both arms meet the preregistered learning and stability gates — shaped
  failed, so this item correctly remains unchecked;
- [x] frozen evaluations and PostgreSQL reconstruction pass;
- [x] runtime gate passes;
- [x] outcome and decision are recorded without retrospective changes;
- [x] full tests, lint, compilation, Podman and Compose validation pass;
- [ ] coherent commits are published to `Security-RL`.
