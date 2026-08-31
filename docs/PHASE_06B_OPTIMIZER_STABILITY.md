# Phase 6B — Optimizer-stability pilot

## Status

Preregistered before training. No seed-33 policy or evaluation result was
inspected when this protocol was written.

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

## Acceptance record

- [ ] schedule implementation and historical-hash regression tests pass;
- [ ] protocol and inputs are frozen from a clean committed tree;
- [ ] both arms meet the preregistered learning and stability gates;
- [ ] frozen evaluations and PostgreSQL reconstruction pass;
- [ ] runtime gate passes;
- [ ] outcome and decision are recorded without retrospective changes;
- [ ] full tests, lint, compilation, Podman and Compose validation pass;
- [ ] coherent commits are published to `Security-RL`.
