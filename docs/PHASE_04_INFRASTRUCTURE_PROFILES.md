# Phase 4 — Infrastructure-Agnostic Enterprise Profiles

## Objective

Represent legacy, cloud and hybrid estates as configuration variants of the
same partially observable typed-graph simulator, then train one mask-aware PPO
policy over that distribution and evaluate frozen weights on unseen topology
seeds for every profile.

The simulator remains entirely in memory. It performs no host discovery,
packet transmission, authentication, vulnerability probing or exploitation
against a real system.

## Relationship to earlier phases

- Phase 1 remains the frozen fixed-topology research control.
- Phase 2's on-premises generator remains frozen; selecting `on_premises`
  delegates to that generator and preserves its topology hashes.
- Phase 3's `AgentKnowledge`, fixed observation/action spaces, knowledge-only
  action masks, frozen evaluation and causal trajectory reconstruction remain
  the behavioral and evidence boundary.
- Phase 4 adds deployment profiles without adding a second simulator.

## Architecture

`configs/enterprise_profiles.yaml` defines bounded node-type families and
trust-boundary labels. `generate_profile_topology()` feeds all profiles into
`EnterpriseCyberEnv`, so action semantics, reward behavior, discovery state,
partial observability and trajectory events are shared.

| Profile | Network | Compute | Identity | Data |
|---|---|---|---|---|
| Legacy | Segments | Legacy hosts | Identities | Databases |
| Cloud | Cloud networks | Cloud workloads | IAM roles | Storage/resources |
| Hybrid | Both | Standard, legacy and cloud | Identity and IAM | Mixed stores |

Every hybrid sample contains both local and cloud network/compute/identity
types; random sampling cannot collapse a hybrid experiment into a single
infrastructure family.

## Bounded distribution

- Network segments: 1–5 (hybrid minimum is 2)
- Hosts/workloads: 10–60
- Causal pivots: 0–3
- Decoy services: 1–10
- Fixed policy capacity: 96 nodes, 8 vulnerabilities, 200 actions per episode
- Synthetic vulnerabilities only; their full manifest is hashed

The seed controls structure, sizes, types and synthetic entry weakness. A
feasible causal route is constructed before decoys are added.

## Information boundary

The curriculum selects a hidden `(profile, topology_seed)` pair while retaining
fixed Gymnasium spaces. PPO receives only:

1. the `Observation` generated from `AgentKnowledge`; and
2. the boolean action mask generated from that same knowledge.

Ground-truth topology is used by the environment to resolve a selected action
and is consulted after decisions only for provenance/evaluation metrics.

## Experiment protocol

- Training topology seeds: 1–60 for each of legacy, cloud and hybrid
- Validation topology seeds: 1001–1020 for every profile
- Test topology seeds: 2001–2020 for every profile
- Training seed: 43
- PPO budget: 50,176 exact steps
- Evaluation: one deterministic frozen-policy episode per profile/topology
- Primary unit: one independently generated profile/topology pair
- Required metrics: success, steps-to-goal, reward, coverage, failed actions,
  invalid mask selections, and success by profile

The validation and test phases reject training-seed overlap. Checkpoint and
policy hashes are checked before evaluation; policy weights are hashed again
after evaluation to prove no gradient updates occurred.

## Evidence gates

- Profile/config determinism and bounds
- Profile type separation and mandatory hybrid composition
- Fixed action and observation spaces
- Feasible discovered path for sampled profile/topology pairs
- No real network use
- Knowledge-only masks with truth-poison regression test
- Clean-worktree canonical training
- Held-out validation and test packages
- PostgreSQL reconstruction of episodes, actions and causal paths
- Full tests, Ruff, Python compilation and Podman Compose validation

## Canonical implementation

- Code commit: `36bf3df05f733e37afd2fcdbab867161c6ae59bf`
- Experiment ID: `infrastructure-generalisation-v1`
- Experiment config SHA-256:
  `d612c6806f10014e0567e53de7e4dabf6bff32a055e4e1ea412463dc29e15637`
- Profile config SHA-256:
  `5f23e4bc8b01bebdce027e5f7067ce62585b75c0050174bfaa9f7d9128f4b675`
- Dependency lock fingerprint: `74a17ae310a31d60`
- Synthetic vulnerability manifest SHA-256:
  `28d43d0d13cc01bb379ce63ed7500daa0d05cc4cde1066d71d614d6ff2079f11`
- Checkpoint SHA-256:
  `59b581910e5a9d031a9d2ef795b17f293abd0efa33190f7c65c7a0f63c226192`
- Frozen policy SHA-256:
  `ce09d4428b2472778ccb03473f04b6873cab7a0a04361983282c66b2242e7945`
- Requested/actual training steps: `50,176 / 50,176`
- Canonical training recorded `git_dirty: false`.

The model checkpoint is retained locally under
`runs/infrastructure-generalisation/` and remains excluded from Git by the
dual-use weights gate.

## Frozen evaluation results

| Split | Profile | Episodes | Success | Mean steps | Mean reward | Mean coverage | Invalid/failed |
|---|---:|---:|---:|---:|---:|---:|---:|
| Validation | Legacy | 20 | 100% | 75.55 | 184.487 | 0.819779 | 0 / 0 |
| Validation | Cloud | 20 | 100% | 74.50 | 183.437 | 0.819779 | 0 / 0 |
| Validation | Hybrid | 20 | 100% | 81.80 | 190.739 | 0.852215 | 0 / 0 |
| Test | Legacy | 20 | 100% | 73.00 | 181.140 | 0.790583 | 0 / 0 |
| Test | Cloud | 20 | 100% | 72.55 | 180.690 | 0.784869 | 0 / 0 |
| Test | Hybrid | 20 | 100% | 73.35 | 181.484 | 0.790885 | 0 / 0 |

Aggregate validation mean steps/reward/coverage were
`77.2833 / 186.2212 / 0.830591`. Aggregate held-out test values were
`72.9667 / 181.1047 / 0.788779`. Both splits had 60/60 successes and zero
invalid mask selections. The policy hash was identical before and after both
evaluation phases, proving the weights remained frozen.

The gated evidence packages are in
`results/infrastructure-generalisation/{validation,test}/`. Each contains the
raw episode CSV, action-level JSONL, causal attack paths, metadata and summary.
All 120 successful episodes produced a causal path; held-out paths contained
10–19 state-changing events and were reconstructed only from recorded
prerequisites/outcomes.

## PostgreSQL reconstruction

| Split | Experiment | Run | Status | Episodes | Steps | Profile allocation |
|---|---:|---:|---|---:|---:|---|
| Validation | 266 | 152 | complete | 60 | 4,637 | 20 legacy / 20 cloud / 20 hybrid |
| Test | 267 | 153 | complete | 60 | 4,378 | 20 legacy / 20 cloud / 20 hybrid |

Every stored episode reached the goal. Each database episode retains its
deployment profile, topology seed/hash, evaluation seed, coverage and outcome;
each step retains the action target, state-change flag, prerequisites and
outcomes needed for causal reconstruction.

## Validation record

- Focused Phase 4 suites: 53 passed (one unrelated slow test deselected)
- Full Podman suite: 399 passed, 13 skipped, 2 pre-existing SciPy precision
  warnings; 170.17 seconds
- Native PySide6 GUI suite: 38 passed
- Ruff: passed across `src`, `gui`, `tests`, `tools` and `scripts`
- Python compilation: passed with bytecode redirected to `/tmp`
- Podman Compose configuration: passed
- Real 256-step train/evaluate smoke run: three profile episodes succeeded,
  zero invalid mask selections

## Phase verdict

| Requirement | Verdict |
|---|---|
| One typed graph and action model | PASS |
| Legacy configuration | PASS |
| Cloud configuration | PASS |
| Hybrid configuration with enforced boundary | PASS |
| Frozen on-prem baseline preserved | PASS |
| Seeded structural distribution | PASS |
| `AgentKnowledge` partial observability | PASS |
| Knowledge-only action masks / zero truth access | PASS |
| Cross-profile MaskablePPO training | PASS |
| Unseen-seed frozen evaluation | PASS |
| Causal attack-path reconstruction | PASS |
| PostgreSQL reconstruction | PASS |
| Simulation-only safety | PASS |

Phase 4 is complete. It establishes enterprise-wide infrastructure profiles as
configurations of one simulator; it does not claim that one training seed and
one synthetic distribution demonstrate universal real-world generalisation.
