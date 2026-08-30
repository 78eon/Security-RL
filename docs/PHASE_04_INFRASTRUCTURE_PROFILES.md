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

## Current status

Implementation and canonical experiment evidence are recorded in the final
section of this dossier after the phase gates complete.

