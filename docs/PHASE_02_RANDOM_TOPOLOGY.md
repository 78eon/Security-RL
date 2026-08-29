# Phase 2 — Partial-Observable Random Topology Discovery

## Status

Release candidate. This document is completed with final gate evidence before the phase commit
is pushed.

## Objective

Given a seeded on-prem enterprise topology hidden from the policy, progressively discover
hosts, services, vulnerabilities and connectivity; interleave discovery and simulated attack
actions; reach the crown jewel using only agent-visible knowledge; and reconstruct the
successful trajectory.

This phase implements the proposal's topology-generation stage for 10–60 hosts and 1–5
subnets. The original fixed NASim topology and its sparse-versus-shaped experiment remain the
control and are not modified.

## Architecture decision

The knowledge model is enterprise-wide and infrastructure-agnostic:

```text
Infrastructure profile configuration
             |
             v
       typed TrueTopology             simulator-only trust boundary
             |
     legitimate action result
             v
        AgentKnowledge                policy-visible facts only
             |
             v
         Observation                  fixed-size numeric policy input
```

`TrueTopology`, `AgentKnowledge`, `Observation`, node types and edge types are shared core
types. On-prem is the first constrained generator profile. Legacy, cloud and hybrid profiles
must use the same types and interfaces; they must not create parallel knowledge models.

## Distribution v1

The committed `configs/onprem_topology.yaml` defines:

- 10–60 hosts;
- 1–5 network segments;
- 0–3 required pivots;
- 1–3 application/API hops;
- 1–10 decoy services;
- a fixed capacity of 96 typed entities, eight vulnerabilities and 200 steps;
- disjoint training seeds 1–60, validation seeds 1001–1020 and test seeds 2001–2020.

Every topology contains a seeded variant of a feasible route through entry discovery, service
assessment, synthetic vulnerability exploitation, credential acquisition, authentication,
optional multi-segment pivoting, database access and crown-jewel access. Counts, placement,
service profile, application depth, pivot depth and distractors vary by seed.

## Security and research invariants

1. Generation and transitions are in-memory and simulation-only; no scanner, socket, exploit
   process or live target is used.
2. The same seed and configuration produce byte-equivalent canonical topology data and the
   same SHA-256 topology hash.
3. Configuration fails closed when capacity cannot contain its worst-case topology.
4. The on-prem profile rejects cloud, hybrid and legacy node types.
5. Action and observation spaces are identical across topology seeds.
6. Hidden topology size, identifiers and structure do not alter the initial observation or
   action mask.
7. Unknown entity names cannot be addressed through the action catalogue.
8. Action validity is derived from `AgentKnowledge`, not hidden ground truth.
9. Newly entered segments become discoverable, allowing discovery and attack actions to be
   interleaved.
10. The feasibility oracle consumes only the public action catalogue, action mask and
    `AgentKnowledge`. It validates the environment and is explicitly not PPO evidence.

## Implementation map

- `src/rlredteam/enterprise/onprem.py`: configuration, generator, hashes, seed splits,
  curriculum environment and knowledge-only feasibility oracle.
- `src/rlredteam/enterprise/environment.py`: segment revelation after authentication/pivot.
- `configs/onprem_topology.yaml`: versioned distribution parameters and capacities.
- `scripts/onprem_demo.py`: reconstructable JSON demonstration for one hidden seed.
- `tests/test_onprem_topology.py`: distribution, secrecy, bounds, safety, split and path tests.
- `Makefile`: `make onprem-demo` operator command.

## Acceptance evidence

- All 100 preregistered train/validation/test topologies are generated within bounds and use
  only on-prem node types.
- All 100 have a successful, reconstructable knowledge-derived path to `asset_crown`.
- A multi-segment regression test proves `PIVOT -> DISCOVER_NETWORK` interleaving.
- Different hidden topologies with the same visible boundary produce identical initial policy
  inputs.
- Socket constructors are denied during generation and an entire episode.
- `make onprem-demo` emits a topology/config hash and successful JSON trajectory while clearly
  labelling the policy as a feasibility oracle, not PPO.

Final repository-wide tests, lint, compilation, GUI regression and Podman validation are
recorded in the release commit handoff.

## Operation

```bash
make build
make onprem-demo
podman compose run --rm app pytest -q -p no:cacheprovider tests/test_onprem_topology.py
```

Use `--print-topology` only for analyst/debug inspection after an episode. It crosses the
policy trust boundary and must never be passed to a training or evaluation policy.

## Limitations and next phase

Phase 2 proves distribution validity and simulator feasibility; it does not claim PPO
generalisation. Phase 3 trains PPO across the training topology distribution, freezes the
checkpoint, evaluates only on disjoint unseen seeds, records provenance and reconstructs the
policy's actual trajectories. Action masking is required by the proposal and is a Phase 3
training boundary requirement.
