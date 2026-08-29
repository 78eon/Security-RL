# Phase 3 — Mask-aware PPO and unseen on-prem topology evaluation

## Outcome

Phase 3 trains PPO across a preregistered distribution of hidden on-premises
topologies, freezes the checkpoint, evaluates it on topology seeds that were
never used for training, persists complete trajectories in PostgreSQL, and
renders a trace-derived attack graph in the native desktop application.

This is an offline simulator. No discovery action opens a socket, invokes a
scanner, launches an exploit, or communicates with a real network.

## Research question

Can one PPO policy, operating only on `Observation` and the valid-action mask
derived from `AgentKnowledge`, discover and complete attack paths on unseen
members of a bounded on-premises topology distribution?

This phase establishes the method and evidence pipeline. It does not claim that
one training seed or a short smoke run is a statistically final result.

## Preserved control

The fixed NASim sparse-versus-shaped experiment remains unchanged. Its code,
configuration, frozen hashes, checkpoints and evaluation results are not used
as training inputs for Phase 3. Phase 3 is a separate experiment with a typed
enterprise environment and separate output directories.

## Preregistered split

`OnPremGeneralisationSplit` is immutable in code:

| Purpose | Topology seeds | Policy updates allowed |
|---|---:|---|
| Training | 1–60 | Yes |
| Validation | 1001–1020 | No |
| Held-out test | 2001–2020 | No |

Each realised topology is hashed. The training manifest records the full
seed-to-hash mapping for all three groups. Evaluation fails closed if a requested
topology seed overlaps the training group.

## Policy boundary and zero-leakage controls

The decision flow is:

```text
TrueTopology --simulated transition--> AgentKnowledge --> Observation
                                           |
                                           +--> valid action mask

Observation + mask --> frozen/training MaskablePPO --> opaque action index
```

The policy never receives node identifiers, topology size, edges, crown-jewel
locations, vulnerability identities that have not been discovered, topology
seed, or topology hash. Catalogue actions address fixed discovery-order slots.
`action_masks()` calls the knowledge-only validity function. A regression test
replaces `TrueTopology` with an object that raises on every attribute access and
proves mask computation still succeeds.

The official `sb3-contrib==2.9.0` `MaskablePPO` implementation is pinned to the
same release as `stable-baselines3`. The mask is supplied during training and
explicitly supplied to every evaluation `predict` call.

## Source map

| Source | Responsibility |
|---|---|
| `configs/experiments/onprem_generalisation.yaml` | Frozen PPO parameters, budget and evaluation seeds |
| `enterprise/environment.py` | Knowledge-only `action_masks()` interface and evidence-rich events |
| `enterprise/onprem.py` | Hidden topology curriculum with fixed observation/action spaces |
| `enterprise/generalisation.py` | Training, provenance gates, frozen evaluation, packaging and persistence |
| `scripts/train_onprem.py` | Podman training CLI |
| `scripts/evaluate_onprem.py` | Validation/test evaluation CLI |
| `storage/schema.sql` | Generic enterprise entity, prerequisite, outcome and coverage fields |
| `gui/backend.py` | PostgreSQL trace-to-path reconstruction adapter |
| `gui/views/research_console.py` | Native Qt graph and timed trace replay |
| `tests/test_onprem_generalisation.py` | Split, leakage, freezing and reconstruction invariants |

## Training provenance

`training_manifest.json` records:

- code commit and working-tree state;
- dependency configuration hash;
- experiment and topology configuration hashes;
- training seed and exact 50,176-step budget (196 complete rollouts);
- complete PPO hyperparameters;
- train, validation and test topology seeds;
- every realised topology hash;
- synthetic vulnerability manifest hash;
- checkpoint SHA-256 and in-memory policy SHA-256;
- the gated weight-release status.

Canonical training refuses a dirty or unknown working-tree state. The explicit
`--allow-dirty` switch exists only for development smoke tests and the resulting
state is recorded. Such a checkpoint is rejected by normal evaluation.

## Dedicated evaluation

Evaluation loads a frozen checkpoint and never calls `learn`. For every unseen
topology and evaluation episode seed it records:

- success and terminal reason;
- steps to goal and total reward;
- known and true node counts, reported only after decisions;
- discovery coverage;
- invalid masked selections and failed simulated actions;
- realised topology seed and hash;
- every action, target entity, prerequisite, outcome and state change.

Policy parameters are hashed before and after evaluation. Any change is a hard
failure. Raw episodes are written to CSV, full trajectories to JSONL, causal
paths to JSON, and the protocol plus hashes to evaluation metadata.

## Trace-derived path reconstruction

The causal path builder starts at the observed goal event and back-chains over
the prerequisites and outcomes recorded by successful state-changing actions.
It does not query `TrueTopology` for a shortest path. Therefore the displayed
route represents what the policy actually established, not an omniscient route
that the policy may never have observed.

PostgreSQL reconstructs:

```text
experiment -> evaluation run -> episode -> ordered steps
                                      -> target/prerequisites/outcomes
                                      -> goal and coverage outcome
```

The PySide6 Paths workspace reads these rows through `BackendPort`, shows the
stored paths, and replays the selected causal route as a native `QPainter`
graph. There is no browser, HTML, JavaScript, Electron or QWebEngine.

## Podman commands

```bash
make build
make db-up
make onprem-train
make onprem-eval
make gui-build
make gui
```

Direct development smoke runs must be labelled as such:

```bash
podman compose run --rm app python scripts/train_onprem.py \
  --timesteps 1024 --output runs/onprem-smoke --allow-dirty
podman compose run --rm app python scripts/evaluate_onprem.py \
  --run runs/onprem-smoke --split validation --limit 1 \
  --output results/onprem-smoke --allow-unverifiable
```

## Acceptance gates

- [x] Mask-aware PPO executes in the Podman-only training image.
- [x] Training samples only seeds 1–60.
- [x] Evaluation rejects training-seed overlap.
- [x] Action masking is proven independent of `TrueTopology` reads.
- [x] Frozen evaluation verifies unchanged policy parameters.
- [x] Evaluation emits episode, raw trajectory, causal-path and metadata files.
- [x] PostgreSQL reconstructs generic enterprise entity trajectories.
- [x] Native desktop graph replays backend-derived path evidence.
- [ ] Canonical clean 50,176-step checkpoint completed.
- [ ] Validation and held-out test packages completed from that checkpoint.
- [ ] Full release gate and exact evidence recorded below.

## Development evidence (not a research result)

The initial 1,024-step smoke checkpoint completed three episodes on unseen
topology seed 1001 with a 100% success rate, 139 mean steps to goal, full
coverage and zero invalid masked selections. A separate 256-step database smoke
run reconstructed 3 episodes and 399 ordered step rows; its causal route reduced
133 raw events to 13 prerequisite-linked events ending at `asset_crown`.

Both smoke checkpoints used explicit development overrides and are excluded
from canonical research evidence.

## Remaining limitation after this phase

The distribution is on-premises only and uses one topology generator. The typed
graph and knowledge model are infrastructure-neutral, but legacy, cloud and
hybrid profiles remain a later phase. Generalisation claims must remain limited
to the preregistered on-premises distribution until those profiles receive
their own clean training and unseen-seed evaluation.
