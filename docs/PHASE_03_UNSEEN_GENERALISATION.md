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
- [x] Canonical clean 50,176-step checkpoint completed.
- [x] Validation and held-out test packages completed from that checkpoint.
- [x] Full release gate and exact evidence recorded below.

## Canonical evidence — 2026-08-30

Training identity:

| Field | Value |
|---|---|
| Code commit | `c369d4c51c92da31a24b85eadbc2257b1c9d7186` |
| Working tree at training | clean (`git_dirty: false`) |
| Actual training steps | 50,176 |
| Training topology seeds | 1–60 |
| Experiment config SHA-256 | `228d57a9bfd6eaf4bfbcfb39d973b75211592c3ae06eb09a32ab289900cb2540` |
| Topology config SHA-256 | `98fa63b38c6412cb84b7f334050f5b3164cd94a82c58cdd900ba6c1c88256411` |
| Synthetic vulnerability manifest SHA-256 | `586a16454e0daae4c8414f769376770f45432256ba2cad5a86692f0941529363` |
| Checkpoint SHA-256 | `a38a86c585e5532c5a0478db8f1f5745b9b8a8964b570e95b40ba497c4eb9626` |
| Policy parameter SHA-256 | `545676ca4c1419c3bff8bdf5464e8876a8164dfd3f6ed5d9ab446da86a4594c4` |

Frozen unseen-topology evaluation:

| Metric | Validation seeds 1001–1020 | Held-out test seeds 2001–2020 |
|---|---:|---:|
| Topologies / episodes | 20 / 20 | 20 / 20 |
| Goal success rate | 100% | 100% |
| Mean steps to goal | 85.7 | 92.4 |
| Step range | 34–158 | 36–155 |
| Mean total reward | 194.6315 | 200.5490 |
| Mean discovery coverage | 87.62% | 86.98% |
| Coverage range | 56.52–100% | 48.39–100% |
| Invalid masked selections | 0 | 0 |
| Causal path length range | 10–20 | 10–19 |
| Policy hash unchanged | PASS | PASS |
| Checkpoint hash matched manifest | PASS | PASS |

PostgreSQL reconstruction:

| Split | Experiment ID | Episodes | Ordered steps | All goals | Status |
|---|---:|---:|---:|---|---|
| Validation | 246 | 20 | 1,714 | yes | complete |
| Test | 247 | 20 | 1,848 | yes | complete |

The desktop backend loaded 12 recent held-out paths from experiment 247. Their
raw traces contained 37–155 steps, their causal replay graphs contained 11–19
nodes, and every graph ended at `asset_crown`.

Canonical local artifacts (policy weights remain gitignored and gated):

```text
runs/onprem-generalisation/
├── model.zip
└── training_manifest.json

results/onprem-generalisation/
├── validation/
│   ├── episodes.csv
│   ├── trajectories.jsonl
│   ├── attack_paths.json
│   ├── evaluation_metadata.json
│   └── summary.json
└── test/
    ├── episodes.csv
    ├── trajectories.jsonl
    ├── attack_paths.json
    ├── evaluation_metadata.json
    └── summary.json
```

Release gates after canonical evaluation:

- full Podman backend suite: 361 passed, 13 skipped, 2 existing SciPy warnings;
- native GUI suite: 38 passed;
- Ruff: passed;
- Python compilation with bytecode redirected to `/tmp`: passed;
- Podman Compose configuration: passed;
- checkpoint, policy-freeze, split, PostgreSQL and GUI reconstruction audits: passed.

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
