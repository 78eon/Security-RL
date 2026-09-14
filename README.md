# RLRedTeam

Simulation-only reinforcement learning agent for attack-path discovery on randomised
enterprise topologies, with a CVE/CVSS-severity-weighted reward.

The enterprise simulator provides typed, partially observable on-premises, legacy, cloud and
hybrid profiles covering
network segments, hosts, services, applications, APIs, identities, databases, security
controls and data assets. `TrueTopology`, `AgentKnowledge` and `Observation` are explicit
separate layers, and opaque discovery-order action slots do not reveal hidden entity names or
counts. Discovery actions are offline graph-state transitions, never live scanning. The fixed
NASim topology remains the experiment control.

An unprivileged, scope-gated backend is also provided for evidence collection in an explicitly
authorized isolated hybrid lab. It performs conservative discovery and imports Greenbone
reports; it does not autonomously exploit systems. Local operating notes are intentionally
kept outside the published repository.


## Quick start

```bash
cp .env.example .env          # add your NVD_API_KEY (never committed)
make build                    # build the training image
make db-up                    # start PostgreSQL on :5433
make test                     # full suite, including Postgres and NASim integration
make rollout                  # deterministic random-policy rollout
make enterprise-demo          # typed discovery-to-crown-jewel demonstration
make onprem-demo              # hidden on-prem topology feasibility trajectory
make onprem-train             # mask-aware PPO over training topology seeds 1-60
make onprem-eval              # frozen evaluation on held-out seeds + PostgreSQL
make onprem-verify            # verify checkpoint, evidence and PostgreSQL reconstruction
make infrastructure-train     # train across legacy/cloud/hybrid profiles
make infrastructure-eval      # frozen held-out evaluation for every profile
make lab-build                # build isolated-range evidence collector
make train                    # 50k-step PPO pilot on the shaped reward

make gui-build                # build the desktop app image (once)
make gui                      # launch the analyst desktop app
```

All installation, testing and application launch commands use Podman. No Python or Qt
packages need to be installed on the host.

The desktop console is backend-driven. It reads stored execution steps and experiments
from PostgreSQL, run summaries and configuration snapshots from `runs/`, and the frozen
CVE catalogue from SQLite. If PostgreSQL is unavailable it clearly switches to artefact
mode. Completed runs are read-only: campaign controls and configuration writes are not
shown until a real scheduler or configuration service exists.
The Paths workspace reconstructs prerequisite-linked enterprise attack paths from
PostgreSQL and replays them as a native Qt graph; it does not calculate an omniscient
route from hidden topology.
The Simulation workspace lets a demonstrator choose a configured enterprise profile and
topology seed, runs the actual offline Python backend on a Qt worker, and displays the
generated entities and trace-derived path. It does not enumerate nearby or real networks.

Random-topology and cross-profile protocols are enforced by their committed experiment
configurations, manifests and tests; detailed research notes remain local.

## Fixed-budget research experiment

The prospective 200,000-step sparse-versus-shaped protocol is declared in
`configs/experiments/experiment_01_fixed_budget.yaml`. After its frozen manifest is committed,
validate or execute it entirely inside Podman:

```bash
podman compose run --rm app python scripts/run_experiment.py \
  --config configs/experiments/experiment_01_fixed_budget.yaml --dry-run
podman compose run --rm app python scripts/run_experiment.py \
  --config configs/experiments/experiment_01_fixed_budget.yaml
```

Generated checkpoints and results remain local and gitignored.

## Explainable attack-path reports

Phase 14 reconstructs deterministic report facts from recorded trajectory events. The
pipeline keeps policy actions, resolved simulator actions, recorded events, versioned MITRE
mappings and report facts as separate layers. ATT&CK and ATLAS semantics come exclusively
from `configs/catalogues/mitre.yaml`; ordinary RL control is never labelled as ATLAS.

Generate the example simulation report from the existing `experiment_01` trajectory package,
persist its derived facts to PostgreSQL, and verify exact reconstruction:

```bash
make phase14-report
make phase14-verify
```

The desktop console's **Path Report** workspace reads those stored derived facts. It shows a
MITRE timeline, observed AgentKnowledge targets, selected-trajectory facts, provenance and an
observed path-critical CVE ranking. Normal reports never read hidden `TrueTopology`.

Two report modes are enforced:

- `simulation_report` may describe synthetic PPO/simulator actions, rewards and knowledge;
- `evidence_report` may describe imported scanner observations, but rejects PPO, checkpoint,
  RL-action and simulator-reward claims.

Observed path criticality is deliberately not presented as a patching counterfactual. For
example, “CVE-X was path-critical in 61% of observed successful trajectories” describes the
recorded sample only; it does not claim patching would block 61% of all attacks. Generated
reports remain under ignored `results/` and are not published.

## Mitigation counterfactual evaluation

Phase 15 promotes an optional Phase 14 observed path-critical CVE into a separate causal
experiment. A versioned evaluation-only wrapper is applied after the original environment
is reconstructed. It leaves the action space and deterministic CVE assignment intact, but
prevents actions assigned to the selected CVE from changing simulator state.

The same frozen PPO checkpoint is evaluated in original and mitigated conditions with an
identical ordered episode-seed set. Policy hashes are checked before, between and after the
conditions; no training or gradient-update interface is used. Run and verify the local,
ignored example with:

```bash
make phase15-report
make phase15-verify
```

The desktop **Mitigation** workspace reads the persisted paired outcomes from PostgreSQL. It
shows original versus mitigated success, steps, reward, crown-jewel reach and MITRE/path
summaries. The terminology is intentionally strict: **observed path criticality** is derived
from recorded paths, while **mitigation effect** is measured by frozen-policy reruns. The
effect is specific to the checkpoint, topology, intervention and evaluation seeds; it does
not claim that disabling a CVE blocks every possible attack.

## CyberBattleSim adapter

CyberBattleSim is an additive second simulator backend; NASim remains the fixed experimental
control. Both adapters implement `SimulatorAdapter`, which keeps native actions, Security-RL
semantic events, catalogue-derived MITRE mappings and reports as separate layers. The initial
scope is intentionally small: a chain of four nodes for training and a held-out six-node chain
for deterministic evaluation.

CyberBattleSim's native vulnerability labels are recorded as
`simulator_vulnerability_id`. They are never promoted to a CVE unless the simulator provides a
real CVE identifier. PPO receives only an `AgentKnowledge` encoding and the simulator-visible
action mask; hidden topology is not part of its observation.

Run the complete integration through its isolated, network-disabled Podman runtime:

```bash
make cyberbattle-smoke    # successful native trajectory + Phase 14-compatible report
make cyberbattle-train    # independent standard PPO checkpoint
make cyberbattle-eval     # frozen deterministic evaluation on held-out seeds/scenario
make cyberbattle-verify   # adapter tests and evidence verifier
```

The CyberBattle image inherits Security-RL's numerical stack and applies a narrowly scoped
compatibility shim for an upstream NumPy 2 construction call. It does not alter the main NASim
image. Generated checkpoints and reports remain under ignored `runs/cyberbattle/` and
`results/cyberbattle/`.

## CybORG adapter

CybORG is an additive third simulator backend through the same
`SimulatorAdapter` contract. The controlled first integration uses the official
four-host `Scenario1.yaml`, an external Red agent and single-agent MaskablePPO.
Only Red's native observation and visible action prerequisites update
`AgentKnowledge`; native simulator ground truth is not read by the adapter.

The CybORG compatibility stack is confined to `Dockerfile.cyborg`. It pins
CybORG 3.1 by source revision and its required legacy Gym/NumPy combination,
without changing the NASim or CyberBattle images. Runtime targets disable the
network, drop Linux capabilities and mount source/configuration read-only:

```bash
make cyborg-smoke    # native Scenario1 trajectory + Phase 14-compatible report
make cyborg-train    # single-agent masked PPO with fixed training seeds
make cyborg-eval     # frozen deterministic evaluation on disjoint seeds
make cyborg-verify   # contract/security tests and evidence verifier
```

CybORG exploit names remain `simulator_vulnerability_id` values and are not
invented as CVEs. Generated evidence remains under ignored `runs/cyborg/` and
`results/cyborg/`.

## Component-level reward ablation

Phase 18 keeps the fixed NASim topology, CVE catalogue, PPO settings and
matched seed sets constant while independently enabling CVSS weighting, MITRE
tactic shaping, informative-success shaping, the failure penalty and the
objective reward. The historical sparse/shaped implementation and hashes are
unchanged; only configs with an explicit `components` block use the new model.

```bash
make reward-components-freeze   # preregister immutable inputs
make reward-components-dry-run  # inspect the seven-arm, 21-run grid
make reward-components-run      # train and dedicated-evaluate frozen policies
make reward-components-verify   # verify pairing, hashes and policy immutability
```

## Causal Attack Path Explorer

Phase 19 derives semantic node and edge facts only from Phase 14 trajectory
facts and recorded AgentKnowledge deltas. Every edge carries evidence IDs,
episode/step provenance, action semantics, access transition, MITRE catalogue
mappings and all recorded exposure/access details. Missing facts remain
`unknown`; hidden topology is rejected. PostgreSQL stores normalized node,
edge and knowledge-evidence rows and remains authoritative.

```bash
make phase19-report   # derive and persist the graph
make phase19-verify   # deterministic, PostgreSQL and headless PySide6 checks
make gui              # inspect Attack Path and Knowledge Flow tabs
```

## Reproducing the environment

```bash
make rollout                  # seed 42 by default
podman run --rm -v "$PWD:/app:z" -w /app localhost/sourcecode_app:latest \
    python scripts/rollout_random.py --seed 42 --reward-config configs/shaped.yaml
```

Prints the topology config hash, CVE manifest digest, observation and action spaces, the
CVE assignment, and one episode per seed. The same seed always reproduces the same
topology and the same rollout.

## How reproducibility works

A topology is **generated**, not hand-written, but it is fully recoverable from two
values that are logged with every experiment:

- `topology_config_hash` — digest of `configs/topology.yaml` (the generation rules)
- `topology_seed` — the specific instance

CVE assignment is likewise a deterministic function of the topology seed, so no extra
artefact is needed to reconstruct it. The CVE data itself is frozen: a committed SQLite
file with a SHA-256 manifest over a canonical row dump. Training never touches the
network.

## Layout

```
src/rlredteam/
  events.py          AttackEvent -- the plain-data type the reward scores. Zero deps.
  cvss.py            CVSS v3.1 bands and the severity -> weight map.
  catalogue.py       Module 0: frozen SQLite CVE catalogue, opened read-only.
  manifest.py        SHA-256 over a canonical row dump (not the .sqlite bytes).
  assign.py          Seeded stratified CVE assignment for generated topologies.
  reward.py          The reward engine: shaped / sparse / native modes.
  topology.py        Seeded nasim.generate wrapper + config hashing.
  nasim_adapter.py   NASim reward bridge and common-contract facade.
  simulator_adapter.py  Common simulator semantic contract and transition schema.
  cyberbattle_adapter.py  AgentKnowledge-only CyberBattleSim chain adapter.
  cyberbattle_study.py  PPO train/evaluate/report/provenance orchestration.
  cyborg_adapter.py  Observable-only CybORG Scenario1 semantic adapter.
  cyborg_study.py  MaskablePPO train/evaluate/report/provenance orchestration.
  enterprise/        Hidden truth, agent knowledge, observations and on-prem simulator.
  train.py           PPO entry point: seeding, episode collection, logging.
  storage/           Module 4: PostgreSQL schema and batched episode logger.
  attack_path_report.py  Phase 14 deterministic facts, phases and observed criticality.
  phase14_completion.py Fail-closed source-to-report reconstruction verifier.
  mitigation.py       Phase 15 overlay, paired evaluation, statistics and provenance.
  phase15_completion.py Fail-closed counterfactual report verifier.

gui/                 Native PySide6 research console — separate Podman image
  backend.py         typed snapshot adapter for database and persisted artefacts
  data/              PostgreSQL repository + runs/ reader; no Qt imports
  workers/           QThreadPool queries and isolated Qt training facade
  views/             eight-workspace research console and stable main window
  theme.py/.qss      desktop design tokens and supplied dark visual system

configs/             topology.yaml, shaped.yaml, sparse.yaml
data/                cve_catalogue.sqlite, its manifest, raw NVD provenance JSON
scripts/             training, frozen-checkpoint evaluation and experiment runners
tools/               fetch_nvd.py -- one-shot, online, never imported by training
runs/                checkpoints and artefacts. GITIGNORED (see below).
```

The reward core imports no nasim, gymnasium or torch, so it is testable without an
environment. Simulator-specific code remains isolated in the NASim and CyberBattle adapters.

## Responsible research

- **Simulation only.** Synthetic, generated topologies. No live network, no scanning, no
  human subjects.
- **Trained weights are gated by default.** `runs/` is gitignored; policy checkpoints are
  not committed and not released without an explicit supervised release decision
  (charter §10).
- **The CVE catalogue is a severity lookup table, not an exploit collection.** It holds
  public NVD metadata — score, vector, CWE, URL — for historical, largely patched
  vulnerabilities. It contains no exploit code.
- **The NVD API key lives in `.env`**, which is gitignored, and is read only by
  `tools/fetch_nvd.py`. Training is fully offline.

## Known limitations

Stated plainly rather than discovered later:

1. The `exfil` tactic bonus (+1.5) is **unreachable** — NASim has no exfiltration action.
   It is defined so its zero count is visible, pending a supervisor decision.
2. Tactic bonuses pay only on the first *informative* success per target. Without this,
   scan-spam earns up to +1000 against a +100 crown jewel and farming becomes optimal.
   This bounds shaping but makes the reward history-dependent within an episode.
3. CVE-to-exploit pairing is synthetic. The severities are real and NVD-sourced; the
   binding to a specific *generated* service is a seeded draw, not genuine service
   correspondence.
4. n=10 with Bonferroni is underpowered for anything but large effects.
