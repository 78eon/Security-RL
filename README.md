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

## Neo4j derived graph analysis

Phase 20 projects only PostgreSQL-authoritative Phase 19 node, edge and
knowledge-evidence facts into a rebuildable Neo4j service. The projection is
graph-ID scoped, every relationship retains evidence references, and no hidden
topology is accepted. Neo4j is optional and never replaces PostgreSQL or the
canonical JSON/CSV evidence.

Add a distinct strong `NEO4J_PASSWORD` to `.env`, then run:

```bash
make neo4j-export    # rebuild Neo4j from the latest PostgreSQL causal graph
make neo4j-analyze   # path, bottleneck, centrality and MITRE analyses
make neo4j-verify    # idempotence, source integrity and live Podman checks
```

## MLflow observability mirror

Phase 21 mirrors deterministic PostgreSQL run summaries, experiment parameters,
hashes and canonical artifact references into an optional MLflow service.
PostgreSQL and the canonical files remain authoritative; the training image has
no MLflow dependency and canonical checkpoints/results are never uploaded.

```bash
make mlflow-sync     # mirror recent PostgreSQL runs
make mlflow-verify   # verify idempotence and source immutability
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

## Convergence-first PPO protocol

This pipeline is **not evidence of convergence**. No 1M run is launched by tests.
The new, independent namespace is `results/convergence_v1/`; previous configurations,
checkpoints and results remain read-only. The unrelated PDF work is not part of this task.

1. Review `configs/convergence_criterion_v1.yaml` with the supervisor, **before**
   preregistration. Its numeric thresholds are explicit operational proposals, not
   quoted supervisor requirements. Commit the reviewed config and source; scientific
   commands refuse a dirty checkout. Keep the same commit and container image for the
   registered candidate, including its evaluation.
2. Run `make build` only if the application image is missing, then `make db-up`,
   `make reproducibility-check` and `make convergence-verify`.
3. Run `make convergence-1m-freeze` to preregister inputs, not to claim convergence.
   Then explicitly start `make convergence-1m-run` (ten long runs), followed by
   `make convergence-1m-analyze`. The first configuration which passes is frozen
   automatically; no best-of-many checkpoint selection is performed.
4. Only if all ten baseline runs complete and their aggregate assessment fails:
   `make convergence-normalized-freeze`, `make convergence-normalized-run`, then
   `make convergence-normalized-analyze`. This writes a direct baseline/normalised
   comparison using **original unnormalised shaped episode returns**.
5. Read the reward and diagnostic time series. If necessary, create an explicitly
   versioned child config (`stage: tuning`, `parent: <yaml path>`, `factor`, `rationale`).
   Allowed factors, in order: `learning_rate` (may include `linear_to_zero` schedule),
   then `n_steps` **or** `batch_size` separately, then `ent_coef`. Other changes,
   multi-factor changes, missing rationale and stage-skipping are rejected. Use the
   generic `convergence-1m-*` targets with `CONVERGENCE_CONFIG=<child.yaml>`.
   Diagnostic advice never launches a sweep or changes a config.
6. Once `first_stable.json` exists, further tuning is blocked. Set
   `CONVERGENCE_CONFIG` to that candidate and run `make convergence-sparse-run`, then
   `make convergence-evaluate`. Both commands are blocked until the shaped gate
   reconstructs successfully. Sparse is trained with the same hyperparameters and
   budget; only the reward condition differs. The experimental unit is ten matched
   **training seeds 42–51**, with ten matched evaluation episodes (seeds 42–51) per
   policy. These reused seeds are **not held-out generalisation evidence**.

Exact new experiment configs:

- `configs/experiments/experiment_01_convergence_1m.yaml`: unchanged baseline PPO,
  shaped reward, fixed topology seed 42, training/evaluation seeds 42–51, 1,000,000
  requested timesteps. PPO completes full 2,048-step rollouts: **1,001,472 actual
  timesteps**, recorded rather than silently rounded away. Learning rate 0.0003,
  batch size 64, epochs 10, gamma 0.99, GAE 0.95, clip 0.2, entropy coefficient 0.01.
- `configs/experiments/experiment_01_convergence_normalized_1m.yaml`: identical PPO,
  topology, reward and seeds; enables reward-only SB3 `VecNormalize`.

Advantage normalisation is explicitly enabled in both arms (the existing SB3
default). Reward normalisation uses a running variance of discounted returns
(`gamma=0.99`, `epsilon=1e-8`):
`r_train = clip(r_original / sqrt(return_variance + epsilon), -10, 10)`.
Observations are **never normalised or augmented**. Original shaped/native rewards
and semantic step records remain in the canonical episode collector/PostgreSQL.
Each normalised checkpoint saves and hashes `vecnormalize.pkl`. Frozen evaluation
loads and validates that state, disables updates, and reports untransformed rewards;
with `norm_obs=false`, policy input is exactly the original partial observation.
The evaluation wrapper forces deterministic action selection, uses no gradients,
and checks policy hashes and PPO update counts before/after. PostgreSQL and canonical
CSV/JSON remain authoritative; no new observability backend is introduced.
Because the historical episode uniqueness key does not include the run ID,
convergence evaluation uses a separate PostgreSQL experiment linked to its training
experiment/run and checkpoint hash in notes and canonical metadata. This supports
reused seeds without altering the old schema or colliding with training episodes.

Assessment schema `security-rl-convergence-assessment-v1` uses **timestep-based**
thirds, not episode-count thirds. Current proposed limits: at least 30 episodes in
initial/final thirds; final mean rises by at least 10% of the reward scale; four
final-third time-block means span at most 15%; absolute final-third OLS slope is
at most 10%; last-block drop from earlier middle/final-third blocks is at most 15%
(a lower but flat final third cannot hide a boundary collapse). The scale
is `max(100, abs(final_mean))`. Final-third explained variance must average above
zero with an OLS change no worse than -0.05 over that third. At least 8/10 seeds
must pass, with final means spanning at most 35% of the aggregate reward scale.
These are configurable definitions of plateau/consistency, **not a statistical
proof of optimality**. Missing evidence fails closed. Diagnostics are collected
after every PPO update, including the final one, to CSV; missing values are empty,
not invented zeros. Curves use a trailing 20-episode mean and fixed plot metadata.

Each candidate contains an exclusive-create `registration.json`, `shaped-<seed>/`
directories with episodes, diagnostics, manifest, policy, optional normalisation
state and hash-bound completion record; `analysis/` holds per-seed raw/smoothed
curves, reward/diagnostic PNGs, final-third summaries and aggregate JSON. A passing
`assessment.json` is reconstructed from raw evidence before `first_stable.json`
freezes exact config/criterion, commit, source hashes, topology/CVE hashes and every
checkpoint. Completed runs can be verified/skipped on restart; **partial runs are
not automatically retrained or overwritten**. Preserve them and resolve recovery
explicitly. The study lock prevents concurrent writers. Results remain gitignored.

Safe first commands: reproducibility and verification. Long training requires
reviewed thresholds, a clean committed checkout, PostgreSQL and preregistration.
No frozen stable candidate or final evaluation result is supplied by implementation.

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
