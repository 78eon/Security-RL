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
reports; it does not autonomously exploit systems. See
[the hybrid lab runbook](docs/HYBRID_LAB_RUNBOOK.md).


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

The random-topology and cross-profile protocols are documented in
[Phase 3](docs/PHASE_03_UNSEEN_GENERALISATION.md) and
[Phase 4](docs/PHASE_04_INFRASTRUCTURE_PROFILES.md).

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
  nasim_adapter.py   The only module that knows about both NASim and the reward core.
  enterprise/        Hidden truth, agent knowledge, observations and on-prem simulator.
  train.py           PPO entry point: seeding, episode collection, logging.
  storage/           Module 4: PostgreSQL schema and batched episode logger.

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
environment. All environment-specific code lives in `nasim_adapter.py`.

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
