# RLRedTeam

Simulation-only reinforcement learning agent for attack-path discovery on randomised
enterprise topologies, with a CVE/CVSS-severity-weighted reward.

MSc dissertation project (CT7P01NI). **Essential tier.** Simulation only — no live,
production or third-party network is used, and no human subjects are involved.

## Status

| Component | State |
|---|---|
| Module 0 — frozen CVE catalogue (SQLite + SHA-256 manifest) | done, tested |
| Reward core — CVE/CVSS × weight + tactic bonus, sparse toggle | done, tested |
| Module 4 — PostgreSQL episode logger | done, tested |
| Topology generator + NASim adapter + random rollout | done, tested |
| PPO training loop (`train.py`) | in progress — see `docs/PPO_BRIEF.md` |
| Evaluation n=10 | spec written (`docs/EVAL_SPEC.md`), awaiting runs |

147 tests passing, ruff clean.

## Quick start

```bash
cp .env.example .env          # add your NVD_API_KEY (never committed)
make build                    # build the training image
make db-up                    # start PostgreSQL on :5433
make test                     # full suite, including Postgres and NASim integration
make rollout                  # deterministic random-policy rollout
```

`make` auto-detects docker or podman.

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
  storage/           Module 4: PostgreSQL schema and batched episode logger.

configs/             topology.yaml, shaped.yaml, sparse.yaml
data/                cve_catalogue.sqlite, its manifest, raw NVD provenance JSON
docs/                PPO_BRIEF, EVAL_SPEC, FLOWCHART_SPEC, report_snippets
scripts/             rollout_random.py, run_ablation.py (stub)
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

Recorded in `docs/EVAL_SPEC.md`, summarised here:

1. The `exfil` tactic bonus (+1.5) is **unreachable** — NASim has no exfiltration action.
   It is defined so its zero count is visible, pending a supervisor decision.
2. Tactic bonuses pay only on the first *informative* success per target. Without this,
   scan-spam earns up to +1000 against a +100 crown jewel and farming becomes optimal.
   This bounds shaping but makes the reward history-dependent within an episode.
3. CVE-to-exploit pairing is synthetic. The severities are real and NVD-sourced; the
   binding to a specific *generated* service is a seeded draw, not genuine service
   correspondence.
4. n=10 with Bonferroni is underpowered for anything but large effects.
