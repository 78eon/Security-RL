# Essential baseline flowchart — drawing specification

Box/branch specification for the Essential baseline diagram, in the style of the
existing `rlredteam_baseline.png`. The shaped-vs-sparse toggle must be **explicit** —
give it its own decision diamond rather than burying it inside the reward box.

## Boxes and flow

```
        [ configs/topology.yaml ]        [ data/cve_catalogue.sqlite ]
        num_hosts 8, num_services 5      16 real CVEs from NVD
        num_exploits 8, r_sensitive 100  CVSS 3.7 - 10.0
                  |                              |
                  |  config_hash                 |  sha256 manifest
                  v                              v
        +---------------------+        +--------------------------+
        | nasim.generate()    |        | seeded stratified CVE    |
        | topology_seed       |------->| assignment per topology  |
        +---------------------+        +--------------------------+
                  |                              |
                  +--------------+---------------+
                                 v
                    +--------------------------+
                    |  NASim environment       |
                    |  obs Box(234)            |
                    |  actions Discrete(120)   |
                    +--------------------------+
                                 |
                                 v  (obs, reward, terminated, truncated, info)
                    +--------------------------+
                    |  NASimEventAdapter       |
                    |  resolves action ->      |
                    |  AttackEvent + CVE       |
                    +--------------------------+
                                 |
                                 v
                          < REWARD MODE? >          <-- THE TOGGLE. Decision diamond.
                            /          \
                     shaped/            \sparse
                          v              v
        +----------------------+   +------------------------+
        | CVE/CVSS x weight    |   | 1.0 if crown jewel     |
        | + tactic bonus       |   | else 0                 |
        |   recon    +1.0      |   |                        |
        |   exploit  +0.5      |   | (no severity signal)   |
        |   privesc  +2.0      |   |                        |
        | + crown jewel +100   |   |                        |
        | - failed action  -5  |   |                        |
        | first-success only   |   |                        |
        +----------------------+   +------------------------+
                          \              /
                           v            v
                    +--------------------------+
                    |  PPO (Stable-Baselines3) |
                    |  seeds 42-51             |
                    +--------------------------+
                                 |
                    +------------+------------+
                    v                         v
        +----------------------+   +------------------------+
        | PostgreSQL           |   | checkpoints -> runs/   |
        | experiments/         |   | GATED, not committed   |
        | episodes/steps       |   | (charter §10)          |
        +----------------------+   +------------------------+
                    |
                    v
        +--------------------------------+
        | Evaluation, n=10, seeds 42-51  |
        | ON NATIVE REWARD               |
        | paired t-test + Cohen's d      |
        | + Bonferroni over 5 metrics    |
        +--------------------------------+
```

## Things the drawing must show

1. **The toggle as a decision diamond**, with both branches drawn and rejoining before
   PPO. That is the whole point of the diagram.
2. **Both arms rejoin** — identical everything downstream. Reinforces that the ablation
   isolates the reward.
3. **The two hash artefacts** (`config_hash`, `sha256`) flowing into the PostgreSQL box,
   since those are what prove the arms were comparable.
4. **Evaluation labelled "on native reward"** — this is a deliberate methodological
   choice, not an implementation detail, and it will be asked about.
5. **`runs/` marked as gated** — dual-use weight gating, charter §10.

## Things to leave out (Essential scope only)

Neo4j, FastAPI/Celery, React dashboard, MLflow, DQN/A3C baselines, the full MITRE ATT&CK
mapper, PySide6 desktop app. All are Important/Optional tier and would muddy an
Essential-scoped diagram.

## Suggested caption

> Figure N — Essential-tier RLRedTeam pipeline. A seeded generator produces a random
> topology whose exploits are bound to real NVD CVEs by a deterministic stratified draw.
> The shaped-vs-sparse toggle is the sole difference between ablation arms; all other
> components, including the topology and CVE assignment, are held identical and verified
> by hash. Evaluation is performed on the native NASim reward in both arms so the
> comparison is well-posed.
