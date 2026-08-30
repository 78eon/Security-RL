# Phase 5 — Backend-Driven Desktop Simulation

## Objective

Make the native PySide6 desktop application demonstrate a running enterprise
attack-path simulation as a graph, using the same Python backend and profile
configuration as the research environment. No profile, topology, entity list
or trajectory may be fabricated by the view.

## Security boundary

The workspace lists configured **simulated enterprise profiles**, not nearby
wireless or production networks. Running it opens no socket, executes no scan,
authenticates to no asset and launches no exploit. The generated graph exists
only in process memory. Real lab discovery remains a separate explicitly
authorized and scope-gated workflow.

## Data flow

```text
profile + seed selected in Qt
        -> QThreadPool worker
        -> ApplicationBackend.run_simulation()
        -> generate_profile_topology()
        -> EnterpriseCyberEnv + AgentKnowledge transitions
        -> recorded prerequisites/outcomes
        -> causal path reconstruction
        -> plain SimulationData
        -> native QPainter replay + entity table
```

Qt widgets remain on the main thread. The blocking simulation call runs in the
existing worker pool and returns a frozen plain-data object across the signal
boundary.

## Demonstration agent

The interactive workspace uses the deterministic knowledge-only feasibility
agent. This makes a repeatable and fast UI demonstration without loading Torch
into the GUI image. It is labelled as such and is not presented as PPO research
evidence. The trained policy results remain available through the persisted
Attack Paths and experiment workspaces.

## Podman-only packaging

The GUI image adds only pinned `gymnasium`, `numpy` and `PyYAML`, which are
needed by the offline graph simulator. It still excludes Torch, PPO, browser
engines, JavaScript, Electron and live scanning tools. Nothing is installed on
the host.

## Acceptance gates

- Backend profile list comes from `DeploymentProfile`.
- Selecting a profile/seed generates an actual typed topology.
- Result contains topology hash, node/edge details, metrics and a causal trace.
- Hybrid detail contains both legacy and cloud entities.
- Graph replay uses backend-returned trajectory data.
- Simulation runs through `QThreadPool`, never the Qt UI thread.
- Socket-poison test proves the backend demo performs no network access.
- GUI image, headless GUI suite, backend suite, lint and compilation pass.

## Implementation record

- Implementation commit: `6523bdecfabb54fe96a0d6b96b9bac10710d22ad`
- Backend methods: `simulation_profiles()` and `run_simulation()`
- Plain result contract: `SimulationData`
- Native workspace: `SimulationPage`
- Causal reconstruction was extracted into the dependency-light
  `enterprise/trajectory.py`; the GUI no longer imports the Torch-backed
  training stack.

The first GUI-image test exposed that dependency leak and failed with
`ModuleNotFoundError: torch`. The architecture was corrected rather than adding
Torch to the desktop image. The rebuilt image then passed all desktop tests.

## Validation evidence

- Rebuilt Podman GUI image: `localhost/rlredteam-gui:latest`
- Headless native GUI suite: 41 passed
- Focused backend/reconstruction suites: 17 passed
- Full Podman suite from clean implementation commit: 401 passed, 13 skipped,
  2 pre-existing SciPy precision warnings; 169.88 seconds
- Full non-training suite: 383 passed, 12 skipped, 19 deselected, 2 existing
  warnings; 98.34 seconds
- Ruff over `src`, `gui`, `tests`, `tools` and `scripts`: passed
- Python compilation with bytecode redirected to `/tmp`: passed
- Podman Compose validation: passed
- Offscreen visual smoke: 1280×800 native render, hybrid seed 2001,
  goal reached in 25 raw steps with 16 causal events
- Visual smoke SHA-256:
  `09a3ea1d784dd4b13e2bc576f338fa57cd0ba794a68d566e697b275d2f33ef9a`

## Phase verdict

| Requirement | Verdict |
|---|---|
| Backend-derived profile selector | PASS |
| Backend-generated typed topology | PASS |
| Worker-thread execution | PASS |
| Native graph replay | PASS |
| Entity details from generated graph | PASS |
| Causal, non-omniscient path | PASS |
| No hardcoded simulation result | PASS |
| No real network access | PASS |
| No Torch/PPO in GUI image | PASS |
| Podman-only installation | PASS |

Phase 5 is complete. The desktop now demonstrates the running Python simulator
through a native graph and entity view. It intentionally does not discover or
display nearby real networks.
