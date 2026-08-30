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

## Current status

Implementation validation and the final commit are recorded here after the
phase gates complete.
