# Frontend plan — RLRedTeam analyst desktop app

PySide6 desktop application. Four views: results, attack-path replay, run browser, and
training control. Read **and** launch.

> **Scope note.** The charter places the dashboard in Optional (Tier 2), gated behind
> Essential and Important completion. Essential is not finished — the n=10 evaluation has
> not been run. Worth stating in the next supervision meeting why this is starting now, so
> it reads as a deliberate choice rather than scope drift (risk R7).

---

## 1. The constraint that shapes everything

The GUI runs **natively on the host**; training runs **inside the container**.

| | Host (GUI) | Container (training) |
|---|---|---|
| Python | 3.13.14 | 3.11 |
| Has torch / nasim | **No** | Yes |
| Has PySide6 | Will install | No |
| Reaches Postgres | Yes, `localhost:5433` | Yes, service name |

The GUI therefore **cannot train in-process** — the host has neither the right Python nor
torch, and installing them would duplicate the environment the whole project treats as the
reproducibility boundary.

**The GUI shells out.** Training is launched as a subprocess:

```
podman compose run --rm app python -m rlredteam.train --seed N --timesteps T ...
```

Three reasons beyond the dependency issue, each independently sufficient:

1. **A 200k-step run would freeze the UI.** Qt is single-threaded for widgets; a blocking
   call in the event loop makes the window unresponsive and the OS marks it "not
   responding" mid-demo.
2. **A training crash would take the app down** if it ran in-process.
3. **Killing a run** is trivial with a process handle and painful in-process.

This keeps the GUI's dependencies tiny: `PySide6`, `psycopg`, `pyqtgraph`. No torch.

---

## 2. Data sources

| Source | Read for |
|---|---|
| PostgreSQL `experiments` | run list, provenance hashes, seeds |
| PostgreSQL `episodes` | learning curves, per-run metrics, comparison |
| PostgreSQL `steps` | **attack-path replay** — needs `--log-steps` |
| `runs/<name>/config.snapshot.json` | topology + reward settings for a run |
| `runs/<name>/episodes.csv` | **live monitoring** — flushed per episode during training |
| `runs/<name>/summary.json` | headline numbers |
| `data/cve_catalogue.sqlite` | CVE detail on hover in the replay view |

### Gap to close first

`config.snapshot.json` records host and subnet *counts*, but not **which host sits in which
subnet, nor the subnet adjacency matrix**. The replay view needs both to draw a network.

`topology.describe()` must be extended to persist `subnets` (sizes) and `topology` (the
adjacency matrix). Small change to `src/rlredteam/topology.py`, but it must land before the
replay view can be built, and old runs will lack it — the view should degrade to a simple
grouped layout when the field is absent rather than crash.

---

## 3. Module layout

```
gui/                                  ← host-side, separate from src/rlredteam
  __main__.py                         python -m gui
  app.py                              QApplication bootstrap, theme load
  theme.qss                           ← YOUR DESIGN LANDS HERE
  config.py                           DB connection, repo paths, settings persistence

  data/
    repository.py                     all SQL; returns plain dataclasses, never widgets
    runs.py                           reads runs/ folder artefacts
    models.py                         RunSummary, EpisodeRow, StepRow, TopologyView

  workers/
    query.py                          QRunnable wrappers so no query blocks the UI thread
    trainer.py                        QProcess wrapper: launch, stream, kill
    watcher.py                        episodes.csv tail for live curves

  views/
    main_window.py                    shell, navigation, status bar
    runs_view.py                      1. browse runs + provenance
    results_view.py                   2. charts and shaped-vs-sparse comparison
    replay_view.py                    3. attack-path playback
    train_view.py                     4. launch and monitor

  widgets/
    curve_chart.py                    pyqtgraph live line chart
    metric_tile.py                    single headline number
    network_canvas.py                 QGraphicsScene network diagram
    provenance_badge.py               hash chip with copy-to-clipboard
    severity_pill.py                  CVSS band chip (LOW…CRITICAL)
```

**Rule: `data/` and `workers/` never import from `views/`.** One-directional, so the data
layer stays testable headlessly with no QApplication.

---

## 4. The four views

### 4.1 Runs — the browser

Table of every experiment: name, reward mode, seeds, episode count, created, and the three
provenance hashes. Selecting a run shows its episodes and its config snapshot.

The one thing this view must make obvious: **whether two runs are comparable.** Same
`topology_config_hash` and `cve_manifest_sha256` means yes. Show it as a visible state, not
a hash string the user has to diff by eye.

### 4.2 Results — the charts

- Learning curve: native return vs episode, one line per arm, mean with a variance band
- Steps-to-goal over training
- Success rate over training
- Mean CVSS of exploited hosts — the severity-shift claim
- Metric tiles: the five primary evaluation metrics
- Export button → publication-quality PNG for the dissertation

**Charts must plot native reward, not shaped**, with a visible toggle. The two arms optimise
different objectives, so comparing shaped returns is comparing different rulers. The UI
should make the correct comparison the default and the incorrect one deliberate.

### 4.3 Replay — the attack path

The most demonstrable thing the project can show, and the reason `steps` exists.

- Network diagram: hosts grouped by subnet, edges from the adjacency matrix
- Crown jewels visually distinct; host state coloured as undiscovered → discovered →
  user access → root
- Transport controls: step back / play / pause / step forward, speed selector
- Step detail panel: action, target, success, CVE id, CVSS score, points earned
- Timeline scrubber across the episode

Requires runs logged with `--log-steps`. When a run lacks step rows, say so and offer to
re-run rather than showing an empty canvas.

### 4.4 Train — launch and monitor

- Form: seed, topology seed, timesteps, reward config, `--postgres`, `--log-steps`
- Launch → `QProcess` running the compose command
- Live: reward curve updating as episodes complete, plus raw log output
- Stop button that actually kills the process group
- Queue mode for the full 2×10 grid, with progress

**Live updates come from `runs/<name>/episodes.csv`**, which is already flushed after every
episode. Polling that file on a timer is simpler and more robust than parsing stdout, and it
works whether or not `--postgres` was passed. Use a `QTimer` poll rather than
`QFileSystemWatcher` — the watcher is unreliable across bind mounts.

---

## 5. Threading model

Qt's rule: **only the main thread touches widgets.** Violating it produces crashes that are
intermittent and near-impossible to reproduce, so the structure has to prevent it rather
than rely on discipline.

| Work | Mechanism |
|---|---|
| SQL queries | `QRunnable` on `QThreadPool`, result delivered by signal |
| Reading `runs/` files | same |
| Training subprocess | `QProcess` — already async, no thread needed |
| Live CSV tail | `QTimer` on the main thread; the read is small and cheap |

Workers emit signals carrying plain dataclasses. No worker ever holds a widget reference.

---

## 6. Charting

**pyqtgraph in-app, matplotlib for export.**

pyqtgraph is Qt-native and fast enough to redraw a live curve every episode without
stuttering; matplotlib in a Qt canvas is noticeably slower when updating continuously.
But matplotlib produces better publication output, so the Export button renders through it
with the Agg backend rather than screenshotting the widget.

Palette is validated colourblind-safe: `#2a78d6` (blue) for arm 1 and `#eb6834` (orange)
for arm 2 — worst-case CVD separation ΔE 24.7, well clear of the ≥8 threshold. Severity
uses its own separate scale so it never competes with series identity.

---

## 7. What I need from your design

Send whatever form suits you — Figma, screenshots, sketches, or a description. Most useful:

1. **Layout per view** — where navigation lives, panel arrangement, what is primary
2. **Colour palette** — background, surface, text, accent, and states (success/failure/warning)
3. **Typography** — families and sizes for headings, body, and numeric/monospace data
4. **Component styling** — buttons, tables, inputs, cards
5. **Light and dark**, if you want both

### One important caveat about Qt styling

Qt's QSS is **a CSS-like subset, not CSS**. It has no flexbox, no grid, no CSS variables, no
transforms, and limited pseudo-selectors. Layout is done in Python with Qt layout managers,
not in the stylesheet.

So a web-style design translates in spirit rather than literally. **Send the visual
direction — spacing rhythm, colour, hierarchy, component look — and I will build the layout
in Qt to match.** Pixel-exact reproduction of a web mockup is not realistic; a faithful
interpretation is.

---

## 8. Build order

| Phase | Deliverable | Why this order |
|---|---|---|
| 0 | Host venv, `gui/` skeleton, empty window opens | Proves the toolchain before any features |
| 1 | Extend `topology.describe()` to persist subnets + adjacency | Blocks the replay view; do it before generating grid data |
| 2 | `data/` + `workers/` with headless tests | Testable without a GUI; the risky part is the SQL, not the pixels |
| 3 | Runs view | Simplest real view; proves the data layer end to end |
| 4 | Results view + charts | The dissertation payload |
| 5 | Train view | Needs QProcess plumbing and a kill path |
| 6 | Replay view | Most visual work; benefits from everything above being stable |
| 7 | Apply your design as `theme.qss` | Styling last, over a working app |

Phases 1 and 2 do not depend on the design, so they can start immediately while you finalise
the UI/UX.

---

## 9. Risks

| Risk | Mitigation |
|---|---|
| Topology structure not persisted — replay cannot draw the network | Phase 1 fixes it; older runs degrade to a grouped layout |
| `steps` table is empty unless `--log-steps` was passed | Train view defaults it on; replay states clearly when data is absent |
| GUI blocking on a slow query | Everything through the worker pool; no SQL on the main thread |
| Orphaned training processes after the app closes | Track PIDs, kill the process group on exit, warn on close if a run is active |
| Qt cannot reproduce a web mockup exactly | Agreed up front (§7) — faithful interpretation, not pixel-matching |
| GUI work displaces the unfinished n=10 evaluation | Run the grid first (~74 min) so Essential completes regardless |

---

## 10. Dependencies to add

Separate from the training environment — a `requirements-gui.txt` installed into a host venv:

```
PySide6==6.11.1
pyqtgraph==0.14.0
psycopg[binary]==3.2.3
matplotlib==3.9.2       # export only
```

Deliberately no torch, no nasim, no stable-baselines3. If the GUI ever needs them, the
process boundary has been broken and training has leaked into the UI.
