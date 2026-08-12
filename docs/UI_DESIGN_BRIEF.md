# UI/UX master prompt — RLRedTeam desktop app

Copy everything below the line into your design tool. It is self-contained: it assumes no
prior knowledge of the project and carries real data so mockups use true content, never
placeholder text.

---

## THE BRIEF

Design a **desktop application** for a cybersecurity research tool called **RLRedTeam**.

### What the product is

RLRedTeam trains an AI agent to discover attack paths through simulated computer networks.
The agent starts knowing almost nothing about a network, then scans, breaks into machines,
and escalates privileges until it reaches high-value targets. It learns by trial and error.

The research question is whether feeding the agent **real vulnerability-severity data**
(CVSS scores from the US National Vulnerability Database) helps it learn faster than a plain
reward that only says "you reached the target". The app compares two versions of the agent —
**"shaped"** (severity-aware) and **"sparse"** (plain) — and lets a researcher inspect,
replay and launch training runs.

Everything is simulated. No real networks are attacked.

### Who uses it

A single postgraduate researcher, working alone, on their own machine. They use it to:
inspect experiment results, produce figures for a dissertation, replay what the agent did,
and start new training runs. They will also **demonstrate it live to an academic supervisor**
— so it must look credible and legible on a projector, not just on a laptop.

This is a serious research instrument, not a consumer product. It should feel closer to a
scientific workbench or a professional analysis tool than to a marketing dashboard. No
gamification, no celebratory animation, no motivational copy.

### Hard technical constraints — please read, they affect what you can design

The app is built in **Qt (PySide6)**, not the web. Qt styling is a **CSS-like subset**, and
this genuinely limits what is buildable:

- **No flexbox, no CSS grid.** Layout uses Qt layout managers — horizontal, vertical, grid,
  splitter, stacked. Design in terms of rows, columns, splitters and fixed panels.
- **No CSS variables, no transforms, no filters, no blend modes.**
- **No box-shadow on arbitrary widgets** (only crude approximations). Avoid soft elevation
  and floating-card looks; use borders, background steps and dividers to separate surfaces.
- **No rounded corners on some native controls**, and rounding is unreliable on others.
  Prefer squared or minimally rounded geometry.
- **No custom scrollbar overlays** beyond basic recolouring.
- **Animation is limited.** Assume none except simple progress indicators.
- **Icons must be flat SVG or a bundled icon font.** No emoji in the UI.

Design for **1440×900 minimum**, resizable up to 2560 wide. Panels must survive resizing.

Deliver **both a light and a dark theme**. Dark should be a designed palette, not an
inversion — researchers run this in dim rooms and also project it in bright ones.

### The four screens

Navigation between them is persistent — a left sidebar or a top tab bar, your call.

---

#### SCREEN 1 — Runs (the browser)

A table of every training run, with a detail panel for the selected one.

Real columns and sample rows:

| Run name | Reward | Seeds | Episodes | Created |
|---|---|---|---|---|
| shaped-s42-t42 | shaped | 42 | 164 | 12 Aug 16:42 |
| sparse-s43-t43 | sparse | 43 | 16 | 12 Aug 16:43 |

The detail panel shows three **provenance fingerprints** — long hashes that prove two runs
used identical inputs:

```
topology config hash   2aa12404d40a23d0
CVE manifest sha256    c14575707519311e90a571e1b27d0bed16b2d10e41757f60a135ab8756137205
reward config hash     a12546fb33160142
git commit             508ac4f
```

**The single most important thing this screen must communicate:** whether two selected runs
are *comparable* — meaning their topology hash and CVE hash match. A researcher must be able
to see this at a glance without diffing 64-character strings by eye. Design a clear
comparable / not-comparable state. This is the screen's whole reason to exist.

---

#### SCREEN 2 — Results (charts and comparison)

The screen that produces dissertation figures.

Five headline metrics, displayed as a row of stat tiles. Real values:

```
Success rate            1.00      (vs 0.90 random baseline)
Mean steps to goal      443       (vs 565 random)
Mean episode reward    -747       (vs -1279 random)
Reward variance         182.4
High-value hit rate     0.72
```

Below them, line charts comparing the two agents across training:

- Episode reward over time
- Steps to goal over time
- Success rate over time
- Average vulnerability severity exploited

Each chart shows **two series** — "shaped" and "sparse" — as a mean line with a shaded
variance band, across ten runs.

**Series colours are fixed and must not be changed** — they are validated colourblind-safe:
- shaped: `#2a78d6` (blue)
- sparse: `#eb6834` (orange)

Design needed for: chart frame, axes, gridlines, legend, tooltip, variance band treatment,
and an **Export figure** action producing a publication-quality PNG.

There is also a **critical toggle**: charts can plot either "native reward" (the correct,
fair comparison) or "shaped reward" (misleading, because the two agents are scored on
different scales). Native must be the obvious default, and choosing shaped must feel like a
deliberate act. Design that control so the correct choice is the easy one.

---

#### SCREEN 3 — Attack path replay

The most visual screen, and the one that will be demonstrated live.

A **network diagram** — 8 machines arranged in 5 subnets — with playback controls that step
through what the agent did, move by move.

Each machine has a visible state that changes during playback:
- **undiscovered** — agent doesn't know it exists
- **discovered** — known but not compromised
- **user access** — broken into, limited privileges
- **root access** — fully controlled
- **crown jewel** — a high-value target (visually distinct at all times)

Lines between subnets show which can reach which.

Controls: step back, play/pause, step forward, speed selector, and a **timeline scrubber**
across the episode (episodes run 200–1000 moves).

A detail panel shows the current move:

```
step 247 of 443
action      e_srv_4_os_1  (Exploit)
target      host (3, 1)
result      SUCCESS — gained user access
CVE         CVE-2019-11510
severity    10.0  CRITICAL
points      +2.50
```

Vulnerability severity has its **own colour scale**, separate from the two series colours:

```
LOW       3.7      MEDIUM    5.3
HIGH      7.8      CRITICAL  9.8
```

Design a severity indicator (chip, pill or badge) using a green → yellow → orange → red
progression, distinct enough from the blue/orange series colours to never be confused.

---

#### SCREEN 4 — Training control

A form to launch runs, and live monitoring of one in progress.

Form fields: training seed, topology seed, number of steps (default 200,000), reward type
(shaped / sparse), and two checkboxes — log to database, log every move.

While running:
- A live-updating reward chart, redrawn as each episode finishes
- Progress: `episode 47 · 61,000 / 200,000 steps · ~2m 10s remaining`
- A scrolling log output panel
- A **Stop** button that must be unmistakable and always reachable

Also design a **queue view** — the full experiment is 20 runs (2 agent types × 10 seeds,
~74 minutes total). Show queued / running / done / failed states and overall progress.

---

### Content and data character

Design around these realities:

- **Numbers are frequently negative** (rewards like `-1279`) and need aligned decimals.
  Use tabular figures wherever digits sit in columns.
- **Identifiers are long and monospace**: `CVE-2019-11510`, `e_srv_4_os_1`,
  `2aa12404d40a23d0`, `shaped-s42-t42`. They need a monospace face and room to breathe.
- **Some tables are long** — hundreds of episodes, hundreds of steps.
- **Empty states are real and common.** A fresh install has no runs. A run without
  `--log-steps` has no replay data. Design these, and make them explain the fix rather than
  just saying "no data".
- **Failure states matter**: a training run can crash, the database can be unreachable.

### Deliverables

1. Layouts for all four screens, light and dark
2. A colour palette: background, surface levels, borders, primary/secondary/muted text,
   accent, and states (success / warning / error / running)
3. Typography: a UI face and a monospace face, with a defined size and weight scale
4. Components: buttons (primary/secondary/danger), tables, form inputs, dropdowns,
   checkboxes, tabs or sidebar navigation, stat tiles, chips/badges, progress bars, tooltips,
   empty states, dialogs
5. Chart styling: axes, gridlines, legend, tooltip, variance bands
6. Icon direction — flat, single-weight, no emoji

### Direction

Choose a considered visual identity fitting a **security research instrument**. Deliberately
avoid the current generic-AI-dashboard look: purple-to-blue gradient headers, glassmorphism,
oversized rounded cards floating on tinted backgrounds, emoji as section markers, and
Inter-for-everything. This is closer to an oscilloscope, an audio editor, a scientific
analysis suite, or a professional IDE than to a SaaS marketing dashboard.

Dense but not cramped. Information-first. It should look like a tool someone works in for
hours, and it should look credible projected onto a wall in a viva.
