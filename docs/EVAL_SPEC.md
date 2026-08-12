# Evaluation specification (n=10)

Pre-registered before any results were generated. Fix this document before running the
grid; changing metrics after seeing results is what makes a study unfalsifiable.

## Design

| | |
|---|---|
| Arms | `shaped` (CVE/CVSS × weight + tactic bonus) vs `sparse` (1.0 crown-jewel, else 0) |
| Runs | n=10 per arm, seeds **42–51** |
| Blocking | paired per scenario — arms share topology seed, so each seed is a matched pair |
| Controlled | topology config hash and CVE manifest SHA-256 identical across arms |
| Varying | reward mode only |

`scripts/run_ablation.py` asserts the controlled quantities before any compute is spent.

**Topology seed policy — state which you are reporting.** Fixing the topology seed and
varying only the training seed measures *training variance on one network*. Varying both
measures *generalisation across networks*, which is the project's stated goal. They
answer different questions; `--vary-topology` selects the second.

## Metrics

All are in native NASim units or scale-free. **Every arm is evaluated under
`info["native_reward"]`, never the shaped reward** — the two arms optimise different
objectives, so their raw returns are measured with different rulers and the shaped arm
would win trivially on its own scale. Shaping is a training-time device; the native
reward is the task.

| # | Metric | Definition | Primary? |
|---|--------|-----------|:--------:|
| 1 | Path-discovery success rate | fraction of eval episodes reaching all crown jewels within the step limit | yes |
| 2 | Mean steps-to-goal | mean episode length, **conditional on success** | yes |
| 3 | Mean episode reward | mean native return per episode | yes |
| 4 | Reward variance | variance of native return across the 10 seeds | yes |
| 5 | High-value-target hit rate | fraction of episodes compromising ≥1 host whose assigned CVE scores ≥9.0 | yes |
| — | Mean CVSS of exploited hosts | descriptive; supports the severity-shift claim | no |
| — | Technique counts per ATT&CK ID | descriptive; `exfil` expected to be 0 (see below) | no |

Metric 2 is conditional on success: averaging steps over failed episodes silently
rewards an arm that fails fast, since a truncated episode has a fixed length.

## Statistics

- **Paired t-test** across the 10 matched seed pairs. Paired, not independent — each
  seed is one topology seen by both arms, and pairing removes between-topology variance,
  which is the largest noise source here.
- **Cohen's d** for paired samples, reported with **bootstrap 95% CIs** over the seeds.
- **Bonferroni** across the **5 pre-registered primary metrics** (α = 0.05/5 = 0.01).
  Descriptive metrics are reported without correction and labelled as exploratory.
- Report **mean and variance** for every metric, and include the per-seed table in an
  appendix.

**Power limitation, stated up front.** n=10 with Bonferroni at α=0.01 has roughly 80%
power only for large effects (d ≈ 1.5+). Lead with effect sizes and confidence intervals
rather than p-values. A non-significant result here is weak evidence of absence, and the
write-up should say so rather than let a power limitation read as a failed study.

## Results template

Populate from PostgreSQL; regenerate rather than transcribe.

### Table 1 — primary metrics

| Metric | Shaped (mean ± sd) | Sparse (mean ± sd) | Δ | t | p | p_bonf | Cohen's d [95% CI] |
|---|---|---|---|---|---|---|---|
| Path-discovery success rate | | | | | | | |
| Mean steps-to-goal (successes) | | | | | | | |
| Mean episode reward (native) | | | | | | | |
| Reward variance | | | | | | | |
| High-value-target hit rate | | | | | | | |

### Table 2 — per-seed detail (appendix)

| Seed | Topology seed | Arm | Success rate | Steps-to-goal | Native return | HVT hit rate |
|---|---|---|---|---|---|---|
| 42 | | shaped | | | | |
| 42 | | sparse | | | | |
| … | | | | | | |

### Table 3 — descriptive

| Metric | Shaped | Sparse |
|---|---|---|
| Mean CVSS of exploited hosts | | |
| T1046 / T1082 / T1018 / T1057 counts | | |
| T1210 (exploit) / T1068 (privesc) counts | | |
| Exfil tactic count | 0 (unreachable) | 0 (unreachable) |

## Provenance to record with every result

- `topology_config_hash` — identical across arms
- `cve_manifest_sha256` — identical across arms
- `git_sha` — the commit that produced the run
- seed and topology seed per episode

All four are columns in the `experiments` and `episodes` tables already.

## Known limitations to state in the write-up

1. **`exfil` (+1.5) is unreachable.** NASim's action set has no exfiltration action, so
   the tactic is defined but never triggers. Its count is reported as 0. Pending
   supervisor decision: drop it, or redefine it as root-on-crown-jewel (which already
   pays +100 and would double-count).
2. **Anti-farming changes the reward surface.** Tactic bonuses pay only on the first
   *informative* success per target, and exploits/privescs only when access level rises.
   Without this, scan-spam earns up to +1000 against a +100 crown jewel. This bounds
   shaping but means the reward is history-dependent within an episode, not a pure
   function of the current step.
3. **CVE assignment is synthetic.** Generated topologies have positional exploit names,
   so real CVEs are bound to them by a seeded stratified draw rather than by genuine
   service correspondence. The severities are real and sourced from NVD; the pairing to
   a specific generated service is not.
4. **n=10 is underpowered** for anything but large effects — see above.
