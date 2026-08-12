# Report text drafts — Week 3 Task 1

Paste into the report .docx. These close conditions C1 (Hevner), C6 (novelty softening)
and C8 (dual-use gating), plus the three scope lines.

**One item is blocked:** the work plan asks to "cite the TechScience/CMES paper as
related work and frame your delta". That paper is not identified in the work plan — I
need its title, authors or DOI before drafting that paragraph. Everything else is below.

---

## 1. Scope lines (decision 2, to be signed off)

> **Simulation scope.** The Essential tier of this project is conducted entirely in
> simulation, using the Network Attack Simulator (NASim) Gym environment. No live,
> production or third-party network is used at any point, and no human subjects are
> involved.

> **Environment scope.** CybORG and Microsoft CyberBattleSim remain Important-tier and
> are explicitly out of scope for the Essential phase. Should Essential complete and be
> evaluated, they may be revisited as a generalisation study; no Essential deliverable
> depends on them.

> **Weight gating.** Trained attack-policy weights are gated by default. They are not
> included in any public release of this project unless a release decision — withhold,
> redact, or access-control — is agreed with the supervisor before publication. Source
> code, environment configurations and the CVE catalogue may be released publicly; the
> trained policy is the sensitive artefact.

*Implementation note for the meeting: this is enforced, not just stated. `runs/` is in
`.gitignore`, and all checkpoints are written there.*

---

## 2. C1 — Hevner et al. (2004) citation

**Reference list entry (Harvard):**

> Hevner, A.R., March, S.T., Park, J. and Ram, S. (2004) 'Design science in information
> systems research', *MIS Quarterly*, 28(1), pp. 75–105.

**Method section text (§7):**

> This project follows a Design Science Research (DSR) methodology as articulated by
> Hevner et al. (2004), in which knowledge is generated through the construction and
> rigorous evaluation of an artefact. The artefact here is RLRedTeam: a reinforcement
> learning agent, a CVE-severity-weighted reward engine, and the supporting environment
> and persistence layer. Following the DSR guidelines, the artefact is evaluated
> empirically rather than argued for analytically — specifically through a controlled
> ablation comparing a severity-shaped reward against a sparse baseline across ten
> matched random topologies, with effect sizes and confidence intervals reported
> alongside significance tests. The design cycle is made explicit through a versioned
> charter, a conditions tracker, and configuration hashes recorded with every
> experimental run.

---

## 3. C6 — novelty softening

Replace every instance of "first", "inaugural", "pioneering", "novel framework" and "new
benchmark". Suggested replacement paragraph:

> RLRedTeam is not the first reinforcement learning framework for automated red-teaming.
> CybORG and the associated CAGE challenges, Microsoft's CyberBattleSim, and NASim are
> established open environments, and prior work by Standen et al. (2021) and Gangupantulu
> et al. (2021) has applied reinforcement learning to attack-path discovery. The
> contribution of this project is therefore not priority but **a specific combination**:
> the integration of CVE-severity-weighted, MITRE-aligned reward shaping with
> generalisation across randomised topologies, delivered as a single reproducible
> artefact with persistent episode-level logging and an honest statistical evaluation.
> Each component exists in the literature; their combination, and the empirical question
> of whether severity weighting measurably alters discovered attack paths, is what is
> examined here.

**Search-and-replace checklist:** `first` · `inaugural` · `pioneering` · `novel` ·
`new benchmark` · `state-of-the-art` · `unprecedented`.

---

## 4. C8 — dual-use line for ethics/limitations

> **Dual-use considerations.** A trained attack-path-discovery policy is inherently
> dual-use: the same artefact that helps a defender identify exploitable paths could
> assist an attacker in doing so. This project mitigates that in three ways. First, all
> work is conducted in simulation against synthetic topologies, so no policy is trained
> against, or transferable to, a real network. Second, trained policy weights are gated
> by default and excluded from public release pending an explicit, supervised release
> decision; the repository's version control is configured to exclude them. Third, the
> released artefacts — source code, environment configuration and the CVE catalogue —
> contain no capability beyond what is already publicly available in NASim and the
> National Vulnerability Database. The CVE records used are public, historical, and
> largely patched; the catalogue is a severity lookup table, not an exploit collection.

---

## 5. Methodology text you can now write from real numbers

Useful for the build chapter — these are measured, not estimated:

> The CVE catalogue comprises 16 records retrieved from the NVD API 2.0 and frozen as a
> committed SQLite artefact with a SHA-256 manifest. Base scores span CVSS v3.1 3.7
> (LOW) to 10.0 (CRITICAL). Candidates lacking a CVSS v3.1 metric were rejected rather
> than converted from v2, since no defensible conversion exists; three candidates were
> excluded on this basis. Severity is mapped to a reward weight by a normalised power
> law, w = w_min + (w_max − w_min)·(score/10)^γ with γ = 2.0, chosen a priori to widen
> the dynamic range: real remotely-exploitable CVEs cluster between 7.8 and 10.0, where a
> linear map spans only 1.4×, below the seed-to-seed variance observed at n=10. The
> committed catalogue achieves a contrast ratio of 2.84×, and the stratified assignment
> procedure preserves at least 1.8× on every evaluation seed.

> Topologies are generated rather than hand-specified, using NASim's scenario generator
> under an explicit seed, so that randomisation and reproducibility are simultaneously
> satisfied: a topology is fully recoverable from its configuration hash and seed. The
> generator is configured with more exploits (8) than services (5). This is deliberate:
> under a one-exploit-per-service configuration, 7 of 8 hosts in the stock benchmark
> topology admitted exactly one applicable exploit, leaving the agent no choice between
> severities and rendering the CVE weighting a constant incapable of influencing policy.
> At the chosen configuration, all eight hosts offer a genuine choice across seeds 42–51,
> with a mean of 4.0 applicable exploits per host.
