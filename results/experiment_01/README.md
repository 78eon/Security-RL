# experiment_01

This package was generated from dedicated frozen-policy evaluation, not from
training returns. It compares sparse and shaped PPO across
10 matched training seeds on topology seed
42, with 10 held-out episode
seeds per checkpoint. Policy parameters are hashed before and after evaluation;
no gradient updates occur.

- `metadata/`: exact code, configuration, topology, CVE and checkpoint provenance
- `raw/`: evaluation episodes and step-level evidence for every checkpoint
- `summaries/`: analysis JSON and per-run metrics
- `tables/`: paired statistical comparisons and dissertation-ready CSV
- `figures/`: plots generated only from evaluation outcomes
- `trajectories/`: trace-derived attack paths, examples and validation findings

Evaluation design complete: `True` (all 10 matched seeds exist in both arms).
This does **not** mean training converged: all 20 checkpoints fail the
preregistered stability criterion. Treat the current comparisons as provisional
pipeline evidence, not a confirmatory sparse-versus-shaped conclusion. See
`summaries/analysis.json` for exact convergence details and assumption warnings.
