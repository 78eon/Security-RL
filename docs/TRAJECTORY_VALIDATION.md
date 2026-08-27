# Trajectory Validation

Status: **pending a current, dedicated frozen-policy evaluation run**.

Existing `runs/` traces are not accepted as evidence because they were produced
by older commits and the current ablation analyzer consumes training episodes.
This document will be completed from `results/experiment_01/raw/` after the
evaluation pipeline records successful, failed and highest-reward trajectories.

Required checks:

- repeated action count and longest identical-action streak;
- repeated successful actions receiving shaping;
- invalid/impossible actions receiving positive reward;
- high shaped return without goal completion;
- unnecessary actions after a viable path is available;
- goal avoidance and step-limit loops;
- observation/topology leakage;
- representative sparse/shaped success and failure traces.
