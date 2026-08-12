# PPO handover brief

**You write `train.py`. This document contains no training code by design** — your
supervisor requires the PPO loop to be student-owned and reproducible without external
assistance. Everything below is reference material so you can wire it yourself and
explain it in the meeting.

Everything up to the environment boundary is built and tested: the CVE catalogue, the
reward engine, the topology generator, the NASim adapter and the PostgreSQL logger.
`train.py` is the one piece left.

---

## 1. The observation space — be ready to explain this

Measured on the frozen config (`configs/topology.yaml`, 8 hosts, seed 42):

```
observation_space : Box(234,)   float32, values in [0.0, 100.0]
underlying shape  : (9, 26)  ->  9 rows x 26 columns, flattened to 234
```

**Why 9 rows for 8 hosts.** One row per host, plus a final **auxiliary row** carrying
the result of the last action (success/failure and error flags). NASim appends it so a
feed-forward policy can see the outcome of what it just did without needing memory.

**What the 26 columns hold** (indices from `nasim.envs.host_vector.HostVector`):

| Index | Field | Meaning |
|------:|-------|---------|
| 0–9 | address / auxiliary | host address encoding and action-result flags |
| 10 | `compromised` | 1 if the agent has any access to this host |
| 11 | `reachable` | 1 if the host can be reached from a compromised host |
| 12 | `discovered` | 1 if the host is known to exist |
| 13 | `value` | reward value of the host (100 for crown jewels) |
| 14 | `discovery_value` | reward for discovering it |
| 15 | `access` | 0 none, 1 user, 2 root |
| 16–17 | OS indicators | one bit per OS (2 OS configured) |
| 18–22 | service indicators | one bit per service (5 services configured) |
| 23–25 | process indicators | one bit per process (3 processes configured) |

So `26 = 10 + 6 + num_os(2) + num_services(5) + num_processes(3)`. Change
`configs/topology.yaml` and this width changes with it — do not hardcode 234.

**Partial observability.** `fully_obs=False` is the NASim default and we keep it. The
agent starts knowing almost nothing; values are filled in only as scans and exploits
reveal them. This is what makes the task a *discovery* problem rather than a
shortest-path problem, and it is the honest framing for attack-path discovery.

## 2. The action space

```
action_space : Discrete(120)
```

Flat and discrete. For 8 hosts, 8 exploits and 3 privescs:

| Count | Action type |
|------:|-------------|
| 8 | ServiceScan (one per host) |
| 8 | OSScan |
| 8 | SubnetScan |
| 8 | ProcessScan |
| 64 | Exploit (8 hosts × 8 exploits) |
| 24 | PrivilegeEscalation (8 hosts × 3 privescs) |

General formula: `num_hosts × (4 + num_exploits + num_privescs)`.

`env.action_space.get_action(idx)` returns the `Action` object, with `.name`, `.target`,
`.cost`, `.prob`. **`info` from `step()` carries neither the action nor the target** —
this is why `NASimEventAdapter` resolves the action from the index itself. Worth knowing;
it is the least obvious thing about the NASim API.

**Most actions are invalid at any moment** (you cannot exploit a host you have not
discovered). Invalid actions are not masked — they simply fail, costing a step and, under
the shaped reward, −5. Early training is therefore mostly failure, and that is expected.

## 3. What to import

```python
from rlredteam.topology import TopologyConfig, make_env
from rlredteam.catalogue import CVECatalogue
from rlredteam.reward import RewardConfig
from rlredteam.nasim_adapter import RewardWrapper
from rlredteam.storage.postgres_logger import EpisodeLogger, EpisodeRecord, StepRecord
from rlredteam.manifest import digest
```

Building one wrapped environment:

```python
topology_config = TopologyConfig.from_yaml()              # configs/topology.yaml
reward_config   = RewardConfig.from_yaml(path)            # configs/shaped.yaml | sparse.yaml
catalogue       = CVECatalogue.open_default()

env = make_env(topology_config, topology_seed=seed)
env = RewardWrapper(env, catalogue, topology_seed=seed, reward_config=reward_config)
```

`RewardWrapper` is a `gymnasium.Wrapper`, so SB3 accepts it directly. It calls
`assert_goal_dominance()` at construction and raises if the reward config would let
shaping out-earn the goal.

Every `step()` returns `info` containing:

- `info["native_reward"]` — the unmodified NASim reward. **Accumulate this for
  evaluation, not the shaped reward** (see §6).
- `info["reward_breakdown"]` — per-term attribution; `.as_row()` gives a dict ready for
  logging.
- `info["attack_event"]` — the full `AttackEvent`.

## 4. SB3 defaults as the sanity baseline

Start from `PPO("MlpPolicy", env)` unchanged — the work plan asks for defaults as the
baseline, and tuning before the baseline learns is how you end up debugging two things at
once. SB3's PPO defaults are `n_steps=2048, batch_size=64, n_epochs=10, gamma=0.99,
gae_lambda=0.95, clip_range=0.2, ent_coef=0.0, learning_rate=3e-4`.

Two suggestions worth considering, and being able to justify:

- **`ent_coef`** — the default is `0.0`. With 120 actions and most of them invalid early
  on, a small value such as `0.01` keeps exploration alive. If the agent collapses to one
  action, this is the first knob to reach for.
- **`DummyVecEnv`, not `SubprocVecEnv`** — subprocess workers make seeding and
  reproducibility much harder to reason about. Parallelise across *runs* at the process
  level instead.

Note `clip_range` is PPO's **policy-ratio** clip, not reward clipping. Do not conflate
the two in the write-up; examiners notice.

## 5. Seeding checklist — reproducibility is graded

Set every one of these, or runs will not reproduce:

```
PYTHONHASHSEED       (env var, before Python starts; the Dockerfile sets it to 0)
random.seed(seed)
numpy.random.seed(seed)
torch.manual_seed(seed)
torch.set_num_threads(1)      # multi-thread float reduction order is nondeterministic
PPO(..., seed=seed)
env.reset(seed=seed)
make_env(..., topology_seed=...)   # separate from the training seed - see below
```

**Keep the topology seed separate from the training seed.** They answer different
questions. Fixing the topology seed and varying the training seed measures *training
variance on one network*; varying the topology seed measures *generalisation across
networks*, which is your stated research goal. Decide which you are reporting and say so
explicitly.

## 6. Evaluate on native reward, not shaped reward

The shaped and sparse arms optimise different objectives, so their raw returns are
measured with different rulers and cannot be compared — the shaped arm would "win"
trivially on its own scale.

**Every arm must be evaluated under `info["native_reward"]`.** Accumulate that for
`EpisodeRecord.native_reward` and report it. Shaping is a training-time device; the
native reward is the actual task. This single decision is what makes the ablation
well-posed, and it is worth a sentence in the methodology.

## 7. What to log

Call `EpisodeLogger.log_episode()` once per finished episode:

```python
with EpisodeLogger.start(
    name=f"{reward_config.mode}-seed{seed}",
    reward_mode=str(reward_config.mode),
    config_hash=...,                                  # hash of the reward config
    topology_config_hash=topology_config.config_hash(),
    cve_manifest_sha256=digest(catalogue),
    seed_set=[seed],
) as logger:
    ...
    logger.log_episode(EpisodeRecord(
        seed=seed, topology_seed=topology_seed, episode_idx=i,
        total_reward=shaped_sum, native_reward=native_sum,
        length=steps, terminal_state="goal" if terminated else "step_limit",
        goal_reached=terminated, exploited_hosts=[...],
        mean_cvss_exploited=...,
        steps=[StepRecord(...), ...],
    ))
```

Checklist:

- [ ] seed and topology seed on every episode
- [ ] `topology_config_hash` and `cve_manifest_sha256` on the experiment row — these are
      what prove the two ablation arms were comparable
- [ ] native reward **and** shaped reward per episode
- [ ] `goal_reached` and episode length (the success-rate and steps-to-goal metrics)
- [ ] exploited hosts and their CVSS scores (the high-value-target metric)
- [ ] checkpoints written under `runs/` — gitignored, weights gated by default (§10)

## 8. Suggested order of work

1. Wire PPO to one wrapped env, `mode: sparse`, seed 42, **50k steps**. Just make it run.
2. Time it. Extrapolate to 2 arms × 10 seeds before committing to the full grid — this
   is CPU-only on 16GB.
3. Check the sparse baseline actually learns: success rate should beat random. **If it
   does not, stop and fix that before touching the shaped reward.** Debugging a reward
   against a broken baseline confounds everything.
4. Attach the logger; confirm rows land in Postgres.
5. Only then run seeds 42–51 for both arms.

Random-policy reference points from `make rollout` (seed 42, shaped): episodes reach the
goal in roughly 240–830 steps with strongly negative return. A trained policy should be
clearly shorter and higher. Use this as your "better than random" bar.

## 9. Things that will bite

- **`train.py` must not import `tools/fetch_nvd.py`.** That is the online path; training
  is offline by design.
- **The step tuple is 5 values**, `(obs, reward, terminated, truncated, info)`. NASim's
  `truncated` means the step limit was hit, and `terminated` means the goal was reached —
  do not treat them as interchangeable when computing success rate.
- **Do not recompute the native reward yourself.** Read `info["native_reward"]`.
- **Episode boundaries**: call `wrapper.reset()` between episodes or the anti-farming
  state carries over and shaping silently stops paying.
