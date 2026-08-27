# Reward Specification

This document describes the implemented Essential fixed-topology experiment.
The reward selected here is the value PPO optimises; native NASim reward is
always retained separately for comparable evaluation.

## Sparse condition

For event `e`:

```text
R_sparse(e) = 1.0  if e is a successful access-gaining action on a crown jewel
              0.0  otherwise
```

The sparse trigger is identical to the shaped crown-jewel trigger. Failures,
scans, ordinary exploits and privilege escalation receive zero.

## Shaped condition

```text
failed action:
    R_shaped(e) = -5.0

successful action:
    R_shaped(e) = CVE(e) + Tactic(e) + Goal(e)
```

| Component | Coefficient/value | Fires when | Repeatable? |
|---|---:|---|---|
| CVE | `2.5 × severity_weight(CVSS)` | a payable exploit/privesc has a CVE score | only on access improvement |
| Recon tactic | `+1.0` | an informative service/OS/subnet/process scan | once per action/target |
| Exploit tactic | `+0.5` | an exploit raises access | only on access improvement |
| Privilege escalation tactic | `+2.0` | privesc raises access | only on access improvement |
| Crown jewel | `+100.0` | successful access gain on a sensitive host | simulator prevents repeated value payment |
| Failed action | `-5.0` | action fails | every failure |

Severity weighting uses the configured power mapping (`gamma=2.0`,
`w_min=0.25`, `w_max=1.0`). The CVE assignment is a deterministic,
topology-seeded stratified assignment from the frozen 16-record catalogue.

## Anti-farming rule

Scans pay only when they disclose new information and once per action/target.
Exploit and privilege-escalation shaping pays only when access rises
`NONE → USER → ROOT`. The engine analytically checks that the total possible
shaping is less than half the total crown-jewel value.

## Native reward

Native mode returns NASim reward verbatim. Every sparse and shaped transition
also records native reward. Sparse and shaped returns are on different scales,
so the experiment compares frozen policies using native reward and task metrics,
not their training objective totals.

## Representable ATT&CK concepts

The simulator maps service scan (`T1046`), OS scan (`T1082`), subnet scan
(`T1018`), process scan (`T1057`), remote-service exploit (`T1210`) and
privilege escalation (`T1068`). NASim has no exfiltration action; the configured
`exfil` bonus is unreachable and must be reported as zero, not claimed as
coverage.

## Limitations

- CVE-to-generated-service binding is synthetic even though CVE metadata is real.
- Reward history is episode-stateful because anti-farming is required.
- High reward is not proof of a meaningful path; evaluation traces require
  separate trajectory validation.
- Sparse and shaped differ deliberately in the reward mode, while resolved
  non-mode coefficients remain equal but unused in sparse mode.
