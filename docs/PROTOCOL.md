# Evaluation protocol

This document fixes the hypotheses, conditions, metrics and analysis before the main runs. It was written after the environments, experts and a single pilot run existed (insertion, seed 0, two conditions: `state` and `state+proprio+force`, used only to check that the pipeline works and that the tasks are neither trivial nor impossible). Nothing below was changed after looking at the main results. Anything decided later is marked as such in the report.

## 1. Question

When does force or proprioceptive history improve learned manipulation compared with vision alone, and how does its value change under contact uncertainty?

A second question, about representation: is force valuable mainly as a direct control input, or because it helps build a predictive state representation of imminent contact?

## 2. Hypotheses

* **H1 (regime dependence).** Force/proprioceptive inputs add little when the target pose is well estimated and the geometry is forgiving, and add a lot when the visual estimate is biased or unavailable (calibration error, occlusion) or when contact dynamics shift (friction, stiffness).
* **H2 (how, not only whether).** Force-aware policies succeed with lower peak force and smaller force overshoot after first contact. A force-blind policy that still succeeds does so by pushing harder.
* **H3 (actuator as sensor).** Joint efforts (motor-current based, noisy, with an unknown torque-constant error and joint friction) recover a large part of the benefit of a wrist force/torque sensor, and the gap grows with joint friction.
* **H4 (predictive representation).** Training the policy to predict future contact from its inputs (labels derived from the measured wrist force) improves policies that do not receive force as an input. If such a policy approaches the force-input policy, force acts mainly through representation; if not, it acts mainly as a control input.

H4 is the most uncertain; a null result is a result.

## 3. Conditions

All conditions share one network (temporal Transformer, 2 layers, width 64, 16-step history = 0.8 s at 20 Hz), one optimiser (AdamW, lr 3e-4, cosine schedule), batch 128 and 2000 gradient updates. Absent modalities are zeroed before the input projection, so the parameter count is identical across conditions (checked in `tests/test_model.py`).

| name | inputs | auxiliary loss |
|---|---|---|
| `state` | camera estimate of the target, validity flag, tool position | none |
| `state+proprio` | + joint velocities and joint efforts | none |
| `state+force` | + wrist wrench and peak force in the last interval | none |
| `state+proprio+force` | both | none |
| `state+proprio+aux` | state + proprio | future-contact prediction, labels from measured wrist force |
| `state+proprio+force+aux` | all | same |
| `privileged-aux` | all | future true contact and future task phase from the simulator |

The scripted expert is evaluated on the same episodes as a reference point.

## 4. Data

* Demonstrations come from scripted force-aware experts that see the same noisy observations as the full-sensor policy (not privileged), with DART noise (sigma = 0.15 of the action range) on executed actions and clean labels.
* Only successful demonstrations are kept. Counts of rejected attempts are recorded in `data/<task>/episodes.json`.
* Insertion: 400 demonstrations, press: 300, surface: 300. 40 episodes per task are held out for validation loss. For seed *s*, the training subset is a seed-specific permutation of the rest, identical across conditions.

## 5. Evaluation sets

* **In-distribution (`id`)**: 100 episodes, nuisances from the training ranges.
* **Held-out shifts (`heldout:<cond>`)**: 50 episodes each (40 for surface), one physical cause moved outside the training range, everything else from training ranges. Conditions per task are listed in `cambench/evaluation/suite.py`.
* **Sweeps (`sweep:<factor>=<level>`)**: one factor pinned at 6 levels, 30 episodes per level (20 for surface).
* Episode seeds are disjoint from demonstration seeds and identical across conditions and training seeds (common random numbers), so all comparisons are paired.

## 6. Metrics

Primary: **safe success** = task success and peak wrist force at or below the task's damage limit (insertion 25 N, press 35 N, surface 30 N).

Secondary: raw success; peak force (1 kHz, noise-free); contact impulse (integral of |F| above 1 N); force overshoot (peak in the 0.25 s after first contact minus the median over the following 0.5 s); time to success; failure mode (taxonomy per task, defined in each environment's `_failure_mode`).

## 7. Analysis

* Point estimates are means over seeds of per-seed means.
* 95 % intervals come from a hierarchical bootstrap: resample training seeds, then resample the shared episode set.
* Differences between conditions are computed per (seed, episode) pair and bootstrapped the same way. We call something an effect only if the interval excludes zero, and we report effect sizes, not only significance.
* Held-out results are reported per shift and as the mean over shifts. In-distribution and held-out are never pooled into a single headline number.

## 8. Known threats to validity (decided in advance)

* The experts use force. A force-blind student imitating a force-using teacher is at a disadvantage that partly reflects the data, not only the sensor. This is how teleoperated demonstrations work too (the operator feels contact), but it means the comparison is "what a policy can extract from these demonstrations", not "the best possible force-blind controller".
* Tool position history is available to every condition. With an impedance controller, a stalled position while the policy commands motion is itself weak evidence of contact. This is real information a robot always has, so it stays in, but it means `state` is not force-free in a strict sense.
* "Vision" is a simulated pose estimator, not pixels. This isolates the information content of the visual channel from representation learning on images.
