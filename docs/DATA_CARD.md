# Data card: scripted contact-rich demonstrations

## Summary

| | insertion | press | surface |
|---|---|---|---|
| episodes kept | 400 | 300 | 300 |
| attempts | 408 | 304 | 542 |
| rejected (reason) | off target 7, misaligned 1 | unstable 2, over-press 1, under-press 1 | under-force 229, over-force 13 |
| steps (20 Hz) | 20,309 | 22,207 | 58,299 |
| mean episode length | 2.5 s | 3.7 s | 9.7 s |

Numbers are from `data/<task>/episodes.json` for the committed configuration (`configs/main.yaml`, collection seed 0).

The surface task has a high rejection rate because its success criterion (force inside a 4 to 12 N band for 75 % of the trace) is sensitive to the injected action noise. The rejected episodes are not stored. This filter selects demonstrations where the noise happened to be benign, which is a bias worth knowing about when reading the surface results.

## How the data was produced

* **Simulator**: MuJoCo 3.x, 1 kHz physics (2 ms for the surface task), 20 Hz control.
* **Robot**: 3-DoF Cartesian robot, joint impedance (kp 2500 N/m, kv 110 N s/m per axis), set-point velocity actions in [-1, 1] (scaled by 0.1 m/s), set-point lead clamped to 2 cm.
* **Demonstrator**: scripted state machines (`cambench/experts/scripted.py`) that read the same noisy observations the full-sensor policy reads. They are not given the true target pose, stiffness or surface shape.
* **Noise injection**: DART. Executed action = label + N(0, 0.15^2) per axis, clipped. Stored label = clean expert action.
* **Nuisances**: drawn per episode from the training ranges in `cambench/envs/nuisance.py` (`TRAIN`). Values are stored per episode.
* **Filtering**: only successful episodes are kept.

## Files

```
data/<task>/demos.npz       all steps of all kept episodes, stacked
data/<task>/episodes.json   collection settings, rejection counts, per-episode seed, nuisances, outcome metrics
```

## Per-step fields

Policy inputs are noisy and use only what a real robot could measure. Everything else is simulator ground truth, used for labels and analysis, never as an input. The schema is defined (and validated on write) in `cambench/data/schema.py`.

| field | shape | unit | policy input | description |
|---|---|---|---|---|
| `obs_state` | 7 (10 for surface) | m, - | yes | camera estimate of the target minus tool position, task features (surface: estimated height under tool, estimated slope, distance to goal), estimate-valid flag, tool position relative to start |
| `obs_proprio` | 6 | m/s, N | yes | joint velocities, measured joint efforts |
| `obs_force` | 7 | N, N m | yes | tared wrist force and torque, peak force over the last 50 ms |
| `action` | 3 | - | label | expert set-point velocity / vmax |
| `action_exec` | 3 | - | no | executed action (label + DART noise) |
| `tool_pos`, `tool_vel`, `joint_pos`, `setpoint` | 3 | m, m/s | no | robot state |
| `target_true`, `target_est`, `est_valid` | 3, 3, 1 | m | no | target point and its camera estimate |
| `wrench_true`, `wrench_meas` | 6 | N, N m | no | noise-free and measured external wrench |
| `peak_force` | 1 | N | no | true peak force over the last control interval (1 kHz) |
| `contact`, `contact_pos`, `contact_normal_force` | 1, 3, 1 | -, m, N | no | true contact flag, mean contact point (NaN if none), summed normal force |
| `phase` | 1 | - | no | task phase (insertion: free, align, insert, done; press: free, touch, pressing, in tolerance; surface: free, in band, off band, lost contact) |
| `episode_id`, `t` | 1 | - | no | indexing |

## Sensor models

| channel | model |
|---|---|
| camera target estimate | true point + fixed per-episode calibration offset (random direction, magnitude from `calib_offset_mm`) + per-frame Gaussian noise. Near the target the tool occludes the view: with probability `occlusion` the frame is dropped (last value held, valid flag 0), otherwise the noise is scaled by 1 + 4 x occlusion. |
| joint effort | actuator force x (1 + torque-constant error) + Gaussian noise. Real joint Coulomb friction (`joint_friction_n`) acts in the simulation and is not reported, so effort-based force estimates are biased by it. |
| wrist F/T | 6-axis sensor between the robot and tool, tared after settling, Gaussian noise (`ft_noise_n` on force, 1 % of that on torque in N m). Includes tool inertia, like a real sensor. |

## Intended use and limits

For comparing what different sensing and representation choices can extract from the same demonstrations, under controlled shifts. It is not a model of any specific robot or product, and not a substitute for real contact data: MuJoCo's soft contacts, a Cartesian robot without rotational compliance, and analytic sensor noise all simplify what happens on hardware.
