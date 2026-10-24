"""Shared machinery for the contact-rich tasks.

The robot is a 3-DoF Cartesian manipulator (x, y, z prismatic joints) under
joint-space impedance control, carrying a task-specific tool through a 6-axis
force/torque sensor. A Cartesian robot keeps the experiment about *sensing*
rather than about inverse kinematics: joint efforts map one-to-one onto
end-effector forces, so "proprioception" here is the same signal an
actuator-as-sensor approach would use on a real arm, minus the Jacobian.

Control loop
------------
* physics at 1 kHz (``sim_dt``)
* policy at 20 Hz (``control_dt``); one action = desired set-point velocity
* inner loop: MuJoCo position actuators with stiffness ``kp`` and damping ``kv``,
  i.e. a spring-damper between the set-point and the joint. The set-point is
  clamped to stay within ``windup`` metres of the current position, which caps
  the force a policy can apply by "pushing harder" at ``kp * windup``.

Observation groups
------------------
Every task returns three groups so that models can be ablated by masking:

* ``state``   target estimate from the simulated camera (relative to the tool),
              a validity flag and the tool position (relative to its start).
              Previous actions are deliberately not part of any input: with
              set-point control, integrated actions minus measured position
              would leak contact force into the "vision-only" condition.
* ``proprio`` joint velocities and joint efforts (motor-current based, noisy,
              with an unknown torque-constant error).
* ``force``   tared 6-axis wrist wrench (noisy) and the peak force magnitude
              seen during the last control interval.

Privileged quantities (true contact flag, contact location, true wrench,
task phase) are returned in ``info`` and stored in the dataset for labels and
analysis. They are never part of a policy input.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import mujoco
import numpy as np

from .nuisance import Nuisance

GROUPS = ("state", "proprio", "force")


@dataclass
class RobotConfig:
    kp: tuple = (2500.0, 2500.0, 2500.0)   # N/m
    kv: tuple = (110.0, 110.0, 110.0)      # N s/m
    vmax: float = 0.10                     # m/s, max set-point speed
    windup: float = 0.02                   # m, max set-point lead
    carriage_mass: float = 1.0
    tool_mass: float = 0.2
    force_limit: float = 80.0              # actuator saturation [N]


@dataclass
class StepRecord:
    """Everything logged at one control step (see docs/DATA_CARD.md)."""
    obs: Dict[str, np.ndarray]
    action: np.ndarray
    info: Dict[str, object] = field(default_factory=dict)


class Perception:
    """A stand-in for a camera-based pose estimator.

    estimate = true target + fixed calibration offset + per-frame noise.
    Near the target the tool occludes the view: with probability ``occlusion``
    per frame the estimate is dropped (the last value is held and the validity
    flag goes to 0) and otherwise its noise grows with the occlusion level.
    """

    OCCLUSION_RADIUS = 0.03   # m, xy distance at which the tool starts to occlude
    OCCLUSION_HEIGHT = 0.04   # m, above the target

    def __init__(self, nuisance: Nuisance, rng: np.random.Generator, z_scale: float = 0.5):
        self.n = nuisance
        self.rng = rng
        d = rng.normal(size=3)
        d[2] *= z_scale
        d /= np.linalg.norm(d) + 1e-9
        self.offset = d * nuisance.calib_offset_mm * 1e-3
        self.last: Optional[np.ndarray] = None

    def __call__(self, true_target: np.ndarray, tool_pos: np.ndarray):
        occluded = (
            np.linalg.norm(tool_pos[:2] - true_target[:2]) < self.OCCLUSION_RADIUS
            and tool_pos[2] < true_target[2] + self.OCCLUSION_HEIGHT
        )
        sigma = self.n.vis_noise_mm * 1e-3
        if occluded:
            if self.last is not None and self.rng.random() < self.n.occlusion:
                return self.last.copy(), 0.0
            sigma *= 1.0 + 4.0 * self.n.occlusion
        est = true_target + self.offset + self.rng.normal(scale=sigma, size=3)
        self.last = est
        return est.copy(), 1.0


def robot_xml(tool_xml: str, robot: RobotConfig, joint_friction: float, start: np.ndarray) -> str:
    """MJCF for the Cartesian robot. ``tool_xml`` goes inside the tool body,
    whose origin is the force/torque sensor frame."""
    return f"""
    <body name="carriage" pos="{start[0]} {start[1]} {start[2]}" gravcomp="1">
      <joint name="jx" type="slide" axis="1 0 0" range="-0.3 0.3" armature="0.02" frictionloss="{joint_friction}"/>
      <joint name="jy" type="slide" axis="0 1 0" range="-0.3 0.3" armature="0.02" frictionloss="{joint_friction}"/>
      <joint name="jz" type="slide" axis="0 0 1" range="-0.25 0.25" armature="0.02" frictionloss="{joint_friction}"/>
      <geom type="box" size="0.02 0.02 0.02" pos="0 0 0.03" mass="{robot.carriage_mass}"
            contype="0" conaffinity="0" rgba="0.25 0.27 0.3 1"/>
      <geom type="cylinder" size="0.012 0.04" pos="0 0 0.09" mass="0.001"
            contype="0" conaffinity="0" rgba="0.25 0.27 0.3 1"/>
      <body name="tool" pos="0 0 0" gravcomp="1">
        <site name="ft" pos="0 0 0" size="0.004" rgba="1 0.6 0 1"/>
        <geom type="cylinder" size="0.016 0.004" pos="0 0 0.004" mass="0.001"
              contype="0" conaffinity="0" rgba="0.9 0.55 0.1 1"/>
        {tool_xml}
      </body>
    </body>"""


def actuator_xml(robot: RobotConfig) -> str:
    rows = []
    for j, kp, kv in zip(("jx", "jy", "jz"), robot.kp, robot.kv):
        rows.append(
            f'<position name="a_{j}" joint="{j}" kp="{kp}" kv="{kv}" '
            f'forcelimited="true" forcerange="-{robot.force_limit} {robot.force_limit}" '
            f'ctrllimited="false"/>'
        )
    return "<actuator>" + "".join(rows) + "</actuator>"


SENSOR_XML = """
  <sensor>
    <force name="ft_force" site="ft"/>
    <torque name="ft_torque" site="ft"/>
  </sensor>"""


def scene_header(name: str, contact_stiffness: float = 1.0, timeconst: float = 0.01,
                 timestep: float = 0.001) -> str:
    # MuJoCo soft-contact time constant; contact_stiffness < 1 makes contacts
    # softer (e.g. a plastic part instead of machined metal).
    timeconst = timeconst / max(contact_stiffness, 1e-3) ** 0.5
    return f"""
<mujoco model="{name}">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="{timestep}" integrator="implicitfast" cone="elliptic" impratio="5"/>
  <visual>
    <global offwidth="640" offheight="480"/>
    <quality shadowsize="2048"/>
  </visual>
  <default>
    <geom solref="{timeconst:.5f} 1" solimp="0.95 0.99 0.0005 0.5 2" condim="3"/>
  </default>
  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.92 0.92 0.92" rgb2="0.85 0.85 0.87" width="256" height="256"/>
    <material name="grid" texture="grid" texrepeat="8 8"/>
    <texture name="sky" type="skybox" builtin="gradient" rgb1="0.98 0.98 0.97" rgb2="0.80 0.83 0.88" width="256" height="256"/>
  </asset>"""


class ContactEnv:
    """Base class. Subclasses define the scene, the expert-relevant target,
    termination, metrics and phase labels."""

    name = "base"
    control_dt = 0.05
    sim_dt = 0.001
    max_steps = 160
    contact_threshold = 1.0  # N, force magnitude that counts as "in contact"
    contact_timeconst = 0.01
    force_limit = 25.0       # N, damage threshold used for safe success

    def __init__(self, robot: Optional[RobotConfig] = None):
        self.robot = robot or RobotConfig()
        self.n_sub = int(round(self.control_dt / self.sim_dt))
        self.model: Optional[mujoco.MjModel] = None
        self.data: Optional[mujoco.MjData] = None

    # ------------------------------------------------------------------ API
    def reset(self, nuisance: Nuisance, seed: int):
        self.nuisance = nuisance
        self.rng = np.random.default_rng(seed)
        self.seed = seed
        self._sample_task()
        xml = self._build_xml()
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        self._post_compile()
        m, d = self.model, self.data
        self._tool_body = m.body("tool").id
        self._tool_geoms = {g for g in range(m.ngeom) if m.geom_bodyid[g] == self._tool_body}
        self._qadr = np.array([m.joint(j).qposadr[0] for j in ("jx", "jy", "jz")])
        self._dadr = np.array([m.joint(j).dofadr[0] for j in ("jx", "jy", "jz")])
        self._site = m.site("tip").id
        mujoco.mj_forward(m, d)
        self.setpoint = d.qpos[self._qadr].copy()
        d.ctrl[:] = self.setpoint
        # Let the robot settle, then tare the wrist sensor (as on hardware).
        for _ in range(int(0.2 / self.sim_dt)):
            mujoco.mj_step(m, d)
        self.tare_f = d.sensor("ft_force").data.copy()
        self.tare_t = d.sensor("ft_torque").data.copy()
        self.perception = Perception(nuisance, self.rng, z_scale=self._calib_z_scale())
        self.prev_action = np.zeros(3)
        self.t = 0
        self.done = False
        self.success = False
        self.force_trace: List[float] = []      # true |F| at 1 kHz
        self.fz_trace: List[float] = []         # true vertical force at 1 kHz
        self.first_contact_step: Optional[int] = None
        self._init_metrics()
        return self._observe(peak=0.0)

    def step(self, action: np.ndarray):
        assert not self.done, "episode finished; call reset()"
        m, d, r = self.model, self.data, self.robot
        a = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
        q = d.qpos[self._qadr]
        self.setpoint = self.setpoint + a * r.vmax * self.control_dt
        self.setpoint = np.clip(self.setpoint, q - r.windup, q + r.windup)
        d.ctrl[:] = self.setpoint
        peak = 0.0
        for _ in range(self.n_sub):
            mujoco.mj_step(m, d)
            w = self._true_wrench()
            f = float(np.linalg.norm(w[:3]))
            self.force_trace.append(f)
            self.fz_trace.append(float(w[2]))
            peak = max(peak, f)
        self.t += 1
        self.prev_action = a
        obs = self._observe(peak)
        if self.first_contact_step is None and self._last_info["contact"]:
            self.first_contact_step = self.t
        self._update_metrics(peak)
        self.success = self._check_success()
        self.done = self.success or self.t >= self.max_steps or self._check_abort()
        return obs, self.done, self._last_info

    # ------------------------------------------------------------- helpers
    @property
    def tool_pos(self) -> np.ndarray:
        return self.data.site_xpos[self._site].copy()

    def _true_wrench(self) -> np.ndarray:
        """External wrench on the tool (environment -> tool), noise-free."""
        d = self.data
        f = -(d.sensor("ft_force").data - self.tare_f)
        t = -(d.sensor("ft_torque").data - self.tare_t)
        return np.concatenate([f, t])

    def _contacts(self):
        """Privileged contact info: flag, mean location, summed normal force."""
        m, d = self.model, self.data
        pts, fn = [], 0.0
        buf = np.zeros(6)
        for i in range(d.ncon):
            c = d.contact[i]
            if (c.geom1 in self._tool_geoms) != (c.geom2 in self._tool_geoms):
                mujoco.mj_contactForce(m, d, i, buf)
                if buf[0] > 1e-3:
                    pts.append(c.pos.copy())
                    fn += buf[0]
        if pts:
            return True, np.mean(pts, axis=0), fn
        return False, np.full(3, np.nan), 0.0

    def _observe(self, peak: float) -> Dict[str, np.ndarray]:
        d, n = self.data, self.nuisance
        tool = self.tool_pos
        target = self._true_target()
        est, valid = self.perception(target, tool)
        effort = d.actuator_force.copy() * (1.0 + n.effort_gain_err)
        effort += self.rng.normal(scale=n.effort_noise_n, size=3)
        wrench_true = self._true_wrench()
        wrench = wrench_true.copy()
        wrench[:3] += self.rng.normal(scale=n.ft_noise_n, size=3)
        wrench[3:] += self.rng.normal(scale=n.ft_noise_n * 0.01, size=3)
        peak_noisy = max(0.0, peak + self.rng.normal(scale=n.ft_noise_n))
        obs = {
            "state": np.concatenate([
                self._state_features(est, tool),
                [valid],
                tool - self.start_pos,
            ]).astype(np.float32),
            "proprio": np.concatenate([
                d.qvel[self._dadr],
                effort,
            ]).astype(np.float32),
            "force": np.concatenate([
                wrench,
                [peak_noisy],
            ]).astype(np.float32),
        }
        contact, cpos, fn = self._contacts()
        self._last_info = {
            "t": self.t,
            "tool_pos": tool,
            "tool_vel": d.qvel[self._dadr].copy(),
            "joint_pos": d.qpos[self._qadr].copy(),
            "setpoint": self.setpoint.copy(),
            "target_true": target,
            "target_est": est,
            "est_valid": valid,
            "wrench_true": wrench_true,
            "wrench_meas": wrench,
            "peak_force": peak,
            "contact": contact,
            "contact_pos": cpos,
            "contact_normal_force": fn,
            "phase": self._phase(contact),
        }
        return obs

    # ------------------------------------------------- metrics (shared part)
    def _init_metrics(self):
        self.m_peak = 0.0
        self.m_impulse = 0.0

    def _update_metrics(self, peak: float):
        self.m_peak = max(self.m_peak, peak)
        tail = np.asarray(self.force_trace[-self.n_sub:])
        self.m_impulse += float(np.sum(tail[tail > self.contact_threshold]) * self.sim_dt)

    def episode_metrics(self) -> Dict[str, float]:
        trace = np.asarray(self.force_trace)
        overshoot = np.nan
        if self.first_contact_step is not None:
            # Force overshoot: peak force in the 0.25 s after first contact
            # minus the median force over the following 0.5 s (the level the
            # controller settles to). Positive values mean a spike.
            i0 = (self.first_contact_step - 1) * self.n_sub
            w1, w2 = int(0.25 / self.sim_dt), int(0.75 / self.sim_dt)
            win = trace[i0:i0 + w1]
            after = trace[i0 + w1:i0 + w2]
            if len(win):
                ref = np.median(after) if len(after) > w1 // 5 else np.median(win)
                overshoot = float(win.max() - ref)
        out = {
            "success": float(self.success),
            "safe_success": float(self.success and self.m_peak <= self.force_limit),
            "peak_force": float(self.m_peak),
            "impulse": float(self.m_impulse),
            "force_overshoot": overshoot,
            "steps": float(self.t),
            "time_to_success": float(self.t * self.control_dt) if self.success else np.nan,
            "made_contact": float(self.first_contact_step is not None),
        }
        out.update(self._task_metrics())
        out["failure"] = self._failure_mode(out)
        out["sim_dt"] = self.sim_dt
        return out

    # ------------------------------------------------------ subclass hooks
    def _calib_z_scale(self) -> float:
        return 0.5

    def _sample_task(self):
        raise NotImplementedError

    def _build_xml(self) -> str:
        raise NotImplementedError

    def _post_compile(self):
        pass

    def _true_target(self) -> np.ndarray:
        raise NotImplementedError

    def _state_features(self, est: np.ndarray, tool: np.ndarray) -> np.ndarray:
        return est - tool

    def _phase(self, contact: bool) -> int:
        raise NotImplementedError

    def _check_success(self) -> bool:
        raise NotImplementedError

    def _check_abort(self) -> bool:
        return False

    def _task_metrics(self) -> Dict[str, float]:
        return {}

    def _failure_mode(self, m: Dict[str, float]) -> str:
        raise NotImplementedError

    # --------------------------------------------------------- rendering
    camera = "side"

    def header(self) -> str:
        return scene_header(self.name, self.nuisance.contact_stiffness, self.contact_timeconst, self.sim_dt)
