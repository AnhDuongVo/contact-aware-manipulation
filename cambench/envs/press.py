"""Task 2: press a spring-loaded button to a target depth.

The button sits on a preloaded spring with stiction (it does not move until
the push exceeds a breakaway force). Success means holding the button at
8 +/- 1.5 mm depth for 0.5 s without exceeding the force limit or hitting
12 mm (over-press). The button's height is only known through the camera
estimate, so a force-blind controller can only measure depth relative to
where it *thinks* the surface is; touch gives the true reference.

Randomised per episode: button position and height, spring stiffness,
breakaway force, plus the shared perception and sensing nuisances.
"""
from __future__ import annotations

from typing import Dict

import numpy as np

from .base import ContactEnv, SENSOR_XML, actuator_xml, robot_xml

CAP_HALF = 0.012
CAP_HALF_H = 0.005
PUSHER_R = 0.006
PUSHER_HALF_LEN = 0.025
TARGET_DEPTH = 0.008
TOL = 0.0015
OVER_PRESS = 0.012
TRAVEL = 0.015
HOLD_STEPS = 10


class PressEnv(ContactEnv):
    name = "press"
    max_steps = 160
    force_limit = 35.0
    contact_timeconst = 0.01
    PHASES = ("free", "touch", "pressing", "in_tolerance")

    def _calib_z_scale(self) -> float:
        return 1.0  # vertical calibration error matters most here

    def _sample_task(self):
        r = self.rng
        self.btn_xy = r.uniform(-0.03, 0.03, size=2)
        self.top0 = r.uniform(0.03, 0.05)  # resting height of the cap top
        start_xy = self.btn_xy + r.uniform(-0.03, 0.03, size=2)
        self.start_pos = np.array([*start_xy, self.top0 + r.uniform(0.04, 0.07)])

    def _build_xml(self) -> str:
        n = self.nuisance
        k = n.button_stiffness
        base_h = self.top0 - 2 * CAP_HALF_H
        tool = (
            f'<geom name="pusher" type="cylinder" size="{PUSHER_R} {PUSHER_HALF_LEN}" '
            f'pos="0 0 {-PUSHER_HALF_LEN}" mass="{self.robot.tool_mass}" friction="{n.friction} 0.005 0.0001" '
            f'contype="7" conaffinity="7" rgba="0.85 0.35 0.2 1"/>'
            f'<site name="tip" pos="0 0 {-2 * PUSHER_HALF_LEN}" size="0.002" rgba="1 1 0 1"/>'
        )
        carriage = self.start_pos + np.array([0, 0, 2 * PUSHER_HALF_LEN])
        bx, by = self.btn_xy
        return (
            self.header()
            + f"""
  <worldbody>
    <light pos="0.2 -0.3 0.6" dir="-0.3 0.5 -1" diffuse="0.8 0.8 0.8"/>
    <light pos="-0.3 0.2 0.6" dir="0.4 -0.3 -1" diffuse="0.4 0.4 0.4"/>
    <camera name="side" pos="{bx + 0.16} {by - 0.20} {self.top0 + 0.12}" xyaxes="0.78 0.62 0 -0.2 0.25 0.95"/>
    <geom name="table" type="plane" size="0.4 0.4 0.01" material="grid" contype="1" conaffinity="1"/>
    <geom name="housing" type="box" pos="{bx} {by} {base_h / 2}" size="0.03 0.03 {base_h / 2}"
          contype="2" conaffinity="2" rgba="0.55 0.62 0.72 1"/>
    <body name="button" pos="{bx} {by} {base_h + CAP_HALF_H + 0.0005}" gravcomp="1">
      <joint name="btn" type="slide" axis="0 0 1" range="{-TRAVEL} 0" stiffness="{k}" springref="0.002"
             damping="3" frictionloss="{n.button_breakaway_n}" armature="0.01"/>
      <geom name="cap" type="box" size="{CAP_HALF} {CAP_HALF} {CAP_HALF_H}" mass="0.02"
            friction="{n.friction} 0.005 0.0001" contype="4" conaffinity="4" rgba="0.2 0.6 0.35 1"/>
    </body>
    {robot_xml(tool, self.robot, n.joint_friction_n, carriage)}
  </worldbody>
  {actuator_xml(self.robot)}
  {SENSOR_XML}
</mujoco>"""
        )

    def _post_compile(self):
        m = self.model
        self._btn_q = m.joint("btn").qposadr[0]
        self._cap = m.geom("cap").id

    def press_depth(self) -> float:
        return float(-self.data.qpos[self._btn_q])

    def _true_target(self) -> np.ndarray:
        p = self.data.geom_xpos[self._cap]
        return np.array([p[0], p[1], p[2] + CAP_HALF_H])

    def _phase(self, contact: bool) -> int:
        dep = self.press_depth()
        if abs(dep - TARGET_DEPTH) <= TOL:
            return 3
        if dep > 0.0005:
            return 2
        return 1 if contact else 0

    def _init_metrics(self):
        super()._init_metrics()
        self.hold = 0
        self.max_press = 0.0
        self.ever_in_tol = False
        self.depth_hist = []

    def _update_metrics(self, peak):
        super()._update_metrics(peak)
        dep = self.press_depth()
        self.depth_hist.append(dep)
        self.max_press = max(self.max_press, dep)
        inside = abs(dep - TARGET_DEPTH) <= TOL
        self.ever_in_tol |= inside
        self.hold = self.hold + 1 if inside else 0

    def _check_success(self) -> bool:
        return self.hold >= HOLD_STEPS and self.max_press < OVER_PRESS

    def _check_abort(self) -> bool:
        return self.max_press >= OVER_PRESS

    def _task_metrics(self) -> Dict[str, float]:
        tail = np.asarray(self.depth_hist[-HOLD_STEPS:])
        return {
            "final_depth_mm": 1e3 * self.press_depth(),
            "max_depth_mm": 1e3 * self.max_press,
            "depth_error_mm": 1e3 * float(np.mean(np.abs(tail - TARGET_DEPTH))) if len(tail) else np.nan,
        }

    def _failure_mode(self, m: Dict[str, float]) -> str:
        if m["success"]:
            return "ok" if m["peak_force"] <= self.force_limit else "excessive_force"
        if not m["made_contact"]:
            return "miss"
        if self.max_press >= OVER_PRESS:
            return "over_press"
        dep = self.press_depth()
        if dep < TARGET_DEPTH - TOL:
            return "under_press"
        if dep > TARGET_DEPTH + TOL:
            return "over_press"
        return "unstable"
