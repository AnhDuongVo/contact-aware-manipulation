"""Task 3: trace a surface while holding a normal force.

A spherical tool must slide 24 cm along +x (at about 3 cm/s) over a tilted, wavy surface while
keeping the normal force at 8 N. The camera provides a plane fit (height and
slope, with calibration error) but does not resolve the waviness, so the
height profile under the tool is only observable through contact.

Success: reach the end of the path with the contact force inside [4, 12] N for
at least 75 % of the tracing time, and never above the force limit.

Randomised per episode: surface height, slope, waviness (amplitude,
wavelength, phase), friction, contact stiffness, plus shared nuisances.
"""
from __future__ import annotations

from typing import Dict

import numpy as np

from .base import ContactEnv, SENSOR_XML, actuator_xml, robot_xml

TIP_R = 0.008
TOOL_HALF_LEN = 0.02
HF_HALF_X, HF_HALF_Y, HF_ELEV = 0.20, 0.05, 0.06
NCOL, NROW = 400, 20
PATH_HALF = 0.12
F_DES, F_LO, F_HI = 8.0, 4.0, 12.0
BAND_FRAC = 0.75


class SurfaceEnv(ContactEnv):
    name = "surface"
    max_steps = 240   # 12 s
    force_limit = 30.0
    contact_timeconst = 0.02
    sim_dt = 0.002    # softer contact allows a 2 ms physics step
    PHASES = ("free", "in_band", "off_band", "lost_contact")

    def _calib_z_scale(self) -> float:
        return 1.0

    def _sample_task(self):
        r, n = self.rng, self.nuisance
        self.center = np.array([r.uniform(-0.02, 0.02), r.uniform(-0.03, 0.03)])
        self.h0 = r.uniform(0.025, 0.035)
        self.slope = np.tan(np.deg2rad(n.surface_slope_deg))
        self.amp = 1e-3 * n.surface_amp_mm
        self.lam = r.uniform(0.06, 0.15)
        self.phi = r.uniform(0, 2 * np.pi)
        # the camera's plane fit: slope error of ~1 degree
        self.slope_est = self.slope + np.tan(np.deg2rad(r.normal(scale=1.0)))
        self.xs = self.center[0] - PATH_HALF
        self.xg = self.center[0] + PATH_HALF
        self.ys = self.center[1]
        start_xy = np.array([self.xs, self.ys]) + r.uniform(-0.02, 0.02, size=2)
        self.start_pos = np.array([*start_xy, self.height(self.xs) + r.uniform(0.04, 0.06)])

    def height(self, x) -> np.ndarray:
        u = np.asarray(x) - self.center[0]
        return self.h0 + self.slope * u + self.amp * np.sin(2 * np.pi * u / self.lam + self.phi)

    def _build_xml(self) -> str:
        n = self.nuisance
        tool = (
            f'<geom name="stylus" type="cylinder" size="0.004 {TOOL_HALF_LEN}" pos="0 0 {-TOOL_HALF_LEN}" '
            f'mass="0.05" contype="0" conaffinity="0" rgba="0.85 0.35 0.2 1"/>'
            f'<geom name="ball" type="sphere" size="{TIP_R}" pos="0 0 {-2 * TOOL_HALF_LEN - TIP_R}" '
            f'mass="{self.robot.tool_mass - 0.05}" friction="{n.friction} 0.005 0.0001" '
            f'contype="3" conaffinity="3" rgba="0.85 0.35 0.2 1"/>'
            f'<site name="tip" pos="0 0 {-2 * TOOL_HALF_LEN - 2 * TIP_R}" size="0.002" rgba="1 1 0 1"/>'
        )
        carriage = self.start_pos + np.array([0, 0, 2 * TOOL_HALF_LEN + 2 * TIP_R])
        cx, cy = self.center
        return (
            self.header()
            + f"""
  <asset>
    <hfield name="surf" nrow="{NROW}" ncol="{NCOL}" size="{HF_HALF_X} {HF_HALF_Y} {HF_ELEV} 0.005"/>
  </asset>
  <worldbody>
    <light pos="0.2 -0.3 0.6" dir="-0.3 0.5 -1" diffuse="0.8 0.8 0.8"/>
    <light pos="-0.3 0.2 0.6" dir="0.4 -0.3 -1" diffuse="0.4 0.4 0.4"/>
    <camera name="side" pos="{cx + 0.05} {cy - 0.38} 0.16" xyaxes="1 0.05 0 -0.02 0.35 0.94"/>
    <geom name="table" type="plane" size="0.4 0.4 0.01" material="grid" contype="1" conaffinity="1"/>
    <geom name="surface" type="hfield" hfield="surf" pos="{cx} {cy} 0" margin="0.001" friction="{n.friction} 0.005 0.0001"
          contype="2" conaffinity="2" rgba="0.55 0.62 0.72 1"/>
    {robot_xml(tool, self.robot, n.joint_friction_n, carriage)}
  </worldbody>
  {actuator_xml(self.robot)}
  {SENSOR_XML}
</mujoco>"""
        )

    def _post_compile(self):
        m = self.model
        xs = self.center[0] + np.linspace(-HF_HALF_X, HF_HALF_X, NCOL)
        row = np.clip(self.height(xs) / HF_ELEV, 0.0, 1.0)
        m.hfield_data[:] = np.tile(row, NROW)
        self.tracing = False
        self.trace_forces = []

    # The "target" for perception is the start point of the path on the surface.
    def _true_target(self) -> np.ndarray:
        return np.array([self.xs, self.ys, float(self.height(self.xs))])

    def _state_features(self, est: np.ndarray, tool: np.ndarray) -> np.ndarray:
        # plane fit anchored at the (estimated) start point
        h_under = est[2] + self.slope_est * (tool[0] - est[0])
        return np.array([
            *(est - tool),
            h_under - tool[2],
            self.slope_est,
            self.xg - tool[0],
        ])

    def _phase(self, contact: bool) -> int:
        fz = self._true_wrench()[2]
        if not contact:
            return 3 if self.first_contact_step is not None else 0
        return 1 if F_LO <= fz <= F_HI else 2

    def _update_metrics(self, peak):
        super()._update_metrics(peak)
        if self.first_contact_step is not None and not self.tracing:
            self.tracing = True
        if self.tracing:
            # true normal force at 1 kHz over the last control interval
            self.trace_forces.extend(self.fz_trace[-self.n_sub:])

    def _band_fraction(self) -> float:
        if not self.trace_forces:
            return 0.0
        f = np.asarray(self.trace_forces)
        return float(np.mean((f >= F_LO) & (f <= F_HI)))

    def _reached_goal(self) -> bool:
        return self.tool_pos[0] >= self.xg

    def _check_success(self) -> bool:
        return self._reached_goal() and self._band_fraction() >= BAND_FRAC

    def _check_abort(self) -> bool:
        return self._reached_goal()

    def _task_metrics(self) -> Dict[str, float]:
        f = np.asarray(self.trace_forces) if self.trace_forces else np.array([np.nan])
        return {
            "in_band_frac": self._band_fraction(),
            "force_rmse": float(np.sqrt(np.nanmean((f - F_DES) ** 2))),
            "contact_loss_frac": float(np.nanmean(f < 0.5)),
            "progress": float(np.clip((self.tool_pos[0] - self.xs) / (self.xg - self.xs), 0, 1)),
        }

    def _failure_mode(self, m: Dict[str, float]) -> str:
        if m["success"]:
            return "ok" if m["peak_force"] <= self.force_limit else "excessive_force"
        if not m["made_contact"]:
            return "miss"
        if not self._reached_goal():
            return "incomplete"
        if m["contact_loss_frac"] > 0.1:
            return "lost_contact"
        f = np.asarray(self.trace_forces)
        return "over_force" if np.mean(f > F_HI) > np.mean(f < F_LO) else "under_force"
