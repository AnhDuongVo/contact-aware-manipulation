"""Task 1: square peg-in-hole insertion with a chamfered hole.

Geometry: a 20 mm square peg must be inserted 20 mm deep into a 30 mm deep
hole. The hole has a 3 mm 45-degree chamfer, so a lateral error of up to
roughly ``3 mm + clearance/2`` can be corrected by contact if the controller
complies with the lateral contact force; larger errors land the peg on the
flat top face. The fixture sits on a lateral spring (compliant mount).

Randomised per episode: hole position, clearance, friction, fixture
stiffness, contact stiffness, plus all perception and sensing nuisances.
"""
from __future__ import annotations

from typing import Dict

import numpy as np

from .base import ContactEnv, SENSOR_XML, actuator_xml, robot_xml

PEG_HALF = 0.010      # m
PEG_HALF_LEN = 0.030  # m
FIX_HALF = 0.045      # outer half width of the fixture block
FIX_H = 0.030         # height of fixture = hole depth
CHAMFER = 0.003       # chamfer size
DEPTH_GOAL = 0.020    # required insertion depth


class InsertionEnv(ContactEnv):
    name = "insertion"
    max_steps = 160          # 8 s
    force_limit = 25.0
    contact_timeconst = 0.01

    PHASES = ("free", "align", "insert", "done")

    def _sample_task(self):
        r = self.rng
        self.hole_xy = r.uniform(-0.03, 0.03, size=2)
        start_xy = self.hole_xy + r.uniform(-0.03, 0.03, size=2)
        self.start_pos = np.array([*start_xy, FIX_H + r.uniform(0.04, 0.07)])
        self.a = PEG_HALF + 0.5e-3 * self.nuisance.clearance_mm

    def _build_xml(self) -> str:
        n, a, c, W, H = self.nuisance, self.a, CHAMFER, FIX_HALF, FIX_H
        mu = n.friction
        k = n.fixture_stiffness
        m_fix = 0.5
        damp = 2 * 0.3 * np.sqrt(k * m_fix)
        geoms = []

        def box(name, pos, size, euler=(0, 0, 0)):
            geoms.append(
                f'<geom name="{name}" type="box" pos="{pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}" '
                f'size="{size[0]:.6f} {size[1]:.6f} {size[2]:.6f}" euler="{euler[0]} {euler[1]} {euler[2]}" '
                f'friction="{mu} 0.005 0.0001" contype="2" conaffinity="2" rgba="0.55 0.62 0.72 1" mass="0.05"/>'
            )

        lo = H - c  # top of the lower walls
        for s in (+1, -1):
            # walls normal to x span the full block in y
            box(f"wall_x{s}", (s * (a + W) / 2, 0, lo / 2), ((W - a) / 2, W, lo / 2))
            box(f"top_x{s}", (s * (a + c + W) / 2, 0, lo + c / 2), ((W - a - c) / 2, W, c / 2))
            box(f"cham_x{s}", (s * (a + c), 0, lo), (c / np.sqrt(2), a + c, c / np.sqrt(2)), (0, np.pi / 4, 0))
            # walls normal to y fill the gap between the x walls
            box(f"wall_y{s}", (0, s * (a + W) / 2, lo / 2), (a, (W - a) / 2, lo / 2))
            box(f"top_y{s}", (0, s * (a + c + W) / 2, lo + c / 2), (a + c, (W - a - c) / 2, c / 2))
            box(f"cham_y{s}", (0, s * (a + c), lo), (a + c, c / np.sqrt(2), c / np.sqrt(2)), (np.pi / 4, 0, 0))

        tool = (
            f'<geom name="peg" type="box" size="{PEG_HALF} {PEG_HALF} {PEG_HALF_LEN}" '
            f'pos="0 0 {-PEG_HALF_LEN}" mass="{self.robot.tool_mass}" friction="0.1 0.005 0.0001" '
            f'contype="3" conaffinity="3" rgba="0.85 0.35 0.2 1"/>'
            f'<site name="tip" pos="0 0 {-2 * PEG_HALF_LEN}" size="0.002" rgba="1 1 0 1"/>'
        )
        carriage = self.start_pos + np.array([0, 0, 2 * PEG_HALF_LEN])
        return (
            self.header()
            + f"""
  <worldbody>
    <light pos="0.2 -0.3 0.6" dir="-0.3 0.5 -1" diffuse="0.8 0.8 0.8"/>
    <light pos="-0.3 0.2 0.6" dir="0.4 -0.3 -1" diffuse="0.4 0.4 0.4"/>
    <camera name="side" pos="{self.hole_xy[0] + 0.16} {self.hole_xy[1] - 0.20} 0.15" xyaxes="0.78 0.62 0 -0.2 0.25 0.95"/>
    <camera name="close" pos="{self.hole_xy[0] + 0.08} {self.hole_xy[1] - 0.09} 0.08" xyaxes="0.75 0.66 0 -0.3 0.34 0.9"/>
    <geom name="table" type="plane" size="0.4 0.4 0.01" material="grid" contype="1" conaffinity="1" friction="{mu} 0.005 0.0001"/>
    <body name="fixture" pos="{self.hole_xy[0]} {self.hole_xy[1]} 0">
      <joint name="fix_x" type="slide" axis="1 0 0" stiffness="{k}" damping="{damp}"/>
      <joint name="fix_y" type="slide" axis="0 1 0" stiffness="{k}" damping="{damp}"/>
      {''.join(geoms)}
    </body>
    {robot_xml(tool, self.robot, n.joint_friction_n, carriage)}
  </worldbody>
  {actuator_xml(self.robot)}
  {SENSOR_XML}
</mujoco>"""
        )

    def _post_compile(self):
        m = self.model
        self._fix_q = np.array([m.joint("fix_x").qposadr[0], m.joint("fix_y").qposadr[0]])
        # The carriage body starts at the requested pose, so joint zero = start.

    # ------------------------------------------------------------- geometry
    def hole_center(self) -> np.ndarray:
        xy = self.hole_xy + self.data.qpos[self._fix_q]
        return np.array([xy[0], xy[1], FIX_H])

    def _true_target(self) -> np.ndarray:
        return self.hole_center()

    def depth(self) -> float:
        """How far the peg tip is below the top face of the fixture."""
        return FIX_H - self.tool_pos[2]

    def capture_range(self) -> float:
        """Largest lateral offset (per axis) at which the peg still lands on
        the chamfer rather than on the flat top face."""
        return self.a - PEG_HALF + CHAMFER

    def lateral_error(self) -> float:
        return float(np.max(np.abs(self.tool_pos[:2] - self.hole_center()[:2])))

    # --------------------------------------------------------------- labels
    def _phase(self, contact: bool) -> int:
        if self.depth() >= DEPTH_GOAL:
            return 3
        if self.depth() > CHAMFER + 0.0005:
            return 2
        return 1 if contact else 0

    def _check_success(self) -> bool:
        return self.depth() >= DEPTH_GOAL and self.lateral_error() < self.a - PEG_HALF + 1e-4

    def _init_metrics(self):
        super()._init_metrics()
        self.max_depth = 0.0
        self.depth_hist = []

    def _update_metrics(self, peak):
        super()._update_metrics(peak)
        dep = self.depth()
        self.max_depth = max(self.max_depth, dep)
        self.depth_hist.append(dep)

    def _task_metrics(self) -> Dict[str, float]:
        return {
            "final_depth_mm": 1e3 * self.depth(),
            "max_depth_mm": 1e3 * self.max_depth,
            "final_lateral_err_mm": 1e3 * self.lateral_error(),
        }

    def _failure_mode(self, m: Dict[str, float]) -> str:
        if m["success"]:
            return "ok" if m["peak_force"] <= self.force_limit else "excessive_force"
        if not m["made_contact"]:
            return "miss"
        dep = self.depth()
        if dep > CHAMFER + 0.0005:
            recent = np.asarray(self.depth_hist[-20:])
            moving = recent.max() - recent.min() > 0.002
            return "partial" if moving else "jammed"
        if self.lateral_error() > self.capture_range():
            return "off_target"
        return "misaligned"
