"""Scripted demonstrators.

The experts see exactly the same noisy observations that the full-sensor
policy sees (camera estimate + wrist wrench). They are not privileged: they
do not know the true hole pose, the button stiffness or the surface shape.
This mirrors teleoperated demonstrations where the operator feels force, and
it means that whatever the experts achieve is, in principle, achievable by a
policy with the same inputs.

Each expert is a small state machine. Actions are set-point velocities
normalised by ``vmax`` (so they lie in [-1, 1]).
"""
from __future__ import annotations

import numpy as np

VMAX = 0.10


def _clip(v: np.ndarray) -> np.ndarray:
    return np.clip(v / VMAX, -1.0, 1.0)


class InsertionExpert:
    """Approach above the estimated hole, descend, then switch to force-guided
    insertion: regulate the vertical contact force and move laterally along the
    measured lateral contact force (admittance), which slides the peg down the
    chamfer into the bore instead of pushing harder."""

    F_CONTACT = 2.0
    F_DES = 6.0          # N, desired vertical contact force while inserting
    K_ADM = 0.006        # m/s per N of lateral force

    def reset(self):
        self.mode = "approach"
        self.est = None
        self.n_contact = 0

    def __call__(self, obs) -> np.ndarray:
        rel = obs["state"][:3].astype(float)     # estimated target - tool
        valid = obs["state"][3] > 0.5
        f = obs["force"][:3].astype(float)
        if valid or self.est is None:
            self.est = rel
        self.n_contact = self.n_contact + 1 if np.linalg.norm(f) > self.F_CONTACT else 0
        v = np.zeros(3)
        if self.mode == "approach":
            goal = rel + np.array([0, 0, 0.012])
            v = 3.0 * goal
            if np.linalg.norm(goal[:2]) < 0.0015 and abs(goal[2]) < 0.004:
                self.mode = "descend"
        if self.mode == "descend":
            v[:2] = 3.0 * rel[:2]
            v[2] = -0.03
            if self.n_contact >= 1:
                self.mode = "insert"
        if self.mode == "insert":
            v[:2] = self.K_ADM * f[:2]
            v[2] = -0.03 * np.clip(1.0 - f[2] / self.F_DES, -1.0, 1.0)
            if rel[2] > 0.024:     # tip is >24 mm below the estimated top face
                v[2] = -0.005
        return _clip(v)


class PressExpert:
    """Find the button surface by touch, then press to the target depth
    measured from the touch point and hold it."""

    F_CONTACT = 1.5
    DEPTH = 0.008

    def reset(self):
        self.mode = "approach"
        self.z_touch = None

    def __call__(self, obs) -> np.ndarray:
        rel = obs["state"][:3].astype(float)
        z = obs["state"][6]                      # tool z relative to start
        f = obs["force"][:3].astype(float)
        v = np.zeros(3)
        if self.mode == "approach":
            goal = rel + np.array([0, 0, 0.012])
            v = 3.0 * goal
            if np.linalg.norm(goal[:2]) < 0.002 and abs(goal[2]) < 0.004:
                self.mode = "descend"
        if self.mode == "descend":
            v[:2] = 3.0 * rel[:2]
            v[2] = -0.012
            if f[2] > self.F_CONTACT:
                self.z_touch = z
                self.mode = "press"
        if self.mode == "press":
            err = (self.z_touch - self.DEPTH) - z
            v[2] = np.clip(4.0 * err, -0.02, 0.02)
        return _clip(v)


class SurfaceExpert:
    """Touch down at the start of the path, then trace along +x while a PI
    loop on the normal force adjusts height; the estimated slope is used as a
    feed-forward term."""

    F_CONTACT = 1.5
    F_DES = 8.0
    SPEED = 0.03

    def reset(self):
        self.mode = "approach"
        self.i_err = 0.0

    def __call__(self, obs) -> np.ndarray:
        s = obs["state"]
        rel = s[:3].astype(float)           # estimated surface point below tool - tool
        slope = float(s[4])                 # estimated dz/dx of the surface
        to_goal = float(s[5])               # remaining distance along x
        f = obs["force"][:3].astype(float)
        v = np.zeros(3)
        if self.mode == "approach":
            goal = rel + np.array([0, 0, 0.012])
            v = 3.0 * goal
            if np.linalg.norm(goal[:2]) < 0.002 and abs(goal[2]) < 0.004:
                self.mode = "descend"
        if self.mode == "descend":
            v[2] = -0.02
            if f[2] > self.F_CONTACT:
                self.mode = "trace"
        if self.mode == "trace":
            err = self.F_DES - f[2]
            self.i_err = np.clip(self.i_err + err * 0.05, -5, 5)
            vx = self.SPEED if to_goal > 0 else 0.0
            v[0] = vx
            v[1] = 3.0 * rel[1]
            v[2] = slope * vx - 0.004 * err - 0.002 * self.i_err
        return _clip(v)


EXPERTS = {"insertion": InsertionExpert, "press": PressExpert, "surface": SurfaceExpert}
