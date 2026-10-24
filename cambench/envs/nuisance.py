"""Nuisance parameters: everything that varies between episodes and that the
policy cannot observe directly.

Each episode draws one ``Nuisance`` instance. Training data and in-distribution
evaluation use the ``train`` ranges. Held-out evaluation uses ranges that are
deliberately wider or shifted (see ``HELDOUT``), and robustness sweeps pin one
factor to a fixed level while sampling the rest from the training ranges.

All lengths are in millimetres here for readability and converted to metres
inside the environments.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
from typing import Dict, Tuple

import numpy as np

Range = Tuple[float, float]


@dataclass
class Nuisance:
    # --- perception ("vision") -------------------------------------------
    calib_offset_mm: float = 1.0    # magnitude of a fixed per-episode camera-to-robot offset
    vis_noise_mm: float = 0.5       # per-frame noise of the target estimate
    occlusion: float = 0.0          # 0..1, how badly the tool hides the target near contact
    # --- sensing ---------------------------------------------------------
    ft_noise_n: float = 0.3         # std of wrist force noise (torque noise scales with it)
    effort_noise_n: float = 0.5     # std of joint effort (motor current) noise
    effort_gain_err: float = 0.0    # relative error of the motor torque constant
    joint_friction_n: float = 0.5   # unmodelled Coulomb friction in the robot joints
    # --- physics -----------------------------------------------------------
    friction: float = 0.4           # tool/object sliding friction coefficient
    clearance_mm: float = 1.0       # insertion: hole width minus peg width
    fixture_stiffness: float = 2e4  # insertion: lateral stiffness of the fixture mount [N/m]
    button_stiffness: float = 800.0 # press: spring constant of the button [N/m]
    button_breakaway_n: float = 4.0 # press: static friction of the button ("snap")
    surface_slope_deg: float = 0.0  # surface: tilt of the surface along the path
    surface_amp_mm: float = 2.0     # surface: amplitude of unseen waviness
    contact_stiffness: float = 1.0  # multiplier on contact stiffness (1 = rigid default)

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


# Ranges used for demonstrations and in-distribution evaluation.
TRAIN: Dict[str, Range] = {
    "calib_offset_mm": (0.0, 3.0),
    "vis_noise_mm": (0.2, 1.0),
    "occlusion": (0.0, 0.5),
    "ft_noise_n": (0.1, 0.5),
    "effort_noise_n": (0.2, 1.0),
    "effort_gain_err": (-0.05, 0.05),
    "joint_friction_n": (0.0, 1.0),
    "friction": (0.2, 0.6),
    "clearance_mm": (0.6, 1.6),
    "fixture_stiffness": (1e4, 5e4),
    "button_stiffness": (400.0, 1500.0),
    "button_breakaway_n": (2.0, 6.0),
    "surface_slope_deg": (-6.0, 6.0),
    "surface_amp_mm": (0.0, 3.0),
    "contact_stiffness": (0.5, 1.0),
}

# Held-out ranges: never seen in training. Each is a physically plausible
# shift (worse calibration, heavier occlusion, stickier parts, tighter fits).
# ft_noise and effort_noise keep their training ranges here; sensor noise is
# studied with sweeps instead.
HELDOUT: Dict[str, Range] = {
    "calib_offset_mm": (2.0, 3.5),
    "vis_noise_mm": (0.5, 1.5),
    "occlusion": (0.5, 0.9),
    "ft_noise_n": (0.1, 0.5),
    "effort_noise_n": (0.2, 1.0),
    "effort_gain_err": (-0.15, 0.15),
    "joint_friction_n": (1.0, 2.5),
    "friction": (0.6, 0.8),
    "clearance_mm": (0.4, 0.6),
    "fixture_stiffness": (5e3, 1e4),
    "button_stiffness": (1500.0, 2500.0),
    "button_breakaway_n": (6.0, 10.0),
    "surface_slope_deg": (-10.0, 10.0),
    "surface_amp_mm": (3.0, 5.0),
    "contact_stiffness": (0.3, 0.5),
}

# Factors that are sampled log-uniformly because they span orders of magnitude.
LOG_UNIFORM = {"fixture_stiffness"}

# Levels for one-factor-at-a-time robustness sweeps.
SWEEPS: Dict[str, Tuple[float, ...]] = {
    "clearance_mm": (0.3, 0.5, 0.8, 1.2, 1.6, 2.0),
    "friction": (0.1, 0.3, 0.5, 0.7, 0.9, 1.1),
    "occlusion": (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
    "calib_offset_mm": (0.0, 1.0, 2.0, 3.0, 4.0, 5.0),
    "ft_noise_n": (0.0, 0.5, 1.0, 2.0, 3.0, 4.0),
    "joint_friction_n": (0.0, 0.5, 1.0, 2.0, 3.0, 4.0),
    "button_stiffness": (300.0, 800.0, 1500.0, 2000.0, 2500.0, 3000.0),
    "surface_amp_mm": (0.0, 1.0, 2.0, 3.0, 4.0, 5.0),
}


def _draw(rng: np.random.Generator, name: str, lo: float, hi: float) -> float:
    if name in LOG_UNIFORM:
        return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
    return float(rng.uniform(lo, hi))


# Named held-out conditions. Each shifts one physical cause (one or two
# coupled factors) to its HELDOUT range while every other factor keeps its
# training distribution. Shifting everything at once makes every method fail
# and hides which cause matters, so the joint shift is reported only as a
# stress test ("all").
HELDOUT_CONDITIONS: Dict[str, Tuple[str, ...]] = {
    "calibration": ("calib_offset_mm",),
    "occlusion": ("occlusion", "vis_noise_mm"),
    "friction": ("friction",),
    "tight_fit": ("clearance_mm",),
    "soft_fixture": ("fixture_stiffness", "contact_stiffness"),
    "actuator_mismatch": ("joint_friction_n", "effort_gain_err"),
    "stiff_button": ("button_stiffness", "button_breakaway_n"),
    "rough_surface": ("surface_amp_mm", "surface_slope_deg"),
    "all": tuple(f.name for f in fields(Nuisance)),
}


def sample(rng: np.random.Generator, split: str = "train", **pinned: float) -> Nuisance:
    """Draw a nuisance configuration.

    split="train"            all factors from TRAIN.
    split="heldout:<name>"   factors of the named condition from HELDOUT,
                             all others from TRAIN.
    Keyword arguments pin individual factors to fixed values (used by sweeps).
    """
    shifted: Tuple[str, ...] = ()
    if split.startswith("heldout:"):
        cond = split.split(":", 1)[1]
        if cond not in HELDOUT_CONDITIONS:
            raise ValueError(f"unknown held-out condition {cond!r}")
        shifted = HELDOUT_CONDITIONS[cond]
    elif split != "train":
        raise ValueError(f"unknown split {split!r}")
    values = {}
    for f in fields(Nuisance):
        lo, hi = (HELDOUT if f.name in shifted else TRAIN)[f.name]
        values[f.name] = _draw(rng, f.name, lo, hi)
    n = Nuisance(**values)
    unknown = set(pinned) - {f.name for f in fields(Nuisance)}
    if unknown:
        raise KeyError(f"unknown nuisance factors: {sorted(unknown)}")
    return replace(n, **{k: float(v) for k, v in pinned.items()})
