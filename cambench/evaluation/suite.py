"""Evaluation sets per task and the code that runs a model through them.

Evaluation sets
---------------
* ``id``                  in-distribution: nuisances from the training ranges.
* ``heldout:<cond>``      one physical cause shifted outside the training range
                          (see envs/nuisance.py: HELDOUT_CONDITIONS).
* ``sweep:<factor>=<v>``  one factor pinned to a level, others from training
                          ranges; used for robustness curves.

The plan below is the protocol in docs/PROTOCOL.md; changing it changes the
experiment, so it lives in code next to the evaluator rather than in a
config that is easy to edit per run.
"""
from __future__ import annotations

import os
from typing import Dict, List

import pandas as pd

from ..envs.nuisance import SWEEPS
from ..experts.scripted import EXPERTS
from ..training.train import load_policy
from .rollout import ExpertRunner, PolicyRunner, run_episodes

PLAN: Dict[str, Dict] = {
    "insertion": dict(
        n_id=100,
        heldout=["calibration", "occlusion", "friction", "tight_fit", "soft_fixture", "actuator_mismatch"],
        n_heldout=50,
        sweeps=["clearance_mm", "friction", "occlusion", "calib_offset_mm", "ft_noise_n", "joint_friction_n"],
        n_sweep=30,
    ),
    "press": dict(
        n_id=100,
        heldout=["calibration", "occlusion", "stiff_button", "soft_fixture", "actuator_mismatch"],
        n_heldout=50,
        sweeps=["calib_offset_mm", "button_stiffness", "ft_noise_n", "joint_friction_n"],
        n_sweep=30,
    ),
    "surface": dict(
        n_id=100,
        heldout=["calibration", "rough_surface", "friction", "soft_fixture", "actuator_mismatch"],
        n_heldout=40,
        sweeps=["surface_amp_mm", "friction", "ft_noise_n", "joint_friction_n"],
        n_sweep=20,
    ),
}

# Smaller plan for the sample-efficiency study (many more models).
PLAN_SMALL: Dict[str, Dict] = {
    t: dict(n_id=100, heldout=p["heldout"][:3], n_heldout=30, sweeps=[], n_sweep=0)
    for t, p in PLAN.items()
}


def eval_sets(task: str, plan: Dict) -> List[Dict]:
    sets = [dict(name="id", split="train", pins={}, n=plan["n_id"])]
    for c in plan["heldout"]:
        sets.append(dict(name=f"heldout:{c}", split=f"heldout:{c}", pins={}, n=plan["n_heldout"]))
    for f in plan["sweeps"]:
        for v in SWEEPS[f]:
            sets.append(dict(name=f"sweep:{f}={v:g}", split="train", pins={f: v}, n=plan["n_sweep"]))
    return sets


def evaluate_agent(task: str, agent, plan: Dict, meta: Dict) -> pd.DataFrame:
    rows = []
    for s in eval_sets(task, plan):
        res = run_episodes(task, agent, s["name"], s["n"], split=s["split"], pins=s["pins"])
        for r in res:
            r.update(meta)
            r["eval_set"] = s["name"]
            rows.append(r)
    return pd.DataFrame(rows)


def evaluate_checkpoint(ckpt_path: str, plan: Dict, out_csv: str) -> pd.DataFrame:
    model, norm, ckpt = load_policy(ckpt_path)
    agent = PolicyRunner(model, norm, ckpt["config"]["history"])
    meta = dict(task=ckpt["task"], condition=ckpt["condition"], seed=ckpt["seed"], n_demos=ckpt["n_demos"])
    df = evaluate_agent(ckpt["task"], agent, plan, meta)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.to_csv(out_csv, index=False)
    return df


def evaluate_expert(task: str, plan: Dict, out_csv: str) -> pd.DataFrame:
    agent = ExpertRunner(EXPERTS[task])
    df = evaluate_agent(task, agent, plan, dict(task=task, condition="expert", seed=0, n_demos=0))
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.to_csv(out_csv, index=False)
    return df
