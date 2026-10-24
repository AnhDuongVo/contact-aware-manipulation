"""Batched closed-loop evaluation.

Design choices that matter for the comparison:

* **Common random numbers.** Episode ``i`` of an evaluation set always uses
  the same seed, so every condition and every training seed faces the same
  hole positions, nuisance draws and sensor-noise sequences. Differences
  between conditions are then paired, which removes most of the
  episode-to-episode variance from the comparison.
* **Held-out seeds.** Evaluation seeds live in a range that the demonstration
  collector never uses.
* Policies run in lock-step over a batch of environments so that one forward
  pass serves many episodes.
"""
from __future__ import annotations

import zlib
from typing import Callable, Dict, List, Optional

import numpy as np
import torch

from ..data.dataset import Normalizer
from ..envs import make, sample_nuisance

EVAL_SEED_BASE = 2_000_000_000


def eval_seed(eval_set: str, i: int) -> int:
    """Deterministic, collector-disjoint seed for episode i of an eval set."""
    return EVAL_SEED_BASE + (zlib.crc32(eval_set.encode()) % 100_000) * 1000 + i


class PolicyRunner:
    """Keeps an observation history per environment and queries the model."""

    def __init__(self, model, normalizer: Normalizer, history: int):
        self.model, self.norm, self.H = model, normalizer, history

    def reset(self, n: int):
        self.buf = None
        self.n = n

    @torch.no_grad()
    def __call__(self, obs_list: List[Dict[str, np.ndarray]], active: np.ndarray) -> np.ndarray:
        x = np.stack([np.concatenate([o["state"], o["proprio"], o["force"]]) for o in obs_list])
        x = self.norm(x).astype(np.float32)
        if self.buf is None:
            self.buf = np.repeat(x[:, None], self.H, axis=1)   # pad with the first frame
        else:
            self.buf = np.roll(self.buf, -1, axis=1)
            self.buf[:, -1] = x
        out = np.zeros((len(obs_list), 3), dtype=np.float32)
        idx = np.flatnonzero(active)
        if len(idx):
            a, _, _ = self.model(torch.from_numpy(self.buf[idx]))
            out[idx] = a.numpy()
        return out


class ExpertRunner:
    """Wraps the scripted experts in the same interface (used as a reference)."""

    def __init__(self, expert_cls):
        self.cls = expert_cls

    def reset(self, n: int):
        self.experts = [self.cls() for _ in range(n)]
        for e in self.experts:
            e.reset()

    def __call__(self, obs_list, active):
        return np.stack([e(o) if a else np.zeros(3) for e, o, a in zip(self.experts, obs_list, active)])


def run_episodes(task: str, agent, eval_set: str, n_episodes: int, split: str = "train",
                 pins: Optional[Dict[str, float]] = None, batch: int = 50,
                 record: bool = False) -> List[Dict]:
    """Roll out ``agent`` on ``n_episodes`` episodes of an evaluation set.

    Returns one dict per episode: metrics, failure mode, nuisance values and
    (if ``record``) per-step traces for plotting.
    """
    results: List[Dict] = []
    for b0 in range(0, n_episodes, batch):
        ids = list(range(b0, min(b0 + batch, n_episodes)))
        envs = [make(task) for _ in ids]
        obs, nuis, traces = [], [], []
        for env, i in zip(envs, ids):
            s = eval_seed(eval_set, i)
            nu = sample_nuisance(np.random.default_rng(s), split, **(pins or {}))
            nuis.append(nu)
            obs.append(env.reset(nu, seed=s))
            traces.append({k: [] for k in ("tool_pos", "wrench_true", "phase", "action", "contact", "target_est", "target_true")})
        agent.reset(len(ids))
        active = np.ones(len(ids), dtype=bool)
        while active.any():
            acts = agent(obs, active)
            for j, env in enumerate(envs):
                if not active[j]:
                    continue
                if record:
                    info = env._last_info
                    for k in ("tool_pos", "wrench_true", "phase", "contact", "target_est", "target_true"):
                        traces[j][k].append(np.asarray(info[k]))
                    traces[j]["action"].append(acts[j].copy())
                obs[j], done, _ = env.step(acts[j])
                if done:
                    active[j] = False
        for j, (env, i) in enumerate(zip(envs, ids)):
            row = {"episode": i, "seed": eval_seed(eval_set, i), **env.episode_metrics()}
            row.update({f"nu_{k}": v for k, v in nuis[j].to_dict().items()})
            if record:
                row["trace"] = {k: np.asarray(v) for k, v in traces[j].items()}
                row["trace"]["force_1khz"] = np.asarray(env.force_trace)
                row["trace"]["fz_1khz"] = np.asarray(env.fz_trace)
                row["trace"]["sim_dt"] = env.sim_dt
            results.append(row)
    return results
