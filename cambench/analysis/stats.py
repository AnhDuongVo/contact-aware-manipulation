"""Uncertainty estimates.

Two sources of variance matter here: which demonstrations/initialisation a
policy got (training seed) and which episodes it was tested on. Because all
conditions are evaluated on the same episodes (common random numbers) and
trained on the same demonstration subsets per seed, comparisons are paired.

* ``summarize``: mean over seeds of the per-seed mean, with a 95 % interval
  from a hierarchical bootstrap (resample seeds, then episodes within the
  shared episode set).
* ``paired_diff``: per (seed, episode) difference between two conditions,
  bootstrapped the same way. An interval that excludes 0 is what we call an
  effect in the report.
"""
from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import pandas as pd

N_BOOT = 2000


def _matrix(df: pd.DataFrame, metric: str) -> np.ndarray:
    """seeds x episodes matrix of a metric (NaN where missing)."""
    piv = df.pivot_table(index="seed", columns=["eval_set", "episode"], values=metric, aggfunc="mean")
    return piv.to_numpy(dtype=float)


def hierarchical_bootstrap(mat: np.ndarray, rng: np.random.Generator, n_boot: int = N_BOOT):
    n_s, n_e = mat.shape
    stats = np.empty(n_boot)
    for b in range(n_boot):
        si = rng.integers(0, n_s, n_s)
        ei = rng.integers(0, n_e, n_e)
        stats[b] = np.nanmean(mat[np.ix_(si, ei)])
    return stats


def summarize(df: pd.DataFrame, metric: str, by: Iterable[str], seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for key, g in df.groupby(list(by)):
        mat = _matrix(g, metric)
        if mat.size == 0 or np.all(np.isnan(mat)):
            continue
        point = np.nanmean(np.nanmean(mat, axis=1))
        boots = hierarchical_bootstrap(mat, rng, N_BOOT if mat.size > 1 else 1)
        key = key if isinstance(key, tuple) else (key,)
        rows.append({**dict(zip(by, key)), "metric": metric, "mean": point,
                     "lo": np.nanpercentile(boots, 2.5), "hi": np.nanpercentile(boots, 97.5),
                     "n_seeds": mat.shape[0], "n_episodes": mat.shape[1]})
    return pd.DataFrame(rows)


def paired_diff(df: pd.DataFrame, cond_a: str, cond_b: str, metric: str,
                eval_sets: Optional[Iterable[str]] = None, seed: int = 0) -> dict:
    """Mean of (a - b) over matched (seed, eval_set, episode) with a 95 % CI."""
    d = df if eval_sets is None else df[df.eval_set.isin(list(eval_sets))]
    a = d[d.condition == cond_a].set_index(["seed", "eval_set", "episode"])[metric]
    b = d[d.condition == cond_b].set_index(["seed", "eval_set", "episode"])[metric]
    joined = pd.concat([a.rename("a"), b.rename("b")], axis=1, join="inner").dropna()
    if joined.empty:
        return {"a": cond_a, "b": cond_b, "metric": metric, "diff": np.nan, "lo": np.nan, "hi": np.nan, "n": 0}
    diff = (joined.a - joined.b).rename("d").reset_index()
    diff["ep"] = diff.eval_set + "#" + diff.episode.astype(str)
    mat = diff.pivot_table(index="seed", columns="ep", values="d").to_numpy(dtype=float)
    boots = hierarchical_bootstrap(mat, np.random.default_rng(seed))
    return {"a": cond_a, "b": cond_b, "metric": metric,
            "diff": float(np.nanmean(np.nanmean(mat, axis=1))),
            "lo": float(np.nanpercentile(boots, 2.5)), "hi": float(np.nanpercentile(boots, 97.5)),
            "n": int(np.isfinite(mat).sum())}
