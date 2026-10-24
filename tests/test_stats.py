import numpy as np
import pandas as pd

from cambench.analysis.stats import paired_diff, summarize


def make_df():
    rows = []
    rng = np.random.default_rng(0)
    for cond, p in (("a", 0.9), ("b", 0.5)):
        for seed in range(3):
            for ep in range(100):
                rows.append(dict(condition=cond, seed=seed, eval_set="id", episode=ep,
                                 safe_success=float(rng.random() < p)))
    return pd.DataFrame(rows)


def test_summarize_contains_truth():
    s = summarize(make_df(), "safe_success", ["condition"])
    a = s[s.condition == "a"].iloc[0]
    assert a.lo < 0.9 < a.hi


def test_paired_diff_detects_effect():
    r = paired_diff(make_df(), "a", "b", "safe_success")
    assert r["lo"] > 0.25 and r["n"] == 300
