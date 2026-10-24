"""Figures and tables from evaluation CSVs.

    python -m cambench.analysis.report --main results/main --se results/sample_efficiency --out results

Every number in docs/REPORT.md is produced here, from the CSVs written by the
evaluator, so the report can be regenerated after a re-run.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from ..envs.nuisance import SWEEPS  # noqa: E402
from .stats import paired_diff, summarize  # noqa: E402

# ----------------------------------------------------------------- styling
SURFACE = "#fcfcfb"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"
COND_ORDER = [
    "state", "state+proprio", "state+force", "state+proprio+force",
    "state+proprio+aux", "state+proprio+force+aux", "privileged-aux",
]
LABEL = {
    "state": "vision/state",
    "state+proprio": "+ proprio",
    "state+force": "+ wrist F/T",
    "state+proprio+force": "+ proprio + F/T",
    "state+proprio+aux": "+ proprio + contact aux",
    "state+proprio+force+aux": "+ proprio + F/T + contact aux",
    "privileged-aux": "+ proprio + F/T + privileged aux",
    "expert": "scripted expert",
}
# Colour follows the condition everywhere (fixed slots, never re-assigned).
COLOR = {
    "state": "#eb6834", "state+proprio": "#eda100", "state+force": "#e87ba4",
    "state+proprio+force": "#2a78d6", "state+proprio+aux": "#1baf7a",
    "state+proprio+force+aux": "#6250d6", "privileged-aux": "#008300", "expert": "#52514e",
}
MARKER = {
    "state": "o", "state+proprio": "s", "state+force": "D", "state+proprio+force": "^",
    "state+proprio+aux": "v", "state+proprio+force+aux": "P", "privileged-aux": "X", "expert": "*",
}
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
CORE = ["state", "state+proprio", "state+proprio+force", "state+proprio+force+aux"]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 9.5,
    "axes.titlesize": 10.5, "axes.titleweight": "bold", "legend.frameon": False,
    "lines.linewidth": 2.0, "figure.dpi": 130,
})

SUCCESS = "safe_success"


def load(root: str) -> pd.DataFrame:
    files = glob.glob(os.path.join(root, "*", "*", "*", "episodes.csv"))
    files += glob.glob(os.path.join(root, "*", "expert", "episodes.csv"))
    dfs = [pd.read_csv(f) for f in sorted(set(files))]
    if not dfs:
        raise FileNotFoundError(f"no episodes.csv under {root}")
    df = pd.concat(dfs, ignore_index=True)
    df["family"] = df.eval_set.str.split(":").str[0]
    return df


def conds_present(df, order=COND_ORDER):
    have = set(df.condition)
    return [c for c in order if c in have]


def _save(fig, out, name):
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, name)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


# ----------------------------------------------------------------- figures
def fig_overview(df: pd.DataFrame, out: str):
    tasks = [t for t in ("insertion", "press", "surface") if t in set(df.task)]
    conds = conds_present(df) + (["expert"] if "expert" in set(df.condition) else [])
    fig, axes = plt.subplots(1, len(tasks), figsize=(4.1 * len(tasks), 3.6), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, task in zip(axes, tasks):
        d = df[(df.task == task) & df.family.isin(["id", "heldout"])]
        s = summarize(d, SUCCESS, ["condition", "family"])
        for i, c in enumerate(conds):
            for fam, dy, filled in (("id", -0.16, True), ("heldout", 0.16, False)):
                r = s[(s.condition == c) & (s.family == fam)]
                if r.empty:
                    continue
                r = r.iloc[0]
                y = i + dy
                ax.plot([r.lo, r.hi], [y, y], color=COLOR[c], lw=2, solid_capstyle="round")
                ax.plot(r["mean"], y, MARKER[c], ms=8, color=COLOR[c],
                        mfc=COLOR[c] if filled else SURFACE, mew=2)
        ax.set_yticks(range(len(conds)))
        ax.set_yticklabels([LABEL[c] for c in conds])
        ax.invert_yaxis()
        ax.set_xlim(-0.02, 1.02)
        ax.set_title(task)
        ax.set_xlabel("safe success rate")
    axes[0].plot([], [], "o", color=INK2, label="in-distribution")
    axes[0].plot([], [], "o", color=INK2, mfc=SURFACE, mew=2, label="held-out (mean of shifts)")
    axes[0].legend(loc="lower left", bbox_to_anchor=(0, -0.42), ncol=2)
    fig.suptitle("Safe success (task done, peak force below limit), 95% bootstrap CI over seeds x episodes",
                 fontsize=10, color=INK2, y=1.02)
    return _save(fig, out, "fig_overview.png")


def fig_heldout_heatmap(df: pd.DataFrame, task: str, out: str):
    d = df[(df.task == task) & df.family.isin(["id", "heldout"])]
    conds = conds_present(d) + (["expert"] if "expert" in set(d.condition) else [])
    sets = ["id"] + sorted(s for s in d.eval_set.unique() if s.startswith("heldout:"))
    piv = d.groupby(["condition", "eval_set"])[SUCCESS].mean().unstack().reindex(index=conds, columns=sets)
    fig, ax = plt.subplots(figsize=(1.0 * len(sets) + 2.6, 0.42 * len(conds) + 1.2))
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("seq", SEQ)
    ax.imshow(piv.to_numpy(float), cmap=cmap, vmin=0, vmax=1, aspect="auto")
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.iat[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8.5,
                        color="white" if v > 0.55 else INK)
    ax.set_xticks(range(len(sets)))
    ax.set_xticklabels([s.replace("heldout:", "") for s in sets], rotation=30, ha="right")
    ax.set_yticks(range(len(conds)))
    ax.set_yticklabels([LABEL[c] for c in conds])
    ax.grid(False)
    ax.set_title(f"{task}: safe success per evaluation set (mean over seeds)")
    return _save(fig, out, f"fig_heldout_{task}.png")


def fig_sweeps(df: pd.DataFrame, task: str, out: str, conds: List[str] = None):
    d = df[(df.task == task) & (df.family == "sweep")].copy()
    if d.empty:
        return None
    d["factor"] = d.eval_set.str.extract(r"sweep:([a-z_]+)=")[0]
    d["level"] = d.eval_set.str.extract(r"=([-0-9.e]+)$")[0].astype(float)
    conds = [c for c in (conds or CORE) if c in set(d.condition)]
    has_exp = "expert" in set(d.condition)
    factors = [f for f in SWEEPS if f in set(d.factor)]
    ncol = 3
    nrow = int(np.ceil(len(factors) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.0 * ncol, 2.9 * nrow), squeeze=False)
    for ax, f in zip(axes.flat, factors):
        g = d[d.factor == f]
        s = summarize(g, SUCCESS, ["condition", "level"])
        for c in conds + (["expert"] if has_exp else []):
            r = s[s.condition == c].sort_values("level")
            if r.empty:
                continue
            ls = "--" if c == "expert" else "-"
            ax.fill_between(r.level, r.lo, r.hi, color=COLOR[c], alpha=0.12, lw=0)
            ax.plot(r.level, r["mean"], ls, marker=MARKER[c], ms=6, color=COLOR[c], label=LABEL[c])
        ax.set_title(f.replace("_", " "))
        ax.set_ylim(-0.03, 1.03)
        ax.set_ylabel("safe success")
        # mark the training range
        from ..envs.nuisance import TRAIN
        lo, hi = TRAIN[f]
        ax.axvspan(lo, hi, color=GRID, alpha=0.6, lw=0, zorder=0)
    for ax in axes.flat[len(factors):]:
        ax.axis("off")
    h, lab = axes.flat[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=min(5, len(lab)), bbox_to_anchor=(0.5, -0.04))
    fig.suptitle(f"{task}: robustness to one factor at a time (shaded band = training range)",
                 fontsize=10.5, fontweight="bold", y=1.0)
    fig.tight_layout(rect=(0, 0.05, 1, 0.98))
    return _save(fig, out, f"fig_sweeps_{task}.png")


def fig_force(df: pd.DataFrame, task: str, out: str):
    d = df[(df.task == task) & df.family.isin(["id", "heldout"])]
    conds = conds_present(d) + (["expert"] if "expert" in set(d.condition) else [])
    metrics = [("peak_force", "peak contact force [N]"), ("force_overshoot", "force overshoot after first contact [N]"),
               ("impulse", "contact impulse [N s]")]
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 0.38 * len(conds) + 1.4), sharey=True)
    for ax, (m, lab) in zip(axes, metrics):
        for i, c in enumerate(conds):
            v = d[d.condition == c][m].dropna().to_numpy()
            if len(v) == 0:
                continue
            q = np.percentile(v, [10, 25, 50, 75, 90])
            ax.plot([q[0], q[4]], [i, i], color=COLOR[c], lw=1.5, alpha=0.6)
            ax.plot([q[1], q[3]], [i, i], color=COLOR[c], lw=6, solid_capstyle="round")
            ax.plot(q[2], i, "o", ms=7, color=SURFACE, mec=COLOR[c], mew=2)
        ax.set_xlabel(lab)
        if m == "peak_force":
            from ..envs import make
            lim = make(task).force_limit
            ax.axvline(lim, color="#d03b3b", lw=1.2, ls=":")
            ax.text(lim, 1.0, f" limit {lim:g} N", color="#d03b3b", fontsize=8, va="bottom",
                    transform=ax.get_xaxis_transform())
    axes[0].set_yticks(range(len(conds)))
    axes[0].set_yticklabels([LABEL[c] for c in conds])
    axes[0].invert_yaxis()
    fig.suptitle(f"{task}: force transients (median, IQR, 10-90 percentile; in-distribution + held-out)",
                 fontsize=10.5, fontweight="bold", y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return _save(fig, out, f"fig_force_{task}.png")


FAIL_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]


def fig_failures(df: pd.DataFrame, task: str, out: str):
    d = df[(df.task == task) & df.family.isin(["id", "heldout"])]
    conds = conds_present(d) + (["expert"] if "expert" in set(d.condition) else [])
    modes = [m for m in d.failure.value_counts().index if m != "ok"]
    frac = d.groupby("condition").failure.value_counts(normalize=True).unstack(fill_value=0).reindex(conds)
    fig, ax = plt.subplots(figsize=(8.5, 0.4 * len(conds) + 1.3))
    left = np.zeros(len(conds))
    for k, m in enumerate(modes):
        v = frac[m].to_numpy() if m in frac else np.zeros(len(conds))
        ax.barh(range(len(conds)), v, left=left, color=FAIL_COLORS[k % len(FAIL_COLORS)],
                edgecolor=SURFACE, linewidth=2, height=0.7, label=m.replace("_", " "))
        left += v
    ax.set_yticks(range(len(conds)))
    ax.set_yticklabels([LABEL[c] for c in conds])
    ax.invert_yaxis()
    ax.set_xlabel("fraction of episodes (in-distribution + held-out)")
    ax.legend(loc="lower right", ncol=1)
    ax.set_title(f"{task}: failure taxonomy (everything that is not a safe success)")
    return _save(fig, out, f"fig_failures_{task}.png")


def fig_two_by_two(df: pd.DataFrame, out: str):
    """Is force useful as an input, or as a training signal for a predictive state?"""
    cells = [("state+proprio", "no", "no"), ("state+proprio+aux", "no", "yes"),
             ("state+proprio+force", "yes", "no"), ("state+proprio+force+aux", "yes", "yes")]
    tasks = [t for t in ("insertion", "press", "surface") if t in set(df.task)]
    fig, axes = plt.subplots(1, len(tasks), figsize=(3.6 * len(tasks), 3.0), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, task in zip(axes, tasks):
        d = df[(df.task == task) & df.family.isin(["id", "heldout"])]
        s = summarize(d, SUCCESS, ["condition"])
        for c, fin, aux in cells:
            r = s[s.condition == c]
            if r.empty:
                continue
            r = r.iloc[0]
            x = 0 if fin == "no" else 1
            dx = -0.08 if aux == "no" else 0.08
            ax.plot([x + dx, x + dx], [r.lo, r.hi], color=COLOR[c], lw=2)
            ax.plot(x + dx, r["mean"], MARKER[c], ms=8, color=COLOR[c], mfc=COLOR[c] if aux == "yes" else SURFACE, mew=2)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["F/T not an input", "F/T as input"])
        ax.set_xlim(-0.5, 1.5)
        ax.set_title(task)
    axes[0].set_ylabel("safe success (ID + held-out)")
    axes[0].plot([], [], "o", color=INK2, mfc=SURFACE, mew=2, label="no auxiliary loss")
    axes[0].plot([], [], "o", color=INK2, label="+ future-contact prediction (labels from F/T)")
    axes[0].legend(loc="lower left", bbox_to_anchor=(0, -0.42), ncol=2)
    fig.suptitle("Force as a control input vs. force as a training signal", fontsize=10.5, fontweight="bold", y=1.03)
    return _save(fig, out, "fig_force_input_vs_signal.png")


def fig_sample_efficiency(se: pd.DataFrame, main: pd.DataFrame, out: str):
    task = "insertion"
    d = pd.concat([se[se.task == task], main[(main.task == task) & main.condition.isin(set(se.condition))
                                                & main.family.isin(["id", "heldout"])]])
    d = d[d.eval_set.isin(set(se.eval_set))]
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.3), sharey=True)
    for ax, fam in zip(axes, ["id", "heldout"]):
        g = d[d.family == fam]
        s = summarize(g, SUCCESS, ["condition", "n_demos"])
        for c in conds_present(g, CORE):
            r = s[s.condition == c].sort_values("n_demos")
            ax.fill_between(r.n_demos, r.lo, r.hi, color=COLOR[c], alpha=0.12, lw=0)
            ax.plot(r.n_demos, r["mean"], marker=MARKER[c], ms=6, color=COLOR[c], label=LABEL[c])
        ax.set_xscale("log")
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.set_xticks(sorted(d.n_demos.unique()))
        ax.set_xticklabels([str(int(n)) for n in sorted(d.n_demos.unique())])
        ax.set_xlabel("number of demonstrations")
        ax.set_title("in-distribution" if fam == "id" else "held-out shifts")
        ax.set_ylim(-0.03, 1.03)
    axes[0].set_ylabel("safe success")
    axes[1].legend(loc="lower right")
    fig.suptitle("insertion: sample efficiency (same number of gradient updates for every size)",
                 fontsize=10.5, fontweight="bold", y=1.02)
    return _save(fig, out, "fig_sample_efficiency.png")


def fig_training(root: str, out: str):
    rows = []
    for f in glob.glob(os.path.join(root, "*", "*", "*", "train_log.json")):
        with open(f) as fh:
            j = json.load(fh)
        for r in j["log"]:
            rows.append(dict(task=j["task"], condition=j["condition"], seed=j["seed"], **r))
    if not rows:
        return None
    d = pd.DataFrame(rows)
    tasks = [t for t in ("insertion", "press", "surface") if t in set(d.task)]
    fig, axes = plt.subplots(1, len(tasks), figsize=(4 * len(tasks), 3), squeeze=False)
    for ax, task in zip(axes[0], tasks):
        for c in conds_present(d[d.task == task]):
            g = d[(d.task == task) & (d.condition == c)].groupby("step").val_action
            m = g.mean()
            ax.plot(m.index, m.values, color=COLOR[c], marker=MARKER[c], ms=4, label=LABEL[c])
        ax.set_yscale("log")
        ax.set_xlabel("gradient step")
        ax.set_title(task)
    axes[0][0].set_ylabel("validation action MSE")
    h, lab = axes[0][0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    return _save(fig, out, "fig_training_curves.png")


def fig_press_memory(mem: pd.DataFrame, main: pd.DataFrame, out: str):
    """Follow-up: 0.8 s vs 2.4 s history on the press task."""
    conds = [c for c in ("state", "state+proprio", "state+proprio+force") if c in set(mem.condition)]
    a = main[(main.task == "press") & main.condition.isin(conds)].assign(history="0.8 s")
    b = mem.assign(history="2.4 s")
    d = pd.concat([a, b])
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.8))
    ax = axes[0]
    g = d[d.family.isin(["id", "heldout"])]
    s = summarize(g, SUCCESS, ["condition", "history", "family"])
    for i, c in enumerate(conds):
        for k, (h, filled) in enumerate((("0.8 s", False), ("2.4 s", True))):
            for fam, dx in (("id", -0.1), ("heldout", 0.1)):
                r = s[(s.condition == c) & (s.history == h) & (s.family == fam)]
                if r.empty:
                    continue
                r = r.iloc[0]
                x = i * 2.6 + k * 1.1 + dx
                ax.plot([x, x], [r.lo, r.hi], color=COLOR[c], lw=2)
                ax.plot(x, r["mean"], "o" if fam == "id" else "s", ms=7, color=COLOR[c],
                        mfc=COLOR[c] if filled else SURFACE, mew=2)
    ax.set_xticks([i * 2.6 + k * 1.1 for i in range(len(conds)) for k in range(2)])
    ax.set_xticklabels(["0.8 s", "2.4 s"] * len(conds), fontsize=8.5)
    for i, c in enumerate(conds):
        ax.text(i * 2.6 + 0.55, -0.22, LABEL[c], ha="center", va="top", fontsize=9, color=INK2,
                transform=ax.get_xaxis_transform())
    ax.set_ylabel("safe success")
    ax.set_ylim(-0.03, 1.03)
    ax.set_title("ID (circles) and held-out (squares); filled = 2.4 s history")
    ax = axes[1]
    sw = d[d.eval_set.str.startswith("sweep:button_stiffness")].copy()
    sw["level"] = sw.eval_set.str.extract(r"=([-0-9.e]+)$")[0].astype(float)
    s = summarize(sw, SUCCESS, ["condition", "history", "level"])
    for c in conds:
        for h, ls in (("0.8 s", ":"), ("2.4 s", "-")):
            r = s[(s.condition == c) & (s.history == h)].sort_values("level")
            ax.plot(r.level, r["mean"], ls, marker=MARKER[c], ms=5, color=COLOR[c], label=f"{LABEL[c]}, {h}")
    from ..envs.nuisance import TRAIN
    ax.axvspan(*TRAIN["button_stiffness"], color=GRID, alpha=0.6, lw=0, zorder=0)
    ax.set_xlabel("button stiffness [N/m]")
    ax.set_ylim(-0.03, 1.03)
    ax.set_title("button stiffness sweep (shaded = training range)")
    ax.legend(fontsize=7.5, loc="lower left")
    fig.suptitle("press: does a longer history remove the force-as-depth shortcut?", fontsize=10.5,
                 fontweight="bold", y=1.02)
    fig.tight_layout()
    return _save(fig, out, "fig_press_memory.png")


# ----------------------------------------------------------------- tables
def tables(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    out = {}
    d = df[df.family.isin(["id", "heldout"])]
    s = summarize(d, SUCCESS, ["task", "condition", "family"])
    out["success"] = s
    f = []
    for m in ("peak_force", "force_overshoot", "time_to_success"):
        f.append(summarize(d, m, ["task", "condition"]))
    out["force"] = pd.concat(f)
    comps = [
        ("state+proprio", "state", "proprioception over vision/state"),
        ("state+force", "state", "wrist F/T over vision/state"),
        ("state+proprio+force", "state+proprio", "wrist F/T on top of proprioception"),
        ("state+proprio+aux", "state+proprio", "contact-prediction aux without F/T input"),
        ("state+proprio+force+aux", "state+proprio+force", "contact-prediction aux with F/T input"),
        ("state+proprio+force", "state+proprio+aux", "F/T as input vs. F/T only as training signal"),
        ("privileged-aux", "state+proprio+force+aux", "privileged vs. self-supervised aux labels"),
    ]
    rows = []
    for task in sorted(set(d.task)):
        for fam in ("id", "heldout"):
            dd = d[(d.task == task) & (d.family == fam)]
            for a, b, what in comps:
                for metric in (SUCCESS, "peak_force"):
                    r = paired_diff(dd, a, b, metric)
                    rows.append({"task": task, "eval": fam, "comparison": what, **r})
    out["paired"] = pd.DataFrame(rows)
    return out


def write_markdown(tabs: Dict[str, pd.DataFrame], path: str):
    lines = ["# Generated result tables", "",
             "Produced by `cambench/analysis/report.py`. Intervals are 95% hierarchical bootstrap "
             "intervals (seeds, then shared episodes).", ""]
    s = tabs["success"]
    for task in sorted(set(s.task)):
        lines += [f"## {task}: safe success", "", "| condition | in-distribution | held-out (mean) |", "|---|---|---|"]
        for c in COND_ORDER + ["expert"]:
            row = []
            for fam in ("id", "heldout"):
                r = s[(s.task == task) & (s.condition == c) & (s.family == fam)]
                row.append("" if r.empty else f"{r['mean'].iloc[0]:.2f} [{r.lo.iloc[0]:.2f}, {r.hi.iloc[0]:.2f}]")
            if any(row):
                lines.append(f"| {LABEL[c]} | {row[0]} | {row[1]} |")
        lines.append("")
    f = tabs["force"]
    lines += ["## Force and timing (ID + held-out pooled)", "",
              "| task | condition | peak force [N] | overshoot [N] | time to success [s] |", "|---|---|---|---|---|"]
    for task in sorted(set(f.task)):
        for c in COND_ORDER + ["expert"]:
            vals = []
            for m in ("peak_force", "force_overshoot", "time_to_success"):
                r = f[(f.task == task) & (f.condition == c) & (f.metric == m)]
                vals.append("" if r.empty else f"{r['mean'].iloc[0]:.2f} [{r.lo.iloc[0]:.2f}, {r.hi.iloc[0]:.2f}]")
            if any(vals):
                lines.append(f"| {task} | {LABEL[c]} | " + " | ".join(vals) + " |")
    lines += ["", "## Paired differences (a - b)", "",
              "| task | eval | comparison | metric | diff | 95% CI |", "|---|---|---|---|---|---|"]
    p = tabs["paired"]
    for _, r in p.iterrows():
        if r.n == 0:
            continue
        star = " *" if (r.lo > 0 or r.hi < 0) else ""
        lines.append(f"| {r.task} | {r['eval']} | {r.comparison} | {r.metric} | {r['diff']:+.3f}{star} | [{r.lo:+.3f}, {r.hi:+.3f}] |")
    lines += ["", "`*` = interval excludes zero."]
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--main", default="results/main")
    p.add_argument("--se", default="results/sample_efficiency")
    p.add_argument("--memory", default="results/press_memory")
    p.add_argument("--out", default="results")
    a = p.parse_args()
    figs = os.path.join(a.out, "figures")
    df = load(a.main)
    print("loaded", len(df), "episodes")
    fig_overview(df, figs)
    fig_two_by_two(df, figs)
    for task in sorted(set(df.task)):
        fig_heldout_heatmap(df, task, figs)
        fig_sweeps(df, task, figs)
        fig_force(df, task, figs)
        fig_failures(df, task, figs)
    fig_training(a.main, figs)
    if os.path.isdir(a.se) and glob.glob(os.path.join(a.se, "*", "*", "*", "episodes.csv")):
        se = load(a.se)
        fig_sample_efficiency(se, df, figs)
        s = summarize(pd.concat([se, df[(df.task == "insertion") & df.condition.isin(set(se.condition))
                                       & df.eval_set.isin(set(se.eval_set))]]),
                      SUCCESS, ["condition", "n_demos", "family"])
        s.to_csv(os.path.join(a.out, "table_sample_efficiency.csv"), index=False)
    if os.path.isdir(a.memory) and glob.glob(os.path.join(a.memory, "*", "*", "*", "episodes.csv")):
        fig_press_memory(load(a.memory), df, figs)
    tabs = tables(df)
    for k, v in tabs.items():
        v.to_csv(os.path.join(a.out, f"table_{k}.csv"), index=False)
    write_markdown(tabs, os.path.join(a.out, "RESULTS_TABLES.md"))
    print("done")


if __name__ == "__main__":
    main()
