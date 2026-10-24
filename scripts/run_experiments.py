"""Run a whole experiment from a YAML config: collect -> train -> evaluate.

    python scripts/run_experiments.py configs/main.yaml --workers 2

Every job writes its outputs next to a marker file, so an interrupted run can
be restarted with the same command and only missing jobs are executed.
"""
import argparse
import itertools
import os
import sys
import time
import traceback
from multiprocessing import get_context

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def job_dir(cfg, task, cond, seed, n_demos):
    tag = "all" if n_demos is None else str(n_demos)
    return os.path.join(ROOT, cfg["out"], task, cond, f"seed{seed}_n{tag}")


def run_job(args):
    cfg, task, cond, seed, n_demos = args
    import torch
    torch.set_num_threads(1)
    from cambench.evaluation.suite import PLAN, PLAN_SMALL, evaluate_checkpoint
    from cambench.training.train import train
    d = job_dir(cfg, task, cond, seed, n_demos)
    ckpt, csv = os.path.join(d, "model.pt"), os.path.join(d, "episodes.csv")
    t0 = time.time()
    try:
        if not os.path.exists(ckpt):
            train(task, cond, seed, os.path.join(ROOT, cfg["data"], task), d, n_demos=n_demos,
                  steps=cfg["train"]["steps"], batch_size=cfg["train"]["batch_size"],
                  lr=cfg["train"]["lr"], overrides=cfg.get("model"), verbose=False)
        if not os.path.exists(csv):
            plan = {"full": PLAN, "small": PLAN_SMALL}[cfg["eval_plan"]][task]
            if cfg.get("eval_scale", 1.0) != 1.0:
                plan = {k: (max(1, int(v * cfg["eval_scale"])) if k.startswith("n_") else v) for k, v in plan.items()}
            evaluate_checkpoint(ckpt, plan, csv)
        return f"ok   {task:9s} {cond:26s} s{seed} n{n_demos}  {time.time() - t0:5.0f}s"
    except Exception:
        return f"FAIL {task} {cond} s{seed} n{n_demos}\n{traceback.format_exc()}"


def run_expert(args):
    cfg, task = args
    from cambench.evaluation.suite import PLAN, PLAN_SMALL, evaluate_expert
    out = os.path.join(ROOT, cfg["out"], task, "expert", "episodes.csv")
    if not os.path.exists(out):
        plan = {"full": PLAN, "small": PLAN_SMALL}[cfg["eval_plan"]][task]
        evaluate_expert(task, plan, out)
    return f"ok   {task} expert"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("config")
    p.add_argument("--workers", type=int, default=2)
    a = p.parse_args()
    with open(a.config) as f:
        cfg = yaml.safe_load(f)

    from cambench.data.collect import collect
    for task in cfg["tasks"]:
        ddir = os.path.join(ROOT, cfg["data"], task)
        if not os.path.exists(os.path.join(ddir, "demos.npz")):
            c = cfg["collect"]
            n_ep = c["episodes"][task] if isinstance(c["episodes"], dict) else c["episodes"]
            collect(task, n_ep, c["seed"], ddir, dart_sigma=c["dart_sigma"])

    n_list = cfg.get("n_demos", [None])
    jobs = [(cfg, t, c, s, n) for t, c, s, n in
            itertools.product(cfg["tasks"], cfg["conditions"], cfg["seeds"], n_list)]
    print(f"{len(jobs)} train/eval jobs, {a.workers} workers", flush=True)
    ctx = get_context("spawn")
    with ctx.Pool(a.workers, maxtasksperchild=1) as pool:
        if cfg.get("eval_expert", False):
            for msg in pool.imap_unordered(run_expert, [(cfg, t) for t in cfg["tasks"]]):
                print(msg, flush=True)
        for i, msg in enumerate(pool.imap_unordered(run_job, jobs)):
            print(f"[{i + 1}/{len(jobs)}] {msg}", flush=True)


if __name__ == "__main__":
    main()
