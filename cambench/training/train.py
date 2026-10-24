"""Behaviour-cloning training with a fixed update budget.

The data split is decided by the *seed* and the number of demonstrations,
never by the condition, so every condition trained with seed s sees exactly
the same demonstrations in the same order of availability.
"""
from __future__ import annotations

import json
import math
import os
import time
from typing import Dict, Optional

import numpy as np
import torch

from ..data.collect import load
from ..data.dataset import Normalizer, WindowDataset, episode_subset, group_dims, stack_obs
from ..models.policy import TemporalPolicy, make_config, n_params

N_VAL = 40          # episodes held out for validation loss (same for all runs)
SPLIT_SEED = 12345  # fixes which episodes are validation episodes


def split_episodes(n_total: int, n_demos: Optional[int], seed: int):
    perm = np.random.default_rng(SPLIT_SEED).permutation(n_total)
    val, pool = perm[:N_VAL], perm[N_VAL:]
    pool = np.random.default_rng(seed).permutation(pool)
    n = len(pool) if n_demos is None else min(n_demos, len(pool))
    return pool[:n], val


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)


def train(task: str, condition: str, seed: int, data_dir: str, out_dir: str,
          n_demos: Optional[int] = None, steps: int = 3000, batch_size: int = 128,
          lr: float = 3e-4, weight_decay: float = 1e-4, warmup: int = 200,
          log_every: int = 250, overrides: Optional[Dict] = None, verbose: bool = True) -> Dict:
    set_seed(seed)
    arrays, meta = load(data_dir)
    tr_eps, val_eps = split_episodes(meta["n_episodes"], n_demos, seed)
    tr, va = episode_subset(arrays, tr_eps), episode_subset(arrays, val_eps)
    cfg = make_config(condition, **(overrides or {}))
    norm = Normalizer.fit(stack_obs(tr))
    aux_src = None if cfg.aux == "none" else cfg.aux
    # Validation always gets both label types so aux accuracy can be reported.
    ds_tr = WindowDataset(tr, cfg.history, norm, aux_src, cfg.aux_k)
    ds_va = WindowDataset(va, cfg.history, norm, aux_src or "contact", cfg.aux_k)
    model = TemporalPolicy(group_dims(arrays), cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    def lr_at(step):
        if step < warmup:
            return (step + 1) / warmup
        p = (step - warmup) / max(1, steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * p)) * 0.95 + 0.05

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    g = torch.Generator().manual_seed(seed)
    log, t0 = [], time.time()
    for step in range(steps):
        model.train()
        ids = torch.randint(0, len(ds_tr), (batch_size,), generator=g)
        losses = model.loss(*ds_tr.batch(ids))
        opt.zero_grad(set_to_none=True)
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if (step + 1) % log_every == 0 or step == steps - 1:
            row = {"step": step + 1, **{f"train_{k}": float(v.detach()) for k, v in losses.items()}}
            row.update(evaluate_loss(model, ds_va))
            row["time"] = time.time() - t0
            log.append(row)
            if verbose:
                print(f"[{task}|{condition}|s{seed}|n{len(tr_eps)}] step {step + 1} "
                      f"train {row['train_action']:.4f} val {row['val_action']:.4f} "
                      f"aux_acc {row['val_aux_bin_acc']:.3f} ({row['time']:.0f}s)", flush=True)
    os.makedirs(out_dir, exist_ok=True)
    ckpt = {
        "state_dict": model.state_dict(), "config": cfg.to_dict(), "dims": group_dims(arrays),
        "normalizer": norm.state_dict(), "task": task, "condition": condition, "seed": seed,
        "n_demos": int(len(tr_eps)), "n_params": n_params(model), "steps": steps,
    }
    torch.save(ckpt, os.path.join(out_dir, "model.pt"))
    summary = {k: v for k, v in ckpt.items() if k not in ("state_dict", "normalizer")}
    summary["log"] = log
    with open(os.path.join(out_dir, "train_log.json"), "w") as f:
        json.dump(summary, f, indent=1)
    return summary


@torch.no_grad()
def evaluate_loss(model: TemporalPolicy, ds: WindowDataset, bs: int = 2048) -> Dict[str, float]:
    model.eval()
    se, n, correct_b, correct_c = 0.0, 0, 0, 0
    for i in range(0, len(ds), bs):
        ids = torch.arange(i, min(i + bs, len(ds)))
        x, y, yb, yc = ds.batch(ids)
        a, aux, _ = model(x)
        se += float(((a - y) ** 2).mean(1).sum())
        correct_b += int(((aux[:, 0] > 0).float() == yb).sum())
        correct_c += int((aux[:, 1:].argmax(-1) == yc).sum())
        n += len(ids)
    return {"val_action": se / n, "val_aux_bin_acc": correct_b / n, "val_aux_cls_acc": correct_c / n}


def load_policy(path: str):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    from ..models.policy import PolicyConfig
    c = dict(ckpt["config"])
    c["groups"] = tuple(c["groups"])
    model = TemporalPolicy(ckpt["dims"], PolicyConfig(**c))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, Normalizer.from_state_dict(ckpt["normalizer"]), ckpt
