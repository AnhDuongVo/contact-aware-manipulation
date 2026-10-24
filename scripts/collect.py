"""Collect scripted demonstrations.

    python scripts/collect.py --task insertion --episodes 400 --seed 0
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cambench.data.collect import collect  # noqa: E402

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True, choices=["insertion", "press", "surface"])
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dart-sigma", type=float, default=0.15)
    p.add_argument("--out", default=None)
    a = p.parse_args()
    out = a.out or os.path.join("data", a.task)
    collect(a.task, a.episodes, a.seed, out, dart_sigma=a.dart_sigma)
