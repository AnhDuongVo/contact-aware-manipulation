"""Side-by-side rollout videos and force traces for the same episode.

    MUJOCO_GL=osmesa python scripts/make_videos.py --task insertion \
        --eval-set heldout:calibration --episodes 0 1 2

Both policies face the identical episode (same seed: hole pose, nuisances,
sensor noise), so any difference in behaviour comes from the policy.
"""
import argparse
import os
import sys

os.environ.setdefault("MUJOCO_GL", "osmesa")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import imageio.v2 as imageio  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from cambench.analysis.report import COLOR, LABEL, SURFACE, INK2  # noqa: E402
from cambench.envs import make, sample_nuisance  # noqa: E402
from cambench.evaluation.rollout import PolicyRunner, eval_seed  # noqa: E402
from cambench.training.train import load_policy  # noqa: E402

W, H = 480, 360


def font(size):
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
              "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"):
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def rollout(task, ckpt, eval_set, split, i, camera):
    model, norm, meta = load_policy(ckpt)
    agent = PolicyRunner(model, norm, meta["config"]["history"])
    env = make(task)
    s = eval_seed(eval_set, i)
    obs = env.reset(sample_nuisance(np.random.default_rng(s), split), seed=s)
    agent.reset(1)
    renderer = mujoco.Renderer(env.model, H, W)
    frames, forces, done = [], [], False
    while True:
        renderer.update_scene(env.data, camera=camera)
        frames.append(renderer.render().copy())
        forces.append(env._last_info["peak_force"])
        if done:
            break
        a = agent([obs], np.array([True]))[0]
        obs, done, _ = env.step(a)
    renderer.close()
    m = env.episode_metrics()
    return frames, np.asarray(env.force_trace), env.sim_dt, m, meta["condition"], env


def annotate(frame, title, color, force, t, status, limit):
    img = Image.fromarray(frame)
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, W, 30], fill=(252, 252, 251))
    d.rectangle([0, 0, 8, 30], fill=color)
    d.text((14, 6), title, fill=(11, 11, 11), font=font(16))
    bar = min(1.0, force / (1.5 * limit))
    col = (208, 59, 59) if force > limit else (42, 120, 214)
    d.rectangle([10, H - 26, 10 + int((W - 20) * bar), H - 12], fill=col)
    d.rectangle([10 + int((W - 20) / 1.5), H - 30, 12 + int((W - 20) / 1.5), H - 8], fill=(208, 59, 59))
    d.text((12, H - 50), f"t={t:4.2f}s   |F|={force:5.1f} N   {status}", fill=(11, 11, 11), font=font(14))
    return np.asarray(img)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="insertion")
    p.add_argument("--root", default="results/main")
    p.add_argument("--a", default="state")
    p.add_argument("--b", default="state+proprio+force+aux")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval-set", default="heldout:calibration")
    p.add_argument("--episodes", type=int, nargs="+", default=[0])
    p.add_argument("--camera", default="side")
    p.add_argument("--out", default="results/videos")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    split = a.eval_set if a.eval_set.startswith("heldout:") else "train"
    for ep in a.episodes:
        runs = []
        for cond in (a.a, a.b):
            ck = os.path.join(a.root, a.task, cond, f"seed{a.seed}_nall", "model.pt")
            runs.append(rollout(a.task, ck, a.eval_set, split, ep, a.camera))
        limit = runs[0][5].force_limit
        n = max(len(r[0]) for r in runs)
        video = []
        for k in range(n):
            panels = []
            for frames, trace, dt, m, cond, env in runs:
                j = min(k, len(frames) - 1)
                f = float(np.max(trace[max(0, j - 1) * env.n_sub: j * env.n_sub + 1])) if j else 0.0
                status = "" if j < len(frames) - 1 else (m["failure"].replace("_", " ") if m["failure"] != "ok" else "success")
                rgb = tuple(int(COLOR[cond][i:i + 2], 16) for i in (1, 3, 5))
                panels.append(annotate(frames[j], LABEL[cond], rgb, f, j * env.control_dt, status, limit))
            video.append(np.concatenate(panels, axis=1))
        tag = f"{a.task}_{a.eval_set.replace(':', '-')}_ep{ep}"
        imageio.mimsave(os.path.join(a.out, f"{tag}.mp4"), video, fps=20, macro_block_size=8)
        imageio.mimsave(os.path.join(a.out, f"{tag}.gif"), [v[::2, ::2] for v in video[::2]], duration=0.1, loop=0)
        # force traces of the same episode
        fig, ax = plt.subplots(figsize=(7.5, 2.8))
        fig.patch.set_facecolor(SURFACE)
        for frames, trace, dt, m, cond, env in runs:
            t = np.arange(len(trace)) * dt
            ax.plot(t, trace, color=COLOR[cond], lw=1.4,
                    label=f"{LABEL[cond]}: {m['failure'].replace('_', ' ') if m['failure'] != 'ok' else 'success'}, "
                          f"peak {m['peak_force']:.1f} N")
        ax.axhline(limit, color="#d03b3b", ls=":", lw=1.2)
        ax.text(0.2, limit + 0.5, f"damage limit {limit:g} N", color="#d03b3b", fontsize=8)
        ax.set_xlabel("time [s]")
        ax.set_ylabel("|F| at the wrist [N] (1 kHz)")
        ax.set_title(f"{a.task}, {a.eval_set}, episode {ep}: same episode, two policies", fontsize=10, color=INK2)
        ax.legend(loc="upper left", fontsize=8)
        fig.savefig(os.path.join(a.out, f"{tag}_force.png"), bbox_inches="tight", dpi=130)
        plt.close(fig)
        print("wrote", tag, [(r[4], r[3]["failure"], round(r[3]["peak_force"], 1)) for r in runs])


if __name__ == "__main__":
    torch.set_num_threads(1)
    main()
