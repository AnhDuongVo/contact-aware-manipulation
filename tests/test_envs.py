import numpy as np
import pytest

from cambench.envs import ENVS, make, sample_nuisance
from cambench.envs.nuisance import HELDOUT, TRAIN, Nuisance
from cambench.experts.scripted import EXPERTS


@pytest.mark.parametrize("task", sorted(ENVS))
def test_reset_step_shapes(task):
    env = make(task)
    obs = env.reset(sample_nuisance(np.random.default_rng(0)), seed=0)
    assert set(obs) == {"state", "proprio", "force"}
    assert obs["proprio"].shape == (6,) and obs["force"].shape == (7,)
    obs2, done, info = env.step(np.zeros(3))
    assert obs2["state"].shape == obs["state"].shape
    assert not done
    for k in ("contact", "phase", "wrench_true", "target_true"):
        assert k in info


@pytest.mark.parametrize("task", sorted(ENVS))
def test_determinism(task):
    env = make(task)
    traj = []
    for _ in range(2):
        obs = env.reset(sample_nuisance(np.random.default_rng(3)), seed=3)
        out = [obs["force"].copy()]
        for _ in range(10):
            obs, _, _ = env.step(np.array([0.2, -0.1, -0.5]))
            out.append(obs["force"].copy())
        traj.append(np.stack(out))
    np.testing.assert_allclose(traj[0], traj[1])


@pytest.mark.parametrize("task", sorted(ENVS))
def test_tare_free_space(task):
    """After taring, the true wrench in free space is ~0."""
    env = make(task)
    env.reset(Nuisance(), seed=1)
    _, _, info = env.step(np.zeros(3))
    assert np.linalg.norm(info["wrench_true"][:3]) < 0.5
    assert not info["contact"]


@pytest.mark.parametrize("task", sorted(ENVS))
def test_expert_solves_nominal(task):
    env, ex = make(task), EXPERTS[task]()
    wins = 0
    for s in range(5):
        obs = env.reset(sample_nuisance(np.random.default_rng(100 + s)), seed=100 + s)
        ex.reset()
        done = False
        while not done:
            obs, done, _ = env.step(ex(obs))
        wins += env.episode_metrics()["success"]
    assert wins >= 4


def test_windup_caps_force():
    """Pushing down as hard as possible is limited by kp * windup."""
    env = make("press")
    env.reset(Nuisance(button_stiffness=3000, calib_offset_mm=0, occlusion=0), seed=0)
    target = env._true_target()
    for _ in range(60):  # move over the button
        rel = target - env.tool_pos
        env.step(np.clip(np.r_[rel[:2] * 30, 0], -1, 1))
    peak = 0.0
    while not env.done:
        _, _, info = env.step(np.array([0, 0, -1.0]))
        peak = max(peak, info["peak_force"])
    cap = env.robot.kp[2] * env.robot.windup
    assert peak < 1.3 * cap


def test_nuisance_splits():
    rng = np.random.default_rng(0)
    n = sample_nuisance(rng, "heldout:friction")
    lo, hi = HELDOUT["friction"]
    assert lo <= n.friction <= hi
    lo, hi = TRAIN["clearance_mm"]
    assert lo <= n.clearance_mm <= hi
    assert sample_nuisance(rng, "train", friction=0.123).friction == 0.123
    with pytest.raises(ValueError):
        sample_nuisance(rng, "heldout:nonexistent")
