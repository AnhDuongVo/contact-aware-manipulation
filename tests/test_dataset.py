import numpy as np

from cambench.data.dataset import Normalizer, WindowDataset, future_labels, stack_obs


def fake_arrays(lengths=(5, 3)):
    ep = np.concatenate([[i] * n for i, n in enumerate(lengths)])
    n = len(ep)
    force = np.zeros((n, 7), np.float32)
    force[2, 2] = 10.0  # contact at step 2 of episode 0
    return {
        "episode_id": ep, "obs_state": np.arange(n, dtype=np.float32)[:, None].repeat(7, 1),
        "obs_proprio": np.zeros((n, 6), np.float32), "obs_force": force,
        "action": np.zeros((n, 3), np.float32), "contact": force[:, 2] > 1,
        "phase": (force[:, 2] > 1).astype(np.int8),
    }


def test_windows_do_not_cross_episodes():
    a = fake_arrays()
    norm = Normalizer(np.zeros(20), np.ones(20))
    ds = WindowDataset(a, history=4, normalizer=norm, aux_source=None)
    x, _, _, _ = ds.batch(np.array([5]))  # first step of episode 1
    assert np.all(x[0, :, 0].numpy() == 5)  # padded with its own first frame
    x, _, _, _ = ds.batch(np.array([4]))
    assert x[0, :, 0].tolist() == [1, 2, 3, 4]


def test_future_contact_labels():
    a = fake_arrays()
    yb, yc = future_labels(a, k=2, source="contact")
    assert yb.tolist()[:5] == [1, 1, 0, 0, 0]  # contact at step 2 seen from steps 0 and 1
    assert yc[0] == 2  # 10 N falls in the 6-12 N bin
    assert yb[5:].sum() == 0  # no leakage across episodes


def test_stack_obs_order():
    a = fake_arrays()
    assert stack_obs(a).shape == (8, 20)
