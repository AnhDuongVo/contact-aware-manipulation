import numpy as np
import torch

from cambench.models.policy import CONDITIONS, TemporalPolicy, make_config, n_params

DIMS = {"state": 7, "proprio": 6, "force": 7}


def test_identical_capacity():
    counts = {c: n_params(TemporalPolicy(DIMS, make_config(c))) for c in CONDITIONS}
    assert len(set(counts.values())) == 1, counts


def test_masked_groups_have_no_influence():
    torch.manual_seed(0)
    m = TemporalPolicy(DIMS, make_config("state")).eval()
    x = torch.randn(4, 16, 20)
    x2 = x.clone()
    x2[..., 7:] = torch.randn(4, 16, 13) * 100  # change proprio + force only
    a1, _, _ = m(x)
    a2, _, _ = m(x2)
    torch.testing.assert_close(a1, a2)


def test_causal():
    """The action at the last step must not depend on padding order beyond the window."""
    torch.manual_seed(0)
    m = TemporalPolicy(DIMS, make_config("state+proprio+force")).eval()
    x = torch.randn(2, 16, 20)
    a, aux, z = m(x)
    assert a.shape == (2, 3) and aux.shape == (2, 5) and z.shape == (2, 64)
    assert torch.all(a.abs() <= 1)


def test_aux_loss_only_when_enabled():
    m0 = TemporalPolicy(DIMS, make_config("state+proprio"))
    m1 = TemporalPolicy(DIMS, make_config("state+proprio+aux"))
    x, y = torch.randn(8, 16, 20), torch.rand(8, 3) * 2 - 1
    yb, yc = torch.randint(0, 2, (8,)).float(), torch.randint(0, 4, (8,))
    assert "aux_bin" not in m0.loss(x, y, yb, yc)
    assert "aux_bin" in m1.loss(x, y, yb, yc)
