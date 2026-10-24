"""Temporal Transformer policy with modality masking.

Every experimental condition uses exactly the same network. Conditions differ
only in (a) which observation groups are zeroed before the input projection
and (b) whether the auxiliary head receives a loss. Parameter count, depth,
width, history length, optimiser and number of updates are therefore
identical across conditions, so a difference in outcome can be attributed to
the information available, not to capacity.

Architecture
------------
    obs window [B, H, F]  --mask-->  Linear(F, d)  + learned position embedding
    -> causal TransformerEncoder (pre-norm)  -> latent z_t (last token)
    -> aux head: Linear(d, 1 + 4)  (future-contact logit, 4-way future class)
    -> action head: MLP([z_t, sigmoid(contact), softmax(class)]) -> tanh -> [-1, 1]^3

Feeding the auxiliary predictions into the action head makes the predicted
contact state an explicit part of the policy's state, which is what the
"predictive representation" condition tests.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

N_CLS = 4


@dataclass
class PolicyConfig:
    groups: Tuple[str, ...] = ("state", "proprio", "force")
    aux: str = "none"            # none | contact | privileged
    history: int = 16
    d_model: int = 64
    n_layers: int = 2
    n_heads: int = 4
    dropout: float = 0.1
    aux_k: int = 5
    aux_weight: float = 0.5

    def to_dict(self):
        d = asdict(self)
        d["groups"] = list(self.groups)
        return d


class TemporalPolicy(nn.Module):
    def __init__(self, dims: Dict[str, int], cfg: PolicyConfig):
        super().__init__()
        self.cfg = cfg
        self.dims = dict(dims)
        mask = []
        for g in ("state", "proprio", "force"):
            mask += [1.0 if g in cfg.groups else 0.0] * dims[g]
        self.register_buffer("mask", torch.tensor(mask))
        in_dim = len(mask)
        d = cfg.d_model
        self.inp = nn.Linear(in_dim, d)
        self.pos = nn.Parameter(torch.zeros(1, cfg.history, d))
        nn.init.normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(d, cfg.n_heads, 4 * d, cfg.dropout,
                                           batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, cfg.n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.aux_head = nn.Linear(d, 1 + N_CLS)
        self.act_head = nn.Sequential(
            nn.Linear(d + 1 + N_CLS, 128), nn.GELU(), nn.Linear(128, 128), nn.GELU(), nn.Linear(128, 3)
        )
        causal = torch.triu(torch.full((cfg.history, cfg.history), float("-inf")), diagonal=1)
        self.register_buffer("causal", causal)

    def forward(self, x: torch.Tensor):
        x = x * self.mask
        h = self.inp(x) + self.pos[:, -x.shape[1]:]
        h = self.encoder(h, mask=self.causal[: x.shape[1], : x.shape[1]], is_causal=True)
        z = self.norm(h[:, -1])
        aux = self.aux_head(z)
        feats = torch.cat([z, torch.sigmoid(aux[:, :1]), torch.softmax(aux[:, 1:], -1)], -1)
        action = torch.tanh(self.act_head(feats))
        return action, aux, z

    def loss(self, x, y, yb, yc):
        a, aux, _ = self(x)
        l_act = F.mse_loss(a, y)
        out = {"action": l_act}
        total = l_act
        if self.cfg.aux != "none":
            l_bin = F.binary_cross_entropy_with_logits(aux[:, 0], yb)
            l_cls = F.cross_entropy(aux[:, 1:], yc)
            total = total + self.cfg.aux_weight * (l_bin + l_cls)
            out.update(aux_bin=l_bin, aux_cls=l_cls)
        out["total"] = total
        return out


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


# The experimental conditions. Names are used in configs, files and figures.
CONDITIONS: Dict[str, Dict] = {
    "state":                  dict(groups=("state",), aux="none"),
    "state+proprio":          dict(groups=("state", "proprio"), aux="none"),
    "state+force":            dict(groups=("state", "force"), aux="none"),
    "state+proprio+force":    dict(groups=("state", "proprio", "force"), aux="none"),
    "state+proprio+aux":      dict(groups=("state", "proprio"), aux="contact"),
    "state+proprio+force+aux": dict(groups=("state", "proprio", "force"), aux="contact"),
    "privileged-aux":          dict(groups=("state", "proprio", "force"), aux="privileged"),
}


def make_config(condition: str, **overrides) -> PolicyConfig:
    spec = dict(CONDITIONS[condition])
    spec.update(overrides)
    return PolicyConfig(**spec)
