"""Stage I — Suspicious-Slice Discovery (Sec. 3.2).

Eq. 6:  f_j = AttnPool(Z_j; q_slice),    g = AttnPool({f_j}; q_vol)
Eq. 7:  δ± = G±([η±; g])
Eq. 8:  α_j = softmax( cos(f_j, δ+)/τ_s,  cos(f_j, δ-)/τ_s )_+

Continuous α_j is passed unchanged to Stages II/III (no Top-R pruning).
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Stage1Config:
    c_l: int = 768
    init_temp: float = 1.0


class AttnPool(nn.Module):
    """Single learnable query, scaled-dot attention pool over a token set."""

    def __init__(self, dim: int):
        super().__init__()
        self.q = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.trunc_normal_(self.q, std=0.02)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.scale = dim ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, C). Returns (B, C).
        b = x.shape[0]
        q = self.q.expand(b, -1, -1)
        k = self.k_proj(x)
        v = self.v_proj(x)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        w = attn.softmax(dim=-1)
        return (w @ v).squeeze(1)


class AnomalyHead(nn.Module):
    def __init__(self, cfg: Stage1Config):
        super().__init__()
        c = cfg.c_l
        self.cfg = cfg

        self.eta_pos = nn.Parameter(torch.zeros(c))
        self.eta_neg = nn.Parameter(torch.zeros(c))
        nn.init.trunc_normal_(self.eta_pos, std=0.02)
        nn.init.trunc_normal_(self.eta_neg, std=0.02)

        def mk_g():
            return nn.Sequential(nn.Linear(2 * c, c), nn.GELU(), nn.Linear(c, c))
        self.g_pos = mk_g()
        self.g_neg = mk_g()

        # Eq. 6: independent single-query pools for intra-slice (q_slice) and
        # inter-slice (q_vol).
        self.intra_pool = AttnPool(c)
        self.inter_pool = AttnPool(c)

        # Learnable τ_s via softplus.
        self.log_tau = nn.Parameter(torch.tensor(0.0))

    @property
    def tau_s(self) -> torch.Tensor:
        return F.softplus(self.log_tau) + 1e-3

    def forward(self, feats: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """feats: (B, D, N_L, C). Returns (alpha (B,D), f_slice (B,D,C))."""
        b, d, n, c = feats.shape
        # Intra-slice pool: collapse (B*D) at once.
        f_slice = self.intra_pool(feats.reshape(b * d, n, c)).view(b, d, c)
        # Inter-slice pool over slices for the volume reference g.
        g = self.inter_pool(f_slice)                                        # (B, C)

        delta_pos = self.g_pos(torch.cat([self.eta_pos.expand(b, -1), g], dim=-1))   # (B, C)
        delta_neg = self.g_neg(torch.cat([self.eta_neg.expand(b, -1), g], dim=-1))

        f_n = F.normalize(f_slice, dim=-1)                                  # (B, D, C)
        dpos = F.normalize(delta_pos, dim=-1).unsqueeze(1)
        dneg = F.normalize(delta_neg, dim=-1).unsqueeze(1)
        cos_pos = (f_n * dpos).sum(dim=-1)                                  # (B, D)
        cos_neg = (f_n * dneg).sum(dim=-1)
        tau = self.tau_s
        alpha = torch.softmax(
            torch.stack([cos_pos / tau, cos_neg / tau], dim=-1), dim=-1
        )[..., 0]                                                            # (B, D)
        return alpha, f_slice
