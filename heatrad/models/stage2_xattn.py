"""Stage II — Organ-Attributed Confirmation (Sec. 3.3).

Single cross-attention with text anchors {t_k} as queries and slice tokens
U_j = Z_j W_vis as keys/values, applied independently per slice. No
self-attention over queries, no Q-Former blocks.

Eqs. 10-12:
    U_j     = Z_j W_vis                                  in R^{N_L x d}
    a_{k,j} = softmax( (t_k W_Q)(U_j W_K)^T / sqrt(d) )  in R^{N_L}
    q̃_{k,j}= a_{k,j} U_j W_V                            in R^d
"""
from __future__ import annotations
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Stage2Config:
    c_l: int = 768
    d: int = 768
    k: int = 32       # number of organ prompts (RadLex coarse organs)


class OrganCrossAttention(nn.Module):
    """Per-slice cross-attention from K text anchors to N_L slice tokens."""

    def __init__(self, cfg: Stage2Config):
        super().__init__()
        self.cfg = cfg
        d = cfg.d
        self.W_vis = nn.Linear(cfg.c_l, d, bias=False)
        self.W_Q = nn.Linear(d, d, bias=False)
        self.W_K = nn.Linear(d, d, bias=False)
        self.W_V = nn.Linear(d, d, bias=False)
        self.scale = d ** -0.5

    def forward(self, feats: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """feats: (B, D, N_L, C_L); t: (K, d). Returns q̃: (B, D, K, d)."""
        b, dD, n, cl = feats.shape
        k = self.cfg.k
        d = self.cfg.d
        U = self.W_vis(feats)                                # (B, D, N, d)
        K = self.W_K(U)                                      # (B, D, N, d)
        V = self.W_V(U)                                      # (B, D, N, d)
        Q = self.W_Q(t).unsqueeze(0).unsqueeze(0).expand(b, dD, -1, -1)  # (B, D, K, d)

        attn = torch.einsum("bdkc,bdnc->bdkn", Q, K) * self.scale         # (B, D, K, N)
        a = attn.softmax(dim=-1)
        qt = torch.einsum("bdkn,bdnc->bdkc", a, V)                        # (B, D, K, d)
        return qt
