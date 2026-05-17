"""Stage II — Organ-Attributed Confirmation.

Components:
  - Text-anchored organ prototypes t_k via a frozen BiomedBERT + LoRA
    residual phi (LoRA implemented via PEFT in heatrad.py; this module
    just owns the reference for inference).
  - Q-Former Q: K learnable queries + L_q transformer decoder blocks
    (self-attn over queries -> cross-attn to slice tokens -> FFN).
    Returns ~q_{k,j} and final-layer cross-attn weights A_{k,j}.
  - Parameter-tied scalar head h(.) -> beta_{j,k} = sigma(h(~q_{k,j})).
  - Noisy-OR aggregation -> beta_k.
  - Per-slice significance map M_j[n] = max_k beta_{j,k} * A_{k,j}[n].
  - Slice-aggregated visual organ token bar_q_k for L_anc.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# Q-Former block: self-attn (over queries) + cross-attn (queries->visual) + FFN.
# -----------------------------------------------------------------------------
class _MultiHeadAttention(nn.Module):
    def __init__(self, dim: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert dim % n_heads == 0
        self.h = n_heads
        self.dh = dim // n_heads
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)
        self.scale = self.dh ** -0.5

    def forward(self, q, kv, return_attn: bool = False):
        # q: (B, Lq, C); kv: (B, Lk, C)
        b, lq, c = q.shape
        lk = kv.shape[1]
        Q = self.q(q).view(b, lq, self.h, self.dh).transpose(1, 2)
        K = self.k(kv).view(b, lk, self.h, self.dh).transpose(1, 2)
        V = self.v(kv).view(b, lk, self.h, self.dh).transpose(1, 2)
        attn = (Q @ K.transpose(-2, -1)) * self.scale  # (B, h, Lq, Lk)
        attn = attn.softmax(dim=-1)
        attn = self.drop(attn)
        out = (attn @ V).transpose(1, 2).contiguous().view(b, lq, c)
        out = self.o(out)
        if return_attn:
            return out, attn  # attn: (B, h, Lq, Lk)
        return out


class _QFormerBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.ln_q1 = nn.LayerNorm(dim)
        self.self_attn = _MultiHeadAttention(dim, n_heads, dropout)
        self.ln_q2 = nn.LayerNorm(dim)
        self.cross_attn = _MultiHeadAttention(dim, n_heads, dropout)
        self.ln_q3 = nn.LayerNorm(dim)
        h = int(dim * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(dim, h), nn.GELU(), nn.Linear(h, dim), nn.Dropout(dropout),
        )

    def forward(self, q, kv, return_attn: bool = False):
        q = q + self.self_attn(self.ln_q1(q), self.ln_q1(q))
        x = self.ln_q2(q)
        if return_attn:
            ca_out, ca_w = self.cross_attn(x, kv, return_attn=True)
            q = q + ca_out
        else:
            q = q + self.cross_attn(x, kv)
            ca_w = None
        q = q + self.ffn(self.ln_q3(q))
        return q, ca_w


# -----------------------------------------------------------------------------
# Stage II module.
# -----------------------------------------------------------------------------
@dataclass
class Stage2Config:
    c_l: int = 768          # encoder dim
    d: int = 768            # Q-Former hidden dim (= LLM hidden / 4 typically)
    k: int = 24             # number of organ prototypes
    n_layers: int = 4
    n_heads: int = 12
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    init_temp_c: float = 0.07  # tau_c for L_anc (handled in losses.py)


class OrganQFormer(nn.Module):
    """Stage II: Q-Former with K learnable queries, organ-attribution head,
    noisy-OR aggregation, and per-slice significance map.
    """

    def __init__(self, cfg: Stage2Config):
        super().__init__()
        self.cfg = cfg
        self.proj_v = nn.Linear(cfg.c_l, cfg.d, bias=False)  # W_v
        self.queries = nn.Parameter(torch.zeros(cfg.k, cfg.d))
        nn.init.trunc_normal_(self.queries, std=0.02)
        self.blocks = nn.ModuleList([
            _QFormerBlock(cfg.d, cfg.n_heads, cfg.mlp_ratio, cfg.dropout)
            for _ in range(cfg.n_layers)
        ])
        self.ln_out = nn.LayerNorm(cfg.d)
        # Parameter-tied scalar head h: R^d -> R.
        self.h_head = nn.Linear(cfg.d, 1)

    def forward(self, feats: torch.Tensor, topk_idx: torch.Tensor):
        """feats: (B, D, N_L, C_L);  topk_idx: (B, R) suspicious slice indices.

        Returns:
          q_jk    : (B, R, K, d)  organ-conditioned visual tokens per slice
          A_jk    : (B, R, K, N_L) cross-attn weights (head-mean) of last layer
          beta_jk : (B, R, K)     per-slice per-organ probability
        """
        b, d_full, n, c = feats.shape
        r = topk_idx.shape[1]
        k = self.cfg.k

        # Gather suspicious slice features: (B, R, N, C_L)
        gather_idx = topk_idx.view(b, r, 1, 1).expand(-1, -1, n, c)
        feats_s = feats.gather(1, gather_idx)
        # Project to Q-Former dim and flatten over (B*R, N, d)
        kv = self.proj_v(feats_s).view(b * r, n, self.cfg.d)

        # K queries broadcast to (B*R, K, d)
        q = self.queries.unsqueeze(0).expand(b * r, -1, -1).contiguous()

        last_attn = None
        for i, blk in enumerate(self.blocks):
            return_attn = (i == len(self.blocks) - 1)
            q, attn = blk(q, kv, return_attn=return_attn)
            if return_attn:
                # attn: (B*R, h, K, N) -> head-mean -> (B*R, K, N)
                last_attn = attn.mean(dim=1)
        q = self.ln_out(q)                                # (B*R, K, d)

        # Reshape back.
        q_jk = q.view(b, r, k, self.cfg.d)
        A_jk = last_attn.view(b, r, k, n)
        # Per-slice per-organ logits + sigmoid.
        beta_jk = self.h_head(q_jk).squeeze(-1).sigmoid()  # (B, R, K)

        return q_jk, A_jk, beta_jk


def noisy_or(alpha_top: torch.Tensor, beta_jk: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Eq. eq:noisyor:  beta_k = 1 - prod_j (1 - alpha_j * beta_jk).

    alpha_top: (B, R)
    beta_jk  : (B, R, K)
    returns  : (B, K)
    """
    # Use log-sum to avoid underflow.
    p = (alpha_top.unsqueeze(-1) * beta_jk).clamp(0, 1 - eps)
    log_one_minus = torch.log1p(-p)                # (B, R, K)
    s = log_one_minus.sum(dim=1)                   # (B, K)
    return 1.0 - torch.exp(s)


def slice_aggregated_q(alpha_top: torch.Tensor, q_jk: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Eq. eq:qbar:  bar_q_k = sum_j alpha_j ~q_{k,j} / (sum_j alpha_j + eps)."""
    # alpha_top: (B, R), q_jk: (B, R, K, d).
    a = alpha_top.unsqueeze(-1).unsqueeze(-1)       # (B, R, 1, 1)
    num = (a * q_jk).sum(dim=1)                     # (B, K, d)
    den = alpha_top.sum(dim=1, keepdim=True).unsqueeze(-1) + eps  # (B, 1, 1)
    return num / den                                # (B, K, d)


def significance_map(beta_jk: torch.Tensor, A_jk: torch.Tensor) -> torch.Tensor:
    """Eq. eq:Mj:  M_j[n] = max_k beta_{j,k} * A_{k,j}[n].

    beta_jk: (B, R, K)
    A_jk   : (B, R, K, N)
    returns: (B, R, N)
    """
    weighted = beta_jk.unsqueeze(-1) * A_jk          # (B, R, K, N)
    return weighted.max(dim=2).values                # (B, R, N)
