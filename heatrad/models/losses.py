"""Loss terms for HeatRAD (Sec. 3.6 of method.tex)."""
from __future__ import annotations
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


def loss_att(beta_k: torch.Tensor, y_o: torch.Tensor) -> torch.Tensor:
    """Eq. eq:Latt: per-organ multi-label BCE.

    beta_k: (B, K) in [0,1]
    y_o   : (B, K) in {0,1}
    """
    return F.binary_cross_entropy(beta_k.clamp(1e-6, 1 - 1e-6), y_o.float(), reduction="mean")


def loss_anc(qbar: torch.Tensor, t: torch.Tensor, tau: float = 0.07) -> torch.Tensor:
    """Eq. eq:Lanc: text-anchored InfoNCE (per-organ, target = matching anchor).

    qbar: (B, K, d)
    t   : (K, d)
    """
    qn = F.normalize(qbar, dim=-1)
    tn = F.normalize(t, dim=-1)
    # logits[b, k, k'] = cos(qbar_{b,k}, t_{k'}) / tau
    logits = torch.einsum("bkd,jd->bkj", qn, tn) / tau
    b, k, _ = logits.shape
    targets = torch.arange(k, device=qbar.device).unsqueeze(0).expand(b, -1)
    return F.cross_entropy(logits.reshape(b * k, k), targets.reshape(b * k))


def loss_alpha_sparse(alpha: torch.Tensor, slice_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Sparsity prior on slice anomaly score.

    Penalises mean alpha over *real* slices, encouraging the model to
    keep most α_j low so that the few suspicious slices stand out.
    This regularises against the noisy-OR saturation regime where
    a uniform mid-range alpha already minimises L_att.

    alpha: (B, D)
    slice_mask: (B, D) bool, True for real (un-padded) slices.
    """
    if slice_mask is None:
        return alpha.mean()
    m = slice_mask.float()
    return (alpha * m).sum() / m.sum().clamp(min=1)


def total_loss(
    L_att: torch.Tensor,
    L_anc: torch.Tensor,
    L_gen: torch.Tensor,
    w_att: float = 1.0,
    w_anc: float = 0.5,
    w_gen: float = 1.0,
    L_sparse: Optional[torch.Tensor] = None,
    w_sparse: float = 0.0,
) -> torch.Tensor:
    L = w_att * L_att + w_anc * L_anc + w_gen * L_gen
    if L_sparse is not None and w_sparse > 0:
        L = L + w_sparse * L_sparse
    return L
