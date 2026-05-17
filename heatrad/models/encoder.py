"""Heat-Conduction Slice Encoder (Sec. 3.1 of HeatRAD NeurIPS 2026).

In-plane signed S-HCO via vendored vHeat backbone (with ReLU stripped from
to_k so κ_θℓ may be negative -> amplification of lesion-like residues).
Inter-slice 1-D HCO Ψ uses a true DCT-II along the slice axis with an
unsigned softplus diffusivity (Eq. 5).
"""
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..external.hco.vheat import vHeat


# -----------------------------------------------------------------------------
# 1-D HCO Ψ along slice axis using DCT-II via cosine-basis matmul.
# -----------------------------------------------------------------------------
def _dct_basis(N: int, device, dtype) -> torch.Tensor:
    # Returns (N, N) orthonormal DCT-II matrix W s.t. X̂ = W @ x.
    n = torch.arange(N, device=device, dtype=dtype).view(1, -1)
    k = torch.arange(N, device=device, dtype=dtype).view(-1, 1)
    W = torch.cos(math.pi * k * (2 * n + 1) / (2 * N)) * math.sqrt(2.0 / N)
    W[0] = W[0] / math.sqrt(2.0)
    return W                                                 # (N, N)


class Heat1D(nn.Module):
    """1-D heat-conduction along slice axis.

    Forward Z: (B, N_L, D, C) -> Z' = IDCT(exp(-κ(ω)) ⊙ DCT(Z)) + FFN(LN(Z)).
    κ(ω) ∈ R^{D, C} >0 via softplus on a learnable freq embedding (unsigned,
    per Sec. 3.1: "the slice axis primarily supplies anatomical continuity").
    """

    def __init__(self, dim: int, max_d: int = 256, mlp_ratio: float = 2.0):
        super().__init__()
        self.dim = dim
        self.max_d = max_d
        self.freq_embed = nn.Parameter(torch.zeros(max_d, dim))
        nn.init.trunc_normal_(self.freq_embed, std=0.02)
        self.norm = nn.LayerNorm(dim)
        h = int(dim * mlp_ratio)
        self.ffn = nn.Sequential(nn.Linear(dim, h), nn.GELU(), nn.Linear(h, dim))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: (B, N, D, C). Apply DCT-II along D, gate, IDCT-II.
        b, n, d, c = z.shape
        W = _dct_basis(d, device=z.device, dtype=z.dtype)            # (D, D)
        # DCT along D (axis 2): zhat[..., k, c] = sum_n W[k,n] * z[..., n, c]
        zhat = torch.einsum("kn,bxnc->bxkc", W, z)
        # gate
        kappa = F.softplus(self.freq_embed[:d])                      # (D, C) > 0
        gate = torch.exp(-kappa).unsqueeze(0).unsqueeze(0)           # (1, 1, D, C)
        zhat = zhat * gate
        # IDCT: inverse of orthonormal DCT-II is W^T.
        z_ifft = torch.einsum("nk,bxkc->bxnc", W, zhat)
        z_out = z_ifft + self.ffn(self.norm(z))
        return z_out


# -----------------------------------------------------------------------------
# Top-level slice encoder.
# -----------------------------------------------------------------------------
@dataclass
class EncoderConfig:
    img_size: int = 224
    patch_size: int = 4
    in_chans: int = 3
    depths: Tuple[int, int, int, int] = (2, 2, 9, 2)
    dims: Tuple[int, int, int, int] = (96, 192, 384, 768)
    drop_path_rate: float = 0.1
    mlp_ratio: float = 4.0
    pretrained_path: Optional[str] = None
    gradient_checkpointing: bool = True
    max_d: int = 256


class HeatConductionSliceEncoder(nn.Module):
    """Φ: per-slice signed S-HCO (vHeat-Tiny) + 1-D HCO Ψ along D.

    Forward vol: (B, C_in, D, H, W). Returns feats: (B, D, N_L, C_L).
    """

    def __init__(self, cfg: EncoderConfig):
        super().__init__()
        self.cfg = cfg
        self.backbone = vHeat(
            patch_size=cfg.patch_size,
            in_chans=cfg.in_chans,
            num_classes=0,
            depths=list(cfg.depths),
            dims=list(cfg.dims),
            drop_path_rate=cfg.drop_path_rate,
            mlp_ratio=cfg.mlp_ratio,
            img_size=cfg.img_size,
        )
        self.backbone.classifier = nn.Identity()

        c_l = cfg.dims[-1]
        self.psi = Heat1D(dim=c_l, max_d=cfg.max_d, mlp_ratio=2.0)
        self.norm_out = nn.LayerNorm(c_l)

        if cfg.pretrained_path:
            self.load_vheat_pretrained(cfg.pretrained_path)

    def load_vheat_pretrained(self, path: str) -> None:
        sd = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(sd, dict) and "model" in sd:
            sd = sd["model"]
        own = self.backbone.state_dict()
        loaded, skipped = 0, 0
        for k, v in sd.items():
            if k.startswith("classifier"):
                skipped += 1; continue
            if k in own and own[k].shape == v.shape:
                own[k] = v; loaded += 1
            else:
                skipped += 1
        self.backbone.load_state_dict(own)
        print(f"[encoder] vHeat pretrained: loaded {loaded}, skipped {skipped}")

    def forward(self, vol: torch.Tensor) -> torch.Tensor:
        b, c_in, d, h, w = vol.shape
        x = vol.permute(0, 2, 1, 3, 4).contiguous().view(b * d, c_in, h, w)
        if self.cfg.gradient_checkpointing and self.training:
            x = torch.utils.checkpoint.checkpoint(
                self.backbone.forward_features, x, use_reentrant=False
            )
        else:
            x = self.backbone.forward_features(x)
        # x: (B*D, C_L, H_L, W_L)
        c_l, h_l, w_l = x.shape[1], x.shape[2], x.shape[3]
        x = x.view(b, d, c_l, h_l * w_l).permute(0, 3, 1, 2).contiguous()  # (B, N_L, D, C_L)
        x = self.psi(x)
        feats = x.permute(0, 2, 1, 3).contiguous()                          # (B, D, N_L, C_L)
        feats = self.norm_out(feats)
        return feats
