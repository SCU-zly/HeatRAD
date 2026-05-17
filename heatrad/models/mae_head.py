"""SimMIM-style masked image modeling for the HeatConductionSliceEncoder.

vHeat is a hierarchical *conv* backbone, so ViT-style token dropping is awkward.
SimMIM (Xie et al. 2022) fits conv/hierarchical encoders cleanly: corrupt the
input with a learnable mask token at patch granularity, run the *full* encoder,
then regress the raw (already-normalized) pixels of the masked patches with a
light linear head. Masking is applied per slice independently.

Encoder downsamples 224 -> 14 (stride 16), so the natural patch is 16x16 and the
token grid is 14x14 = 196, matching the encoder's N_L.
"""
from __future__ import annotations
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import HeatConductionSliceEncoder, EncoderConfig


@dataclass
class MAEConfig:
    patch: int = 32          # input pixels per token (224/7, vHeat stride 32)
    grid: int = 7            # token grid per side (N_L = grid*grid = 49)
    mask_ratio: float = 0.6
    in_chans: int = 3
    norm_pix_loss: bool = True


class MAEPretrainer(nn.Module):
    """Encoder + SimMIM reconstruction head.

    forward(vol) -> (loss, pred, mask) where loss is L1 over masked patches.
    """

    def __init__(self, enc_cfg: EncoderConfig, mae_cfg: MAEConfig):
        super().__init__()
        self.encoder = HeatConductionSliceEncoder(enc_cfg)
        self.mae = mae_cfg
        p, c = mae_cfg.patch, mae_cfg.in_chans
        c_l = enc_cfg.dims[-1]
        self.patch_dim = c * p * p
        # Learnable mask token applied in *pixel* space (per channel).
        self.mask_token = nn.Parameter(torch.zeros(c))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        # Light linear decoder: per-token feature -> raw patch pixels.
        self.decoder = nn.Sequential(
            nn.LayerNorm(c_l),
            nn.Linear(c_l, self.patch_dim),
        )

    # --- patch helpers -----------------------------------------------------
    def _patchify(self, vol: torch.Tensor) -> torch.Tensor:
        """(B,C,D,H,W) -> (B, D, G*G, C*P*P)."""
        b, c, d, h, w = vol.shape
        g, p = self.mae.grid, self.mae.patch
        x = vol.view(b, c, d, g, p, g, p)                 # split H,W into G x P
        x = x.permute(0, 2, 3, 5, 1, 4, 6).contiguous()   # (B,D,G,G,C,P,P)
        return x.view(b, d, g * g, c * p * p)

    def _unpatchify(self, patches: torch.Tensor, d: int) -> torch.Tensor:
        """(B, D, G*G, C*P*P) -> (B,C,D,H,W)."""
        b = patches.shape[0]
        g, p, c = self.mae.grid, self.mae.patch, self.mae.in_chans
        x = patches.view(b, d, g, g, c, p, p)
        x = x.permute(0, 4, 1, 2, 5, 3, 6).contiguous()   # (B,C,D,G,P,G,P)
        return x.view(b, c, d, g * p, g * p)

    # --- forward -----------------------------------------------------------
    def forward(self, vol: torch.Tensor):
        b, c, d, h, w = vol.shape
        g = self.mae.grid
        n = g * g

        target = self._patchify(vol)                       # (B,D,N,patch_dim)

        # Per-slice random mask (1 = masked / reconstruct).
        mask = (torch.rand(b, d, n, device=vol.device) < self.mae.mask_ratio)
        # Guarantee at least one visible and one masked token per slice.
        allmask = mask.all(dim=-1, keepdim=True)
        allvis = (~mask).all(dim=-1, keepdim=True)
        mask = torch.where(allmask, mask & (torch.arange(n, device=vol.device) != 0), mask)
        mask = torch.where(allvis, mask | (torch.arange(n, device=vol.device) == 0), mask)

        # Build corrupted input: masked patches -> mask token (per channel).
        mt = self.mask_token.view(1, 1, 1, c, 1, 1)
        tgt_img = target.view(b, d, g, g, c, self.mae.patch, self.mae.patch)
        m = mask.view(b, d, g, g, 1, 1, 1)
        corrupt = torch.where(m.bool(), mt, tgt_img)
        corrupt = corrupt.permute(0, 4, 1, 2, 5, 3, 6).contiguous().view(b, c, d, h, w)

        feats = self.encoder(corrupt)                      # (B,D,N,C_L)
        pred = self.decoder(feats)                         # (B,D,N,patch_dim)

        if self.mae.norm_pix_loss:
            mu = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            tgt = (target - mu) / torch.sqrt(var + 1e-6)
        else:
            tgt = target

        loss_map = F.l1_loss(pred, tgt, reduction="none").mean(dim=-1)  # (B,D,N)
        m2 = mask.float()
        loss = (loss_map * m2).sum() / m2.sum().clamp(min=1.0)
        return loss, pred, mask
