"""Top-level HeatRAD model — wires Encoder + Stage I/II/III per Sec. 3.

Single training objective is the autoregressive report loss (Eq. 18).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Dict

import torch
import torch.nn as nn

from .encoder import EncoderConfig, HeatConductionSliceEncoder
from .stage1_anomaly import Stage1Config, AnomalyHead
from .stage2_xattn import Stage2Config, OrganCrossAttention
from .stage3_decoder import DecoderConfig, ReportDecoder
from .text_anchor import TextAnchorConfig, TextAnchorEncoder


@dataclass
class HeatRADConfig:
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    stage1: Stage1Config = field(default_factory=Stage1Config)
    stage2: Stage2Config = field(default_factory=Stage2Config)
    decoder: DecoderConfig = field(default_factory=DecoderConfig)
    text_anchor: TextAnchorConfig = field(default_factory=TextAnchorConfig)


class HeatRAD(nn.Module):
    def __init__(self, cfg: HeatRADConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = HeatConductionSliceEncoder(cfg.encoder)
        self.head1 = AnomalyHead(cfg.stage1)
        self.xattn = OrganCrossAttention(cfg.stage2)
        self.text_anc = TextAnchorEncoder(cfg.text_anchor)
        self.decoder = ReportDecoder(cfg.decoder, num_organs=cfg.stage2.k)

    def forward(self, vol: torch.Tensor,
                report_ids: torch.Tensor,
                report_mask: torch.Tensor,
                organ_drop_mask: Optional[torch.Tensor] = None,
                ) -> Dict[str, torch.Tensor]:
        feats = self.encoder(vol)                                # (B, D, N, C_L)
        alpha, _f = self.head1(feats)                            # (B, D)
        t = self.text_anc()                                       # (K, d)
        qt = self.xattn(feats, t)                                # (B, D, K, d)
        pi_slices, pi_abstain = self.decoder.compute_pi(qt, alpha, t)
        prefix_emb, prefix_mask = self.decoder.build_prefix(qt, pi_slices, t, organ_drop_mask)
        _, loss = self.decoder(prefix_emb, prefix_mask, report_ids, report_mask)
        return dict(loss=loss, alpha=alpha.detach(), pi_abstain=pi_abstain.detach())

    @torch.no_grad()
    def generate(self, vol: torch.Tensor, max_new_tokens: int = 256):
        feats = self.encoder(vol)
        alpha, _ = self.head1(feats)
        t = self.text_anc()
        qt = self.xattn(feats, t)
        pi_slices, _ = self.decoder.compute_pi(qt, alpha, t)
        prefix_emb, prefix_mask = self.decoder.build_prefix(qt, pi_slices, t, None)
        return self.decoder.generate(prefix_emb, prefix_mask, max_new_tokens)
