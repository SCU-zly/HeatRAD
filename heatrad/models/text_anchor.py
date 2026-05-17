"""Text-anchored organ prototypes (Sec. 3.3).

Frozen biomedical text encoder; t_k = E_txt(w_k) computed once at init and
cached as a buffer. No LoRA, no gradients into E_txt.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


@dataclass
class TextAnchorConfig:
    hf_id: str = ""
    organs: List[str] = field(default_factory=list)
    out_dim: int = 768
    pool: str = "cls"
    organ_template: str = "the {organ} region"


class TextAnchorEncoder(nn.Module):
    def __init__(self, cfg: TextAnchorConfig):
        super().__init__()
        self.cfg = cfg
        tok = AutoTokenizer.from_pretrained(cfg.hf_id)
        backbone = AutoModel.from_pretrained(cfg.hf_id)
        backbone.eval()
        for p in backbone.parameters():
            p.requires_grad = False

        prompts = [cfg.organ_template.format(organ=w.replace("_", " ")) for w in cfg.organs]
        enc = tok(prompts, padding=True, return_tensors="pt")

        with torch.no_grad():
            out = backbone(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
            h = out.last_hidden_state
            if cfg.pool == "cls":
                v = h[:, 0]
            else:
                m = enc["attention_mask"].unsqueeze(-1).float()
                v = (h * m).sum(dim=1) / m.sum(dim=1).clamp(min=1)
        h_dim = backbone.config.hidden_size
        self.proj = nn.Linear(h_dim, cfg.out_dim, bias=False)
        self.register_buffer("t_pre", v, persistent=False)
        # Discard backbone — we only need the cached embeddings.
        del backbone

    def forward(self) -> torch.Tensor:
        # (K, out_dim). The proj is the only learnable piece (small adapter).
        return self.proj(self.t_pre)
