"""Stage III — Anomaly-Guided Report Generation (Sec. 3.4).

Eq. 13:  s_{k,j} = α_j · cos(q̃_{k,j}, t_k) / τ_o
Eq. 14:  [π_{k,0}, π_{k,1..D}] = sparsemax( a_k, s_{k,1..D} )      (j=0 abstain)
Eq. 15:  q̄_{k,j} = π_{k,j} · q̃_{k,j}
Eq. 17:  P = ‖_k [d_beg_k; T_k; d_end_k]   with T_k = {q̄_{k,j}: j=1..D}

LLM ΦLLM (Qwen3-8B + LoRA) consumes P + report tokens via inputs_embeds.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model


# -----------------------------------------------------------------------------
# Sparsemax (Martins & Astudillo 2016).
# -----------------------------------------------------------------------------
def sparsemax(z: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Differentiable sparsemax along `dim`. z: any real tensor."""
    z = z.transpose(dim, -1)
    orig_shape = z.shape
    z_flat = z.reshape(-1, orig_shape[-1])
    z_sorted, _ = torch.sort(z_flat, descending=True, dim=-1)
    cssv = torch.cumsum(z_sorted, dim=-1) - 1.0
    rng = torch.arange(1, z_flat.size(-1) + 1, device=z.device, dtype=z.dtype)
    cond = (z_sorted * rng > cssv)
    rho = cond.long().sum(dim=-1, keepdim=True).clamp(min=1)
    tau = cssv.gather(-1, rho - 1) / rho.to(z.dtype)
    p = (z_flat - tau).clamp(min=0.0)
    return p.view(orig_shape).transpose(dim, -1)


# -----------------------------------------------------------------------------
# Decoder.
# -----------------------------------------------------------------------------
@dataclass
class DecoderConfig:
    hf_id: str = ""
    d: int = 768
    init_tau_o: float = 0.5
    init_abstain: float = 1.0
    organ_bag_dropout: float = 0.0       # set per-step from training schedule
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda:
        ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    gradient_checkpointing: bool = True
    max_prefix_tokens: int = 4096
    attn_implementation: str = "flash_attention_2"
    bf16: bool = True


class ReportDecoder(nn.Module):
    def __init__(self, cfg: DecoderConfig, num_organs: int):
        super().__init__()
        self.cfg = cfg
        self.K = num_organs
        dtype = torch.bfloat16 if cfg.bf16 else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.hf_id, trust_remote_code=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        kwargs = dict(torch_dtype=dtype)
        if cfg.attn_implementation:
            kwargs["attn_implementation"] = cfg.attn_implementation
        base = AutoModelForCausalLM.from_pretrained(cfg.hf_id, trust_remote_code=True, **kwargs)
        if cfg.gradient_checkpointing:
            base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

        lcfg = LoraConfig(
            r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
            target_modules=cfg.lora_target_modules, bias="none", task_type="CAUSAL_LM",
        )
        self.llm = get_peft_model(base, lcfg)
        h = base.config.hidden_size
        self.h_dim = h

        # Project Q-token (d) -> LLM hidden h.
        self.prefix_proj = nn.Linear(cfg.d, h, bias=False)
        # Project text anchor to LLM hidden, plus a shared learnable bias for begin/end delimiters.
        self.t_to_h = nn.Linear(cfg.d, h, bias=False)
        self.delim_beg_bias = nn.Parameter(torch.zeros(h))
        self.delim_end_bias = nn.Parameter(torch.zeros(h))
        nn.init.trunc_normal_(self.delim_beg_bias, std=0.02)
        nn.init.trunc_normal_(self.delim_end_bias, std=0.02)

        # Learnable τ_o (positive via softplus).
        self.log_tau_o = nn.Parameter(torch.tensor(0.0))
        # Per-organ abstain anchor a_k.
        self.abstain = nn.Parameter(torch.full((num_organs,), cfg.init_abstain))

    @property
    def tau_o(self) -> torch.Tensor:
        return F.softplus(self.log_tau_o) + 1e-3

    def compute_pi(self, qt: torch.Tensor, alpha: torch.Tensor, t: torch.Tensor):
        """Sparsemax-based slice weights π_{k,j}.

        qt:    (B, D, K, d)   organ-conditioned slice evidence
        alpha: (B, D)         slice anomaly probability
        t:     (K, d)         text anchors

        Returns:
          pi_slices: (B, K, D)   weights for j=1..D (after stripping abstain mass)
          pi_abstain:(B, K)      mass on j=0 (abstain)
        """
        b, D, K, d = qt.shape
        # cos(q̃_{k,j}, t_k)
        qn = F.normalize(qt, dim=-1)
        tn = F.normalize(t, dim=-1)                           # (K, d)
        cos = torch.einsum("bdkc,kc->bdk", qn, tn)            # (B, D, K)
        s = alpha.unsqueeze(-1) * (cos / self.tau_o)          # (B, D, K)
        s = s.permute(0, 2, 1)                                # (B, K, D)
        # Augment with abstain anchor a_k at position 0.
        a = self.abstain.view(1, K, 1).expand(b, -1, -1)      # (B, K, 1)
        z = torch.cat([a, s], dim=-1)                         # (B, K, D+1)
        pi = sparsemax(z, dim=-1)                             # (B, K, D+1)
        return pi[..., 1:], pi[..., 0]

    def build_prefix(self, qt: torch.Tensor, pi_slices: torch.Tensor, t: torch.Tensor,
                     organ_drop_mask: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Construct visual prefix P (Eq. 17).

        qt:        (B, D, K, d)
        pi_slices: (B, K, D)
        t:         (K, d)
        organ_drop_mask: (B, K) in {0,1}; 1=keep this organ's bag this step
                         (None -> all kept).
        Returns prefix_emb (B, L_p, h) and prefix_mask (B, L_p).
        """
        b, D, K, d = qt.shape
        h = self.h_dim
        # Weighted slice evidence: q̄_{k,j} = π_{k,j} q̃_{k,j}  (B, K, D, d)
        qbar = pi_slices.unsqueeze(-1) * qt.permute(0, 2, 1, 3)
        Tk = self.prefix_proj(qbar)                          # (B, K, D, h)

        # Per-organ delimiters.
        t_h = self.t_to_h(t)                                 # (K, h)
        d_beg = (t_h + self.delim_beg_bias).unsqueeze(0).expand(b, -1, -1)  # (B, K, h)
        d_end = (t_h + self.delim_end_bias).unsqueeze(0).expand(b, -1, -1)

        # Stack [d_beg; T_k; d_end] per organ -> (B, K, D+2, h)
        per_organ = torch.cat([d_beg.unsqueeze(2), Tk, d_end.unsqueeze(2)], dim=2)
        # Per-organ keep mask.
        if organ_drop_mask is None:
            keep = torch.ones(b, K, device=qt.device, dtype=qt.dtype)
        else:
            keep = organ_drop_mask.to(qt.dtype)
        per_organ = per_organ * keep.unsqueeze(-1).unsqueeze(-1)

        prefix = per_organ.reshape(b, K * (D + 2), h)
        prefix_mask = keep.unsqueeze(-1).expand(-1, -1, D + 2).reshape(b, K * (D + 2))
        if prefix.shape[1] > self.cfg.max_prefix_tokens:
            prefix = prefix[:, : self.cfg.max_prefix_tokens]
            prefix_mask = prefix_mask[:, : self.cfg.max_prefix_tokens]
        return prefix, prefix_mask

    def forward(self, prefix_emb, prefix_mask, report_ids, report_mask):
        token_emb = self.llm.get_input_embeddings()(report_ids)
        inputs_embeds = torch.cat([prefix_emb.to(token_emb.dtype), token_emb], dim=1)
        attn_mask = torch.cat([prefix_mask.to(report_mask.dtype), report_mask], dim=1)
        ignore = torch.full_like(prefix_mask.long(), -100)
        labels_y = report_ids.masked_fill(report_mask == 0, -100)
        labels = torch.cat([ignore, labels_y], dim=1)
        out = self.llm(inputs_embeds=inputs_embeds, attention_mask=attn_mask,
                       labels=labels, return_dict=True)
        return out.logits, out.loss

    @torch.no_grad()
    def generate(self, prefix_emb, prefix_mask, max_new_tokens: int = 256):
        bos = self.tokenizer.bos_token_id or self.tokenizer.eos_token_id
        b = prefix_emb.shape[0]
        seed_ids = torch.full((b, 1), bos, dtype=torch.long, device=prefix_emb.device)
        seed_emb = self.llm.get_input_embeddings()(seed_ids)
        inputs_embeds = torch.cat([prefix_emb.to(seed_emb.dtype), seed_emb], dim=1)
        attn_mask = torch.cat([prefix_mask, torch.ones_like(seed_ids)], dim=1)
        out = self.llm.generate(
            inputs_embeds=inputs_embeds, attention_mask=attn_mask,
            max_new_tokens=max_new_tokens, do_sample=False, num_beams=1,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        return self.tokenizer.batch_decode(out, skip_special_tokens=True)
