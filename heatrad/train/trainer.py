"""HeatRAD trainer (Sec. 3.4 / 4.2).

Single-phase joint training with the autoregressive report loss (Eq. 18) only.
Organ-bag dropout p_drop is linearly annealed 0.5 -> 0 over the first half of
training. AdamW(0.9,0.999), wd=0.05, lr_vis=1e-4, lr_lora=2e-4, 1-epoch warmup
+ cosine, 30 epochs default.
"""
from __future__ import annotations
import math
import os
import time
from pathlib import Path

import hydra
import torch
import torch.distributed as dist
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from ..data.dataset import ChestCTDataset, collate_fn
from ..models.encoder import EncoderConfig
from ..models.stage1_anomaly import Stage1Config
from ..models.stage2_xattn import Stage2Config
from ..models.text_anchor import TextAnchorConfig
from ..models.stage3_decoder import DecoderConfig
from ..models.heatrad import HeatRAD, HeatRADConfig
from ..utils.distributed import init_ddp, is_main, is_dist


def build_model(cfg: DictConfig) -> HeatRAD:
    encc = cfg.model.encoder
    s1c = cfg.model.stage1
    s2c = cfg.model.stage2
    tac = cfg.model.text_anchor
    dec = cfg.model.decoder
    hcfg = HeatRADConfig(
        encoder=EncoderConfig(
            img_size=encc.img_size, patch_size=encc.patch_size, in_chans=encc.in_chans,
            depths=tuple(encc.depths), dims=tuple(encc.dims),
            drop_path_rate=encc.drop_path_rate, mlp_ratio=encc.mlp_ratio,
            pretrained_path=(encc.pretrained_path if os.path.exists(encc.pretrained_path or "") else None),
            gradient_checkpointing=encc.gradient_checkpointing, max_d=encc.max_d,
        ),
        stage1=Stage1Config(c_l=s1c.c_l, init_temp=s1c.init_temp),
        stage2=Stage2Config(c_l=s2c.c_l, d=s2c.d, k=s2c.k),
        text_anchor=TextAnchorConfig(
            hf_id=tac.hf_id, organs=list(tac.organs), out_dim=tac.out_dim, pool=tac.pool,
            organ_template=tac.get("organ_template", "the {organ} region"),
        ),
        decoder=DecoderConfig(
            hf_id=dec.hf_id, d=dec.d,
            init_tau_o=dec.get("init_tau_o", 0.5),
            init_abstain=dec.get("init_abstain", 1.0),
            lora_r=dec.lora_r, lora_alpha=dec.lora_alpha, lora_dropout=dec.lora_dropout,
            lora_target_modules=list(dec.lora_target_modules),
            gradient_checkpointing=dec.gradient_checkpointing,
            max_prefix_tokens=dec.max_prefix_tokens,
            attn_implementation=dec.attn_implementation, bf16=dec.bf16,
        ),
    )
    return HeatRAD(hcfg)


def cosine_lr(step, total, warmup, peak):
    if step < warmup:
        return peak * (step + 1) / max(1, warmup)
    p = (step - warmup) / max(1, total - warmup)
    return peak * 0.5 * (1.0 + math.cos(math.pi * min(1.0, p)))


def organ_dropout_schedule(step: int, total: int, p_init: float = 0.5) -> float:
    half = max(1, total // 2)
    return max(0.0, p_init * (1.0 - step / half))


@hydra.main(version_base=None, config_path="../../configs", config_name="default")
def main(cfg: DictConfig):
    rank, ws, local = init_ddp()
    device = torch.device(f"cuda:{local}")
    torch.manual_seed(cfg.seed + rank)
    if is_main():
        print(OmegaConf.to_yaml(cfg))

    bsz = int(cfg.train.bsz_per_gpu)
    grad_accum = int(cfg.train.grad_accum)
    total_steps = int(cfg.train.total_steps)
    log_every = int(cfg.train.log_every)
    save_every = int(cfg.train.save_every)

    # Data.
    manifest = cfg.data.manifest_path
    if cfg.data.get("use_dev_subset", False) and os.path.exists(cfg.data.dev_manifest_path):
        manifest = cfg.data.dev_manifest_path
    max_d = int(cfg.data.vol_shape[0]) if "vol_shape" in cfg.data else None
    train_ds = ChestCTDataset(manifest, split="train", organ_count=cfg.model.stage2.k, max_d=max_d)
    sampler = DistributedSampler(train_ds, shuffle=True) if is_dist() else None
    train_dl = DataLoader(
        train_ds, batch_size=bsz, sampler=sampler, shuffle=(sampler is None),
        num_workers=cfg.data.num_workers, collate_fn=collate_fn,
        pin_memory=True, drop_last=True, persistent_workers=cfg.data.num_workers > 0,
    )

    # Model.
    model = build_model(cfg).to(device)
    if is_dist():
        model = DDP(model, device_ids=[local], find_unused_parameters=True,
                    gradient_as_bucket_view=True)

    # Param groups: separate LR for LoRA (decoder) vs vision modules.
    # Also exclude scalar gates (abstain, log_tau, log_tau_o) from weight decay
    # — wd would push the abstain anchor toward 0, fighting the model's choice.
    inner = model.module if hasattr(model, "module") else model
    vis_params, vis_no_wd, lora_params = [], [], []
    no_wd_keys = ("abstain", "log_tau", "log_tau_o")
    for n, p in inner.named_parameters():
        if not p.requires_grad:
            continue
        if "decoder.llm" in n:
            lora_params.append(p)
        elif any(k in n for k in no_wd_keys):
            vis_no_wd.append(p)
        else:
            vis_params.append(p)
    optim = torch.optim.AdamW(
        [
            {"params": vis_params, "lr": cfg.train.optim.lr_vis,
             "weight_decay": cfg.train.optim.weight_decay},
            {"params": vis_no_wd, "lr": cfg.train.optim.lr_vis,
             "weight_decay": 0.0},
            {"params": lora_params, "lr": cfg.train.optim.lr_lora,
             "weight_decay": cfg.train.optim.weight_decay},
        ],
        betas=tuple(cfg.train.optim.betas),
    )
    warmup = int(cfg.train.optim.warmup_ratio * total_steps)

    out_dir = Path(cfg.output_dir)
    if is_main():
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "ckpt").mkdir(exist_ok=True)
        (out_dir / "config.yaml").write_text(OmegaConf.to_yaml(cfg))

    # Optional resume.
    resume_path = cfg.get("resume_from", None)
    step = 0
    if resume_path and os.path.exists(resume_path):
        ck = torch.load(resume_path, map_location="cpu", weights_only=False)
        sd = ck.get("state_dict", ck)
        inner.load_state_dict(sd, strict=False)
        step = int(ck.get("step", 0))
        if is_main():
            print(f"[resume] loaded {resume_path} -> step {step}")
    accum = 0
    t0 = time.time()
    optim.zero_grad(set_to_none=True)
    train_iter = iter(train_dl)
    K = cfg.model.stage2.k

    while step < total_steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            if sampler is not None:
                sampler.set_epoch(step // max(1, len(train_dl)))
            train_iter = iter(train_dl)
            batch = next(train_iter)

        vol = batch["vol"].to(device, non_blocking=True)
        rid = batch["report_ids"].to(device, non_blocking=True)
        rmask = batch["report_mask"].to(device, non_blocking=True)
        # Organ-bag dropout mask: independently drop each organ bag.
        p_drop = organ_dropout_schedule(step, total_steps)
        if p_drop > 0:
            keep = (torch.rand(vol.shape[0], K, device=device) > p_drop).float()
            # Force-keep at least one organ per sample.
            none = keep.sum(dim=1, keepdim=True) == 0
            if none.any():
                fix = torch.zeros_like(keep)
                fix[:, 0] = 1.0
                keep = torch.where(none.expand_as(keep) > 0, fix, keep)
        else:
            keep = None

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = model(vol, rid, rmask, organ_drop_mask=keep)
            loss = out["loss"]

        (loss / grad_accum).backward()
        accum += 1
        if accum >= grad_accum:
            torch.nn.utils.clip_grad_norm_(
                [p for p in inner.parameters() if p.requires_grad], 1.0)
            base_lrs = [cfg.train.optim.lr_vis, cfg.train.optim.lr_vis, cfg.train.optim.lr_lora]
            for g, base_lr in zip(optim.param_groups, base_lrs):
                g["lr"] = cosine_lr(step, total_steps, warmup, base_lr)
            optim.step()
            optim.zero_grad(set_to_none=True)
            accum = 0
            step += 1

            if is_main() and step % log_every == 0:
                dt = time.time() - t0
                ips = step / max(1.0, dt)
                am = out["alpha"].max().item()
                amed = out["alpha"].median().item()
                pa = out["pi_abstain"].mean().item()
                print(f"[step {step:>6}/{total_steps}] {dt:7.1f}s loss={loss.item():.4f} "
                      f"p_drop={p_drop:.2f} α[max/med]={am:.3f}/{amed:.3f} "
                      f"abstain={pa:.3f} lr_vis={optim.param_groups[0]['lr']:.2e} "
                      f"lr_lora={optim.param_groups[2]['lr']:.2e} {ips:.2f}st/s")

            if is_main() and step % save_every == 0:
                _sd = (model.module if hasattr(model, "module") else model).state_dict()
                trim = {k: v for k, v in _sd.items() if v.requires_grad if isinstance(v, torch.Tensor)} if False else _sd
                ck = {"step": step, "state_dict": trim}
                torch.save(ck, out_dir / "ckpt" / f"step_{step:06d}.pt")
                torch.save(ck, out_dir / "ckpt" / "last.pt")
                print(f"  saved {out_dir / 'ckpt' / f'step_{step:06d}.pt'}")

    if is_dist():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
