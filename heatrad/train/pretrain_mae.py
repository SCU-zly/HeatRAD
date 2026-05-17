"""Stage-2: self-supervised MAE/SimMIM pretraining of the heat-encoder.

Trains ONLY the HeatConductionSliceEncoder + a light reconstruction head on the
existing RadGenome volume cache (reports unused). Produces an encoder checkpoint
that Stage-3 (Chinese end-to-end training on Dazhou) warm-starts from instead of
the ImageNet vHeat weights.

Run (4x H20):
    source scripts/env.sh
    torchrun --nproc_per_node=4 -m heatrad.train.pretrain_mae \
        train=mae_pretrain run_name=zh_mae

Smoke test (1 GPU, few steps):
    CUDA_VISIBLE_DEVICES=2 python -m heatrad.train.pretrain_mae \
        train=mae_pretrain run_name=zh_mae_smoke \
        +train.total_steps=5 +train.save_every=5 train.bsz_per_gpu=1
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
from ..models.mae_head import MAEConfig, MAEPretrainer
from ..utils.distributed import init_ddp, is_main, is_dist


def build_mae(cfg: DictConfig) -> MAEPretrainer:
    e = cfg.model.encoder
    enc_cfg = EncoderConfig(
        img_size=e.img_size, patch_size=e.patch_size, in_chans=e.in_chans,
        depths=tuple(e.depths), dims=tuple(e.dims),
        drop_path_rate=e.drop_path_rate, mlp_ratio=e.mlp_ratio,
        pretrained_path=(e.pretrained_path if os.path.exists(e.pretrained_path or "") else None),
        gradient_checkpointing=True,          # B*D slices is memory-heavy
        max_d=e.max_d,
    )
    m = cfg.train.mae
    mae_cfg = MAEConfig(
        patch=int(m.patch), grid=int(m.grid),
        mask_ratio=float(m.mask_ratio), in_chans=e.in_chans,
        norm_pix_loss=bool(m.norm_pix_loss),
    )
    return MAEPretrainer(enc_cfg, mae_cfg)


def cosine_lr(step, total, warmup, peak):
    if step < warmup:
        return peak * (step + 1) / max(1, warmup)
    p = (step - warmup) / max(1, total - warmup)
    return peak * 0.5 * (1.0 + math.cos(math.pi * min(1.0, p)))


@hydra.main(version_base=None, config_path="../../configs", config_name="default")
def main(cfg: DictConfig):
    rank, ws, local = init_ddp()
    device = torch.device(f"cuda:{local}")
    torch.manual_seed(cfg.seed + rank)
    if is_main():
        print(OmegaConf.to_yaml(cfg.train))

    bsz = int(cfg.train.bsz_per_gpu)
    grad_accum = int(cfg.train.grad_accum)
    total_steps = int(cfg.train.total_steps)
    log_every = int(cfg.train.log_every)
    save_every = int(cfg.train.save_every)

    manifest = cfg.data.manifest_path
    max_d = int(cfg.data.vol_shape[0]) if "vol_shape" in cfg.data else None
    ds = ChestCTDataset(manifest, split="train", organ_count=cfg.model.stage2.k, max_d=max_d)
    sampler = DistributedSampler(ds, shuffle=True) if is_dist() else None
    dl = DataLoader(ds, batch_size=bsz, sampler=sampler, shuffle=(sampler is None),
                    num_workers=cfg.data.num_workers, collate_fn=collate_fn,
                    pin_memory=True, drop_last=True,
                    persistent_workers=cfg.data.num_workers > 0)

    model = build_mae(cfg).to(device)
    if is_dist():
        model = DDP(model, device_ids=[local], find_unused_parameters=False,
                    gradient_as_bucket_view=True)
    inner = model.module if hasattr(model, "module") else model

    no_wd, wd = [], []
    for n, p in inner.named_parameters():
        if not p.requires_grad:
            continue
        (no_wd if p.ndim <= 1 or "mask_token" in n else wd).append(p)
    optim = torch.optim.AdamW(
        [{"params": wd, "weight_decay": cfg.train.optim.weight_decay},
         {"params": no_wd, "weight_decay": 0.0}],
        lr=cfg.train.optim.lr, betas=tuple(cfg.train.optim.betas),
    )
    warmup = int(cfg.train.optim.warmup_ratio * total_steps)

    out_dir = Path(cfg.output_dir)
    if is_main():
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "ckpt").mkdir(exist_ok=True)
        (out_dir / "config.yaml").write_text(OmegaConf.to_yaml(cfg))

    step, accum = 0, 0
    t0 = time.time()
    optim.zero_grad(set_to_none=True)
    it = iter(dl)
    while step < total_steps:
        try:
            batch = next(it)
        except StopIteration:
            if sampler is not None:
                sampler.set_epoch(step // max(1, len(dl)))
            it = iter(dl)
            batch = next(it)

        vol = batch["vol"].to(device, non_blocking=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            loss, _, _ = model(vol)
        (loss / grad_accum).backward()
        accum += 1
        if accum >= grad_accum:
            torch.nn.utils.clip_grad_norm_(
                [p for p in inner.parameters() if p.requires_grad], 1.0)
            lr = cosine_lr(step, total_steps, warmup, cfg.train.optim.lr)
            for g in optim.param_groups:
                g["lr"] = lr
            optim.step()
            optim.zero_grad(set_to_none=True)
            accum = 0
            step += 1

            if is_main() and step % log_every == 0:
                dt = time.time() - t0
                print(f"[mae {step:>6}/{total_steps}] {dt:7.1f}s "
                      f"loss={loss.item():.4f} lr={lr:.2e} "
                      f"{step/max(1.0,dt):.2f}st/s", flush=True)

            if is_main() and step % save_every == 0:
                ck = {"step": step, "encoder": inner.encoder.state_dict()}
                torch.save(ck, out_dir / "ckpt" / f"mae_{step:06d}.pt")
                torch.save(ck, out_dir / "ckpt" / "encoder_last.pt")
                print(f"  saved {out_dir / 'ckpt' / f'mae_{step:06d}.pt'}", flush=True)

    if is_main():
        ck = {"step": step, "encoder": inner.encoder.state_dict()}
        torch.save(ck, out_dir / "ckpt" / "encoder_last.pt")
        print(f"[mae] done -> {out_dir / 'ckpt' / 'encoder_last.pt'}", flush=True)
    if is_dist():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
