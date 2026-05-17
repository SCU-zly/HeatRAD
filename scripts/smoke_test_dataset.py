"""End-to-end smoke test: real dev_subset -> encoder -> stage1 -> stage2."""
import torch
from torch.utils.data import DataLoader
from heatrad.data.dataset import ChestCTDataset, collate_fn
from heatrad.models.encoder import EncoderConfig, HeatConductionSliceEncoder
from heatrad.models.stage1_anomaly import Stage1Config, AnomalyHead
from heatrad.models.stage2_qformer import (
    Stage2Config, OrganQFormer, noisy_or, slice_aggregated_q,
)
from heatrad.models.losses import loss_att

import os

ROOT = os.environ["HEATRAD_ROOT"]


def main():
    ds = ChestCTDataset(f"{ROOT}/cache/dev_subset.parquet", split="train", organ_count=32)
    print(f"dataset: {len(ds)} cases")
    dl = DataLoader(ds, batch_size=1, shuffle=True, num_workers=0, collate_fn=collate_fn)
    batch = next(iter(dl))
    print(f"batch vol: {tuple(batch['vol'].shape)}, organ_label: {tuple(batch['organ_label'].shape)}")

    device = "cuda"
    enc = HeatConductionSliceEncoder(EncoderConfig(
        img_size=224, patch_size=4, in_chans=3,
        depths=(2, 2, 9, 2), dims=(96, 192, 384, 768),
        gradient_checkpointing=True, max_d=96,
    )).to(device)
    h1 = AnomalyHead(Stage1Config(c_l=768, topr=8)).to(device)
    qf = OrganQFormer(Stage2Config(c_l=768, d=768, k=32, n_layers=4, n_heads=12)).to(device)

    enc.train(); h1.train(); qf.train()
    vol = batch["vol"].to(device)
    y_o = batch["organ_label"].to(device)

    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        feats = enc(vol)
        print(f"feats: {tuple(feats.shape)}  (B,D,N_L,C_L)")
        alpha, idx, _ = h1(feats)
        a_top = torch.gather(alpha, 1, idx)
        print(f"alpha: {tuple(alpha.shape)}, top R = {idx.shape[1]}")
        q_jk, A_jk, b_jk = qf(feats, idx)
        b_k = noisy_or(a_top, b_jk)
        print(f"beta_k: {tuple(b_k.shape)}, expected (B={vol.shape[0]}, K=32)")
    # BCE outside autocast.
    L = loss_att(b_k.float(), y_o)
    L.backward()
    print(f"L_att = {L.item():.4f}")
    print(f"GPU mem: {torch.cuda.max_memory_allocated()/2**30:.2f} GiB")


if __name__ == "__main__":
    main()
