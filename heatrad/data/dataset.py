"""ChestCTDataset — consumes the cached manifest produced by Phase 4."""
from __future__ import annotations
import os
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class ChestCTDataset(Dataset):
    """Reads pre-cached .npz volumes, .npy organ labels, and tokenized reports.

    Manifest columns required:
      vol_npz, organ_label, report_tok (paths) and 'split' (train/val/test).
    """

    def __init__(self, manifest_path: str, split: str = "train",
                 max_d: Optional[int] = None,
                 organ_count: int = 32):
        df = pd.read_parquet(manifest_path)
        df = df[df["split"] == split].reset_index(drop=True)
        # Required artifacts under the new (loss-free) pipeline are vol+report.
        for col in ("vol_npz", "report_tok"):
            df = df[df[col].notna() & df[col].apply(lambda p: isinstance(p, str) and os.path.exists(p))]
        self.df = df.reset_index(drop=True)
        self.max_d = max_d
        self.organ_count = organ_count

    def __len__(self):
        return len(self.df)

    def _load_vol(self, path: str) -> torch.Tensor:
        z = np.load(path)
        vol = z["vol"].astype(np.float32)            # (C_in, D, H, W)
        v = torch.from_numpy(vol)
        if self.max_d is not None and v.shape[1] > self.max_d:
            v = v[:, : self.max_d]
        return v

    def __getitem__(self, idx: int):
        r = self.df.iloc[idx]
        vol = self._load_vol(r["vol_npz"])               # (C_in, D, H, W)
        tok = torch.load(r["report_tok"], weights_only=True)
        report_ids = tok["input_ids"].long()
        report_mask = tok["attention_mask"].long()
        return dict(
            case_id=r["case_id"], vol=vol,
            report_ids=report_ids, report_mask=report_mask,
        )


def collate_fn(batch):
    """Pad volumes along D, reports along L_y."""
    max_d = max(b["vol"].shape[1] for b in batch)
    max_l = max(b["report_ids"].shape[0] for b in batch)
    vols, masks_d, ids, masks_y, cids = [], [], [], [], []
    for b in batch:
        v = b["vol"]
        pad_d = max_d - v.shape[1]
        if pad_d:
            v = torch.cat([v, v.new_zeros(v.shape[0], pad_d, v.shape[2], v.shape[3])], dim=1)
        vols.append(v)
        masks_d.append(torch.cat([torch.ones(b["vol"].shape[1]),
                                   torch.zeros(max_d - b["vol"].shape[1])]))
        rid = b["report_ids"]
        pad_l = max_l - rid.shape[0]
        if pad_l:
            rid = torch.cat([rid, rid.new_zeros(pad_l)])
            rm = torch.cat([b["report_mask"], b["report_mask"].new_zeros(pad_l)])
        else:
            rm = b["report_mask"]
        ids.append(rid)
        masks_y.append(rm)
        cids.append(b["case_id"])
    return dict(
        case_id=cids,
        vol=torch.stack(vols),                           # (B, C_in, D, H, W)
        slice_mask=torch.stack(masks_d).bool(),          # (B, D)
        report_ids=torch.stack(ids),                     # (B, L_y)
        report_mask=torch.stack(masks_y),                # (B, L_y)
    )
