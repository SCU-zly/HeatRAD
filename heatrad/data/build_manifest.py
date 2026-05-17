"""Build manifest.parquet for RadGenome-ChestCT.

Reads chat_radgenome_train.json + chat_radgenome_val.json and produces a
single parquet with columns: case_id, split, num_slices, slice_dir,
report (concat findings+impression text). The original split between
'train' and 'val' is kept; we hold out a fixed 5% slice of train as
'dev' so dev_4090 runs converge fast.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path

import pandas as pd


def _extract_report(item: dict) -> str:
    convs = item.get("conversations", [])
    for c in convs:
        if c.get("from") == "gpt":
            return c.get("value", "").strip()
    return ""


def main(root: str, out: str, dev_subset_size: int = 200, dev_seed: int = 42) -> None:
    rows = []
    for split, j in (
        ("train", os.path.join(root, "chat_radgenome_train.json")),
        ("val",   os.path.join(root, "chat_radgenome_val.json")),
    ):
        if not os.path.exists(j):
            print(f"[warn] missing {j}; skipping")
            continue
        data = json.load(open(j))
        ct_dir = os.path.join(root, "ct_slices")
        for it in data:
            cid = it["id"]
            sd = os.path.join(ct_dir, cid)
            if not os.path.isdir(sd):
                continue
            n = it.get("num_slices") or len(os.listdir(sd))
            rows.append(dict(
                case_id=cid,
                split=split,
                slice_dir=sd,
                num_slices=int(n),
                report=_extract_report(it),
            ))
    df = pd.DataFrame(rows)
    print(f"manifest: {len(df)} rows  ({(df['split']=='train').sum()} train, {(df['split']=='val').sum()} val)")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)

    # Build a small dev subset (sampled from train) for fast iteration.
    dev_path = os.path.join(os.path.dirname(out), "dev_subset.parquet")
    train_df = df[df["split"] == "train"]
    dev_df = train_df.sample(min(dev_subset_size, len(train_df)), random_state=dev_seed)
    val_df = df[df["split"] == "val"].sample(min(50, (df["split"] == "val").sum()), random_state=dev_seed)
    dev_full = pd.concat([dev_df, val_df]).reset_index(drop=True)
    dev_full.to_parquet(dev_path, index=False)
    print(f"dev_subset: {len(dev_full)} rows  ({len(dev_df)} train, {len(val_df)} val) -> {dev_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--dev_subset_size", type=int, default=200)
    args = p.parse_args()
    main(args.root, args.out, args.dev_subset_size)
