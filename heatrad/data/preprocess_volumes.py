"""Cache PNG slice stacks as .npz volumes.

For RadGenome-ChestCT, ct_slices/{case_id}/{0000..00ND}.png are pre-windowed
RGB at 224x224. We:
  1. Load + stack along D in lexicographic order.
  2. (Optionally) crop or pad to a target D (cfg.max_d).
  3. Normalize to float32 in [0, 1] (per-channel mean/std normalisation is
     deferred to the encoder's standard ImageNet stats).
  4. Permute to (C=3, D, H, W) and save as .npz with key 'vol'.
"""
from __future__ import annotations
import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm


# ImageNet mean/std (for vHeat-Tiny pretrained on ImageNet).
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def load_volume(slice_dir: str, max_d: int) -> np.ndarray:
    files = sorted(p for p in os.listdir(slice_dir) if p.endswith(".png"))
    if not files:
        raise RuntimeError(f"no png in {slice_dir}")
    if len(files) > max_d:
        # Center-crop along D.
        start = (len(files) - max_d) // 2
        files = files[start : start + max_d]
    imgs = []
    for f in files:
        im = Image.open(os.path.join(slice_dir, f)).convert("RGB")
        a = np.asarray(im, dtype=np.float32) / 255.0  # (H, W, 3)
        a = (a - _MEAN) / _STD
        imgs.append(a)
    vol = np.stack(imgs, axis=0)                     # (D, H, W, 3)
    vol = vol.transpose(3, 0, 1, 2).astype(np.float32)  # (3, D, H, W)
    return vol


def _process_one(args):
    cid, slice_dir, out, max_d = args
    if os.path.exists(out):
        return cid, out, None
    try:
        vol = load_volume(slice_dir, max_d=max_d)
        np.savez_compressed(out, vol=vol)
        return cid, out, None
    except Exception as e:
        return cid, None, str(e)


def main(manifest: str, out_dir: str, max_d: int = 96, workers: int = 1) -> None:
    df = pd.read_parquet(manifest)
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    tasks = [
        (r["case_id"], r["slice_dir"], os.path.join(out_dir, f"{r['case_id']}.npz"), max_d)
        for _, r in df.iterrows()
    ]

    cid_to_path = {}
    if workers <= 1:
        for t in tqdm(tasks, desc="vol_npz"):
            cid, out, err = _process_one(t)
            if err:
                print(f"[warn] {cid}: {err}")
            cid_to_path[cid] = out
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(_process_one, t) for t in tasks]
            for f in tqdm(as_completed(futures), total=len(futures), desc="vol_npz"):
                cid, out, err = f.result()
                if err:
                    print(f"[warn] {cid}: {err}")
                cid_to_path[cid] = out

    df["vol_npz"] = df["case_id"].map(cid_to_path)
    df.to_parquet(manifest, index=False)
    print(f"cached {sum(1 for p in cid_to_path.values() if p)} / {len(tasks)} volumes")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--max_d", type=int, default=96)
    p.add_argument("--workers", type=int, default=1)
    args = p.parse_args()
    main(args.manifest, args.out_dir, args.max_d, args.workers)
