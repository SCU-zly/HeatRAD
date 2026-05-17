"""HeatRAD inference + evaluation entry (matches refactored architecture).

Usage:
    cd $HEATRAD_ROOT
    source scripts/env.sh
    CUDA_VISIBLE_DEVICES=2 python -m heatrad.eval.run_eval \
        +ckpt=$HEATRAD_ROOT/runs/heatrad_h20/ckpt/last.pt \
        +eval.n_samples=50 \
        +eval.out=$HEATRAD_ROOT/runs/heatrad_h20/eval/val_50.json
"""
from __future__ import annotations
import json
import os
import time
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..data.dataset import ChestCTDataset, collate_fn
from ..train.trainer import build_model
from .metrics import (
    build_disorder_lexicon, ce_metrics, nlg_metrics, ratescore,
    radgraph_f1, chexbert_f1, strip_prompt_prefix,
)


def _load_ckpt(model: torch.nn.Module, path: str) -> int:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = ck.get("state_dict", ck)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[ckpt] loaded {path} step={ck.get('step', '?')} "
          f"missing={len(missing)} unexpected={len(unexpected)}")
    return int(ck.get("step", 0))


@torch.no_grad()
def _generate_one(model, vol, max_new_tokens: int = 256) -> str:
    feats = model.encoder(vol)
    alpha, _ = model.head1(feats)
    t = model.text_anc()
    qt = model.xattn(feats, t)
    pi_slices, _ = model.decoder.compute_pi(qt, alpha, t)
    prefix_emb, prefix_mask = model.decoder.build_prefix(qt, pi_slices, t, None)
    texts = model.decoder.generate(prefix_emb, prefix_mask, max_new_tokens=max_new_tokens)
    return texts[0]


@hydra.main(version_base=None, config_path="../../configs", config_name="default")
def main(cfg: DictConfig):
    eval_cfg = cfg.get("eval", {})
    ckpt_path = cfg.get("ckpt")
    assert ckpt_path is not None and os.path.exists(ckpt_path), f"need +ckpt=... ({ckpt_path})"

    split = eval_cfg.get("split", "val")
    n_samples = int(eval_cfg.get("n_samples", 0)) or None
    max_new = int(eval_cfg.get("max_new_tokens", 256))
    out_path = eval_cfg.get("out") or os.path.join(cfg.output_dir, "eval", f"{split}.json")
    do_rate = bool(eval_cfg.get("ratescore", True))

    device = torch.device("cuda")
    torch.manual_seed(cfg.seed)

    # Data.
    manifest = cfg.data.manifest_path
    max_d = int(cfg.data.vol_shape[0]) if "vol_shape" in cfg.data else 64
    ds = ChestCTDataset(manifest, split=split, organ_count=cfg.model.stage2.k, max_d=max_d)
    if n_samples:
        ds.df = ds.df.head(n_samples).reset_index(drop=True)
    print(f"[eval] {split}: {len(ds)} cases, ckpt={ckpt_path}")
    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=2,
                    collate_fn=collate_fn, pin_memory=True)

    # Model.
    model = build_model(cfg).to(device).eval()
    # Make sure gradient checkpointing is off for generation.
    if hasattr(model.decoder.llm, "gradient_checkpointing_disable"):
        model.decoder.llm.gradient_checkpointing_disable()
    _load_ckpt(model, ckpt_path)

    tok = model.decoder.tokenizer
    refs, hyps, case_ids = [], [], []
    t0 = time.time()
    for batch in tqdm(dl, desc=f"infer-{split}"):
        vol = batch["vol"].to(device, non_blocking=True)
        rep_ids = batch["report_ids"][0]
        # Ref: detokenize (skip the prompt prefix if present).
        ref_full = tok.decode(rep_ids, skip_special_tokens=True)
        ref = strip_prompt_prefix(ref_full)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            hyp = _generate_one(model, vol, max_new_tokens=max_new)
        refs.append(ref)
        hyps.append(strip_prompt_prefix(hyp))
        case_ids.append(batch["case_id"][0])
    elapsed = time.time() - t0
    print(f"[eval] generated {len(refs)} cases in {elapsed:.1f}s "
          f"({elapsed/max(1,len(refs)):.2f} s/case)")

    # Save raw cases first so a metrics crash doesn't lose generation results.
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    cases_only = [{"case_id": cid, "ref": r, "hyp": h}
                  for cid, r, h in zip(case_ids, refs, hyps)]
    raw_path = str(out_path).replace(".json", "_raw.json")
    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump({"cases": cases_only, "ckpt": ckpt_path}, f, indent=2, ensure_ascii=False)
    print(f"[eval] raw generations -> {raw_path}")

    # Metrics.
    print("[metrics] NLG...")
    nlg = nlg_metrics(hyps, refs)
    print("[metrics] CE (entity lexicon)...")
    train_dis = "/data/DaZhou_data/RadGenome-ChestCT/dataset/radgenome_files/train_case_disorders.csv"
    lex = build_disorder_lexicon(train_dis)
    print(f"  lexicon size: {len(lex)}")
    ce = ce_metrics(hyps, refs, lex)

    print("[metrics] RadGraph F1...")
    rg = radgraph_f1(hyps, refs)
    print("[metrics] CheXbert F1...")
    cb = chexbert_f1(hyps, refs)

    rate = None
    if do_rate:
        print("[metrics] RaTEScore (clinical, may take a while)...")
        rate = ratescore(hyps, refs)

    def _round(d):
        if d is None:
            return None
        return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in d.items()}

    summary = {
        "ckpt": ckpt_path,
        "split": split,
        "n_cases": len(refs),
        "wall_seconds": round(elapsed, 1),
        "NLG": {k: round(v, 4) for k, v in nlg.items()},
        "CE": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in ce.items()},
        "RadGraph": _round(rg),
        "CheXbert": _round(cb),
        "Clinical": {"RaTEScore": round(rate, 4) if rate is not None else None},
    }
    cases = [{"case_id": cid, "ref": r, "hyp": h}
             for cid, r, h in zip(case_ids, refs, hyps)]

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "cases": cases}, f, indent=2, ensure_ascii=False)
    print(f"[eval] saved -> {out_path}")
    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
