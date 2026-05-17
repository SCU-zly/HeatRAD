"""Tokenize reports with the Qwen3 tokenizer; cache as .pt per case."""
from __future__ import annotations
import argparse
import os
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoTokenizer


PROMPT_PREFIX = (
    "Generate a chest CT report with Findings and Impression based on the "
    "provided volumetric scan. Report:\n"
)


def main(manifest: str, tokenizer_path: str, out_dir: str, max_len: int = 512) -> None:
    tok = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    df = pd.read_parquet(manifest)
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    paths, lens = [], []
    for _, r in df.iterrows():
        cid = r["case_id"]
        out = os.path.join(out_dir, f"{cid}.pt")
        if not os.path.exists(out):
            text = (r["report"] or "").strip()
            full = PROMPT_PREFIX + text + tok.eos_token
            enc = tok(full, max_length=max_len, truncation=True, return_tensors="pt")
            torch.save(
                dict(
                    input_ids=enc["input_ids"][0].to(torch.long),
                    attention_mask=enc["attention_mask"][0].to(torch.long),
                ),
                out,
            )
        d = torch.load(out, weights_only=True)
        paths.append(out)
        lens.append(int(d["input_ids"].shape[0]))
    df["report_tok"] = paths
    df["report_len"] = lens
    df.to_parquet(manifest, index=False)
    print(f"tokenized {len(paths)} reports; mean len = {sum(lens)/len(lens):.1f}, max = {max(lens)}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--max_len", type=int, default=512)
    args = p.parse_args()
    main(args.manifest, args.tokenizer, args.out_dir, args.max_len)
