"""Derive y_o (per-case organ multi-hot, length K) from report text.

Strategy: look for any synonym from organ_vocab.yaml in each abnormality
sentence of the report. We define an abnormality sentence as one that
mentions any lesion keyword from RadGenome's organ_lesion_categories.json.
y_o[k] = 1 iff organ k is mentioned in at least one abnormality sentence.

Outputs one .npy per case: cache/organ_label/{case_id}.npy of shape (K,).
"""
from __future__ import annotations
import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import yaml


def load_vocab(path: str) -> Dict[str, List[str]]:
    cfg = yaml.safe_load(open(path))
    organs = cfg["organs"]
    syns = cfg["synonyms"]
    return organs, syns


_SENT_SPLIT = re.compile(r"(?<=[.;\n])\s+")


def split_sentences(text: str) -> List[str]:
    return [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]


def build_pattern(syns: List[str]) -> re.Pattern:
    parts = sorted({re.escape(s.lower()) for s in syns}, key=len, reverse=True)
    return re.compile(r"\b(?:" + "|".join(parts) + r")\b", re.IGNORECASE)


def extract_organ_label(
    report: str,
    organs: List[str],
    syn_patterns: Dict[str, re.Pattern],
    lesion_pattern: re.Pattern,
) -> np.ndarray:
    y = np.zeros(len(organs), dtype=np.float32)
    sents = split_sentences(report.lower())
    for s in sents:
        if not lesion_pattern.search(s):
            continue
        for k, organ in enumerate(organs):
            if syn_patterns[organ].search(s):
                y[k] = 1.0
    return y


def main(manifest: str, out_dir: str, vocab_path: str, lesion_json: str) -> None:
    organs, syns = load_vocab(vocab_path)
    syn_patterns = {o: build_pattern(syns.get(o, [o])) for o in organs}
    lesions = json.load(open(lesion_json))["lesion"]
    lesion_pattern = build_pattern(lesions)

    df = pd.read_parquet(manifest)
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    label_paths = []
    for _, r in df.iterrows():
        cid = r["case_id"]
        out = os.path.join(out_dir, f"{cid}.npy")
        if not os.path.exists(out):
            y = extract_organ_label(r["report"] or "", organs, syn_patterns, lesion_pattern)
            np.save(out, y)
        label_paths.append(out)
    df["organ_label"] = label_paths
    df.to_parquet(manifest, index=False)
    # Quick stats.
    Y = np.stack([np.load(p) for p in label_paths])
    print(f"organ_label: {Y.shape}, mean active organs/case = {Y.sum(1).mean():.2f}")
    print("top organ frequencies:")
    freqs = Y.mean(0)
    for k, o in enumerate(organs):
        print(f"  {o:<28} {freqs[k]*100:5.1f}%")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--vocab", required=True)
    p.add_argument("--lesion_json", required=True)
    args = p.parse_args()
    main(args.manifest, args.out_dir, args.vocab, args.lesion_json)
