"""Organ-attribution multi-label metrics.

We score the volume-level β_k against y_o per case and report:
  - per-organ precision / recall / F1
  - macro-averaged F1
  - micro-averaged F1
  - subset accuracy (exact multi-label match)

This is the central CLAIM table for HeatRAD: organ attribution F1 vs
plain-LLM baselines should be visibly better.
"""
from __future__ import annotations
from typing import Dict, List

import numpy as np


def organ_metrics(beta_k: np.ndarray, y_o: np.ndarray, organs: List[str],
                   tau_beta: float = 0.5, eps: float = 1e-12) -> Dict[str, float]:
    """beta_k: (N, K) probabilities; y_o: (N, K) {0,1}; organs: K names."""
    assert beta_k.shape == y_o.shape, (beta_k.shape, y_o.shape)
    N, K = beta_k.shape
    pred = (beta_k > tau_beta).astype(np.float32)
    y = y_o.astype(np.float32)

    out: Dict[str, float] = {}

    # Per-organ.
    tp = (pred * y).sum(0)
    fp = (pred * (1 - y)).sum(0)
    fn = ((1 - pred) * y).sum(0)
    p = tp / (tp + fp + eps)
    r = tp / (tp + fn + eps)
    f = 2 * p * r / (p + r + eps)
    for k, name in enumerate(organs):
        out[f"P/{name}"] = 100 * float(p[k])
        out[f"R/{name}"] = 100 * float(r[k])
        out[f"F1/{name}"] = 100 * float(f[k])

    # Macro / micro.
    out["Macro-F1"] = 100 * float(f.mean())
    out["Macro-P"] = 100 * float(p.mean())
    out["Macro-R"] = 100 * float(r.mean())

    tp_m = (pred * y).sum()
    fp_m = (pred * (1 - y)).sum()
    fn_m = ((1 - pred) * y).sum()
    p_m = tp_m / (tp_m + fp_m + eps)
    r_m = tp_m / (tp_m + fn_m + eps)
    f_m = 2 * p_m * r_m / (p_m + r_m + eps)
    out["Micro-F1"] = 100 * float(f_m)
    out["Micro-P"] = 100 * float(p_m)
    out["Micro-R"] = 100 * float(r_m)

    # Subset / sample-level.
    out["Subset-Acc"] = 100 * float((pred == y).all(axis=1).mean())
    return out
