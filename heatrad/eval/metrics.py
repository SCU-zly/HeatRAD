"""HeatRAD evaluation metrics:
  - NLG       : BLEU-1/4, METEOR, ROUGE-L
  - CE        : entity-level Precision / Recall / F1 (lexicon from train disorders)
  - RadGraph  : entity + relation F1 (Jain et al. 2021)
  - CheXbert  : 14-label classifier micro/macro F1
  - Clinical  : RaTEScore (MAGIC-AI4Med)
"""
from __future__ import annotations
import re
from typing import List, Dict, Set, Optional

from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer


# ---------------------------------------------------------------------------
# Prompt prefix stripping.
# ---------------------------------------------------------------------------
_PROMPT_PATTERNS = ("Report:\n", "Report:", "Findings:")


def strip_prompt_prefix(text: str) -> str:
    """Strip the training prompt prefix the model learned to reproduce.

    Training tokenization concatenates 'Generate a chest CT report ... Report:\n'
    in front of the actual report, so the LM regenerates that prefix verbatim.
    We anchor on 'Findings:' (the start of the real report); fall back to
    'Report:' if 'Findings:' is missing.
    """
    if not text:
        return text
    i = text.find("Findings:")
    if i >= 0:
        return text[i:].strip()
    j = text.find("Report:")
    if j >= 0:
        return text[j + len("Report:"):].lstrip("\n").strip()
    return text.strip()


# ---------------------------------------------------------------------------
# Lexicon-based entity extraction (CE).
# ---------------------------------------------------------------------------
def build_disorder_lexicon(csv_path: str) -> List[str]:
    import pandas as pd
    df = pd.read_csv(csv_path)
    phrases = set()
    for s in df["Disorders"].dropna().astype(str):
        for p in re.split(r"[,;]", s):
            p = p.strip().lower()
            if 3 <= len(p) <= 80 and p not in {"none", "normal", "n/a"}:
                phrases.add(p)
    return sorted(phrases, key=len, reverse=True)


def extract_entities(text: str, lexicon: List[str]) -> Set[str]:
    t = " " + text.lower() + " "
    return {ph for ph in lexicon if ph in t}


def ce_metrics(preds: List[str], refs: List[str], lexicon: List[str]) -> Dict[str, float]:
    """Micro-averaged entity Precision / Recall / F1."""
    tot_tp = tot_pred = tot_ref = 0
    for p, r in zip(preds, refs):
        pe = extract_entities(p, lexicon)
        re_ = extract_entities(r, lexicon)
        tot_tp += len(pe & re_)
        tot_pred += len(pe)
        tot_ref += len(re_)
    P = tot_tp / max(1, tot_pred)
    R = tot_tp / max(1, tot_ref)
    F = (2 * P * R / (P + R)) if (P + R) > 0 else 0.0
    return {"Precision": P, "Recall": R, "F1": F,
            "n_pred_ent": tot_pred, "n_ref_ent": tot_ref, "n_tp": tot_tp}


# ---------------------------------------------------------------------------
# NLG metrics.
# ---------------------------------------------------------------------------
def _tok(s: str) -> List[str]:
    return word_tokenize(s.lower())


def nlg_metrics(preds: List[str], refs: List[str]) -> Dict[str, float]:
    pred_tok = [_tok(p) for p in preds]
    ref_tok = [[_tok(r)] for r in refs]
    sm = SmoothingFunction().method1
    b1 = corpus_bleu(ref_tok, pred_tok, weights=(1, 0, 0, 0), smoothing_function=sm)
    b4 = corpus_bleu(ref_tok, pred_tok, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=sm)
    meteors = [meteor_score([_tok(r)], _tok(p)) for p, r in zip(preds, refs)]
    rs = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rouges = [rs.score(r, p)["rougeL"].fmeasure for p, r in zip(preds, refs)]
    return {
        "BLEU-1": b1,
        "BLEU-4": b4,
        "METEOR": sum(meteors) / max(1, len(meteors)),
        "ROUGE-L": sum(rouges) / max(1, len(rouges)),
    }


# ---------------------------------------------------------------------------
# RadGraph F1 (entity + relation extraction, Jain et al. 2021).
# ---------------------------------------------------------------------------
def radgraph_f1(preds: List[str], refs: List[str],
                reward_level: str = "all") -> Optional[Dict[str, float]]:
    """RadGraph F1. Returns dict of {simple, partial, complete} or None on failure.

    `simple` = entity-only F1; `partial` and `complete` add relation match.
    """
    try:
        from radgraph import F1RadGraph
    except Exception as e:
        print(f"[metrics] RadGraph unavailable: {e}")
        return None
    try:
        scorer = F1RadGraph(reward_level=reward_level)
        mean_reward, _, _, _ = scorer(hyps=preds, refs=refs)
        if isinstance(mean_reward, (list, tuple)):
            return {"simple": float(mean_reward[0]),
                    "partial": float(mean_reward[1]),
                    "complete": float(mean_reward[2])}
        return {"f1": float(mean_reward)}
    except Exception as e:
        print(f"[metrics] RadGraph failed: {e}")
        return None


# ---------------------------------------------------------------------------
# CheXbert F1 (14-label classifier, Smit et al. 2020).
# ---------------------------------------------------------------------------
def chexbert_f1(preds: List[str], refs: List[str]) -> Optional[Dict[str, float]]:
    """F1-CheXbert. Returns dict of {micro, macro, micro_5, macro_5}."""
    try:
        from f1chexbert import F1CheXbert
    except Exception as e:
        print(f"[metrics] CheXbert unavailable: {e}")
        return None
    try:
        scorer = F1CheXbert()
        accuracy, accuracy_per_sample, chexbert_all, chexbert_5 = scorer(
            hyps=preds, refs=refs)
        # F1CheXbert returns class-report dicts; pull weighted/micro/macro avgs.
        def _get(d, key, field):
            try:
                return float(d[key][field])
            except Exception:
                return None
        return {
            "micro_14": _get(chexbert_all, "micro avg", "f1-score"),
            "macro_14": _get(chexbert_all, "macro avg", "f1-score"),
            "micro_5":  _get(chexbert_5,   "micro avg", "f1-score"),
            "macro_5":  _get(chexbert_5,   "macro avg", "f1-score"),
            "accuracy": float(accuracy),
        }
    except Exception as e:
        print(f"[metrics] CheXbert failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Clinical: RaTEScore.
# ---------------------------------------------------------------------------
def ratescore(preds: List[str], refs: List[str]) -> Optional[float]:
    try:
        from RaTEScore import RaTEScore as _R
    except Exception as e:
        print(f"[metrics] RaTEScore unavailable: {e}")
        return None
    try:
        r = _R()
        s = r.compute_score(preds, refs)
        if isinstance(s, (list, tuple)):
            return float(sum(s) / max(1, len(s)))
        return float(s)
    except Exception as e:
        print(f"[metrics] RaTEScore failed: {e}")
        return None
