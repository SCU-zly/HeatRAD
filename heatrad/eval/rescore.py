"""Re-score a saved {cases:[{case_id,ref,hyp}, ...]} JSON without re-running inference.

Usage:
    python -m heatrad.eval.rescore <raw_json> <out_json>
"""
from __future__ import annotations
import json
import sys

from .metrics import (
    build_disorder_lexicon, ce_metrics, nlg_metrics, ratescore,
    radgraph_f1, chexbert_f1, strip_prompt_prefix,
)


def main(in_path: str, out_path: str):
    with open(in_path) as f:
        d = json.load(f)
    cases = d["cases"]
    refs = [strip_prompt_prefix(c["ref"]) for c in cases]
    hyps = [strip_prompt_prefix(c["hyp"]) for c in cases]
    case_ids = [c["case_id"] for c in cases]
    print(f"[rescore] {len(refs)} cases")
    print("hyp[0] (post-strip, first 200):", repr(hyps[0][:200]))

    print("[metrics] NLG...")
    nlg = nlg_metrics(hyps, refs)
    print("[metrics] CE...")
    lex = build_disorder_lexicon(
        "/data/DaZhou_data/RadGenome-ChestCT/dataset/radgenome_files/train_case_disorders.csv")
    ce = ce_metrics(hyps, refs, lex)
    print("[metrics] RadGraph...")
    rg = radgraph_f1(hyps, refs)
    print("[metrics] CheXbert...")
    cb = chexbert_f1(hyps, refs)
    print("[metrics] RaTEScore...")
    rate = ratescore(hyps, refs)

    def _round(d):
        if d is None: return None
        return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in d.items()}

    summary = {
        "n_cases": len(refs),
        "NLG": {k: round(v, 4) for k, v in nlg.items()},
        "CE": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in ce.items()},
        "RadGraph": _round(rg),
        "CheXbert": _round(cb),
        "Clinical": {"RaTEScore": round(rate, 4) if rate is not None else None},
    }
    cases_out = [{"case_id": cid, "ref": r, "hyp": h}
                 for cid, r, h in zip(case_ids, refs, hyps)]
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "cases": cases_out}, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
