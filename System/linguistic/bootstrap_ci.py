"""Bootstrap confidence intervals for the Ch.5 benchmark results.

Modes:
  infer <bench.json> <out.json> <start> <end>   -- chunked neural inference,
      appends per-sentence predicted tag rows to <out.json> (list).
  ci <bench.json> <neural_preds.json> <baseline.json>  -- 95% bootstrap CIs
      (2000 resamples over sentences) for neural F1, baseline F1, and the
      PAIRED difference (neural - baseline) on the same resamples.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import seqeval.metrics as sm

LABELS = ["O", "B-PHONO", "I-PHONO", "B-ORTHO", "I-ORTHO", "B-SEG", "I-SEG"]


def infer(bench_path, out_path, start, end):
    import torch
    from transformers import AutoTokenizer, AutoModelForTokenClassification
    data = json.load(open(bench_path))[start:end]
    tok = AutoTokenizer.from_pretrained("./final_model")
    model = AutoModelForTokenClassification.from_pretrained("./final_model", torch_dtype=torch.float32)
    model.eval()
    preds_all = json.load(open(out_path)) if os.path.exists(out_path) else []
    B = 16
    for i in range(0, len(data), B):
        batch = data[i:i + B]
        enc = tok([r["tokens"] for r in batch], is_split_into_words=True,
                  truncation=True, max_length=64, padding=True, return_tensors="pt")
        with torch.no_grad():
            preds = model(**enc).logits.argmax(dim=-1)
        for j, r in enumerate(batch):
            wp = {}
            for wid, p in zip(enc.word_ids(batch_index=j), preds[j].tolist()):
                if wid is not None and wid not in wp:
                    wp[wid] = LABELS[p]
            preds_all.append({"idx": start + i + j,
                              "pred": [wp.get(k, "O") for k in range(len(r["tokens"]))]})
    json.dump(preds_all, open(out_path, "w"))
    print(f"infer [{start}:{end}] done, total stored: {len(preds_all)}")


def f1_of(indices, gold, pred):
    g = [gold[i] for i in indices]
    p = [pred[i] for i in indices]
    try:
        return sm.f1_score(g, p)
    except Exception:
        return 0.0


def ci(bench_path, neural_path, baseline_path, n_boot=2000, seed=0):
    data = json.load(open(bench_path))
    gold = [r["tags"] for r in data]
    neural_rows = sorted(json.load(open(neural_path)), key=lambda r: r["idx"])
    assert len(neural_rows) == len(gold), f"{len(neural_rows)} preds vs {len(gold)} gold"
    neural = [r["pred"] for r in neural_rows]
    baseline = json.load(open(baseline_path))["pred_tags"]
    assert len(baseline) == len(gold)

    n = len(gold)
    rng = np.random.default_rng(seed)
    idx_all = list(range(n))
    point_n, point_b = f1_of(idx_all, gold, neural), f1_of(idx_all, gold, baseline)
    ns, bs, ds = [], [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n).tolist()
        fn, fb = f1_of(idx, gold, neural), f1_of(idx, gold, baseline)
        ns.append(fn); bs.append(fb); ds.append(fn - fb)

    def pct(a):
        return (round(float(np.percentile(a, 2.5)), 4), round(float(np.percentile(a, 97.5)), 4))

    out = {"benchmark": os.path.basename(bench_path), "n_sentences": n, "n_boot": n_boot,
           "neural_f1": round(point_n, 4), "neural_ci95": pct(ns),
           "baseline_f1": round(point_b, 4), "baseline_ci95": pct(bs),
           "diff_point": round(point_n - point_b, 4), "diff_ci95": pct(ds),
           "p_diff_leq_0": round(float(np.mean(np.array(ds) <= 0)), 4)}
    print(json.dumps(out, indent=1))
    save = f"../data/linguistic/ci_{os.path.basename(bench_path).replace('.json','')}.json"
    json.dump(out, open(save, "w"), indent=1)
    print("saved ->", save)


if __name__ == "__main__":
    if sys.argv[1] == "infer":
        infer(sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]))
    else:
        ci(sys.argv[2], sys.argv[3], sys.argv[4])
