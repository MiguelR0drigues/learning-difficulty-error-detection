"""Paired bootstrap CI for the H2 difference (final vs stage-2 model on the
identical real evaluation split). Run on the GPU machine AFTER a training
run with SAVE_STAGE2=1 (see RUN_ON_GPU.md):

    SEED=1 python eval_h2_ci.py

Loads ./stage2_model_seed{SEED} and ./final_model_seed{SEED} (or ./final_model
for seed 0), reproduces the exact real_eval split for that seed, stores
per-sentence predictions for both models, and reports the paired 95% CI of
the span-F1 difference over 2000 sentence resamples.
"""
from __future__ import annotations

import json
import os
import random

import numpy as np
import torch
import seqeval.metrics as sm
from transformers import AutoTokenizer, AutoModelForTokenClassification

from train_linguistic_model import load_real, load_synthetic

SEED = int(os.environ.get("SEED", "0"))
LABELS = ["O", "B-PHONO", "I-PHONO", "B-ORTHO", "I-ORTHO", "B-SEG", "I-SEG"]


def real_eval_split(seed: int):
    rng = random.Random(seed)
    synthetic = load_synthetic("../data/linguistic/synthetic_corruption.json")
    real_path = "../data/linguistic/real_bio_combined.json"
    if not os.path.exists(real_path):
        real_path = "../data/linguistic/bea2019_bio_real.json"
    real = load_real(real_path)
    rng.shuffle(synthetic)  # consumes rng state identically to training
    rng.shuffle(real)
    return real[int(len(real) * 0.9):]


def predict(model_dir: str, rows: list, device: str):
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForTokenClassification.from_pretrained(model_dir, torch_dtype=torch.float32)
    model.eval().to(device)
    preds = []
    B = 32
    for i in range(0, len(rows), B):
        batch = rows[i:i + B]
        enc = tok([r["tokens"] for r in batch], is_split_into_words=True,
                  truncation=True, max_length=64, padding=True, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(**enc).logits.argmax(dim=-1)
        for j, r in enumerate(batch):
            wp = {}
            for wid, p in zip(enc.word_ids(batch_index=j), out[j].tolist()):
                if wid is not None and wid not in wp:
                    wp[wid] = LABELS[p]
            preds.append([wp.get(k, "O") for k in range(len(r["tokens"]))])
    return preds


def eval_rows(which: str):
    """Which held-out set to score the paired H2 difference on.
    'real' (default): the seed's real held-out split (same as the headline
    H2 CIs -- but note it was also used to *pick* the synthetic scale, so a
    positive here is peeked-at). 'fce': the FCE test set (383 sent.), never
    touched by scale selection -> the honest confirmation set for an
    H2-at-optimal-scale claim."""
    if which == "fce":
        return load_real("../data/linguistic/fce_test_bio.json")
    return real_eval_split(SEED)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    RUN_TAG = os.environ.get("RUN_TAG", "")
    EVAL_SET = os.environ.get("EVAL_SET", "real")
    rows = eval_rows(EVAL_SET)
    gold = [r["tags"] for r in rows]
    stage2_dir = f"./stage2_model_seed{SEED}{RUN_TAG}"
    if RUN_TAG:
        final_dir = f"./final_model_seed{SEED}{RUN_TAG}"
    else:
        final_dir = f"./final_model_seed{SEED}" if SEED != 0 else "./final_model"
    print(f"seed={SEED} tag='{RUN_TAG}' eval_set={EVAL_SET}: {len(rows)} eval sentences | {stage2_dir} vs {final_dir}")
    p2 = predict(stage2_dir, rows, device)
    p3 = predict(final_dir, rows, device)

    def f1(idx, pred):
        return sm.f1_score([gold[i] for i in idx], [pred[i] for i in idx])

    n = len(gold)
    all_idx = list(range(n))
    f2, f3 = f1(all_idx, p2), f1(all_idx, p3)
    rng = np.random.default_rng(SEED)
    diffs = []
    for _ in range(2000):
        idx = rng.integers(0, n, n).tolist()
        diffs.append(f1(idx, p3) - f1(idx, p2))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    out = {"seed": SEED, "run_tag": RUN_TAG, "eval_set": EVAL_SET, "n": n,
           "stage2_f1": round(f2, 4), "final_f1": round(f3, 4),
           "diff": round(f3 - f2, 4), "diff_ci95": [round(float(lo), 4), round(float(hi), 4)],
           "p_diff_leq_0": round(float(np.mean(np.array(diffs) <= 0)), 4)}
    print(json.dumps(out, indent=1))
    suffix = f"{RUN_TAG}_{EVAL_SET}" if (RUN_TAG or EVAL_SET != "real") else ""
    json.dump(out, open(f"../data/linguistic/h2_paired_ci_seed{SEED}{suffix}.json", "w"), indent=1)
