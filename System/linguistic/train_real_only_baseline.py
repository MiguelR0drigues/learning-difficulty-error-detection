"""The decisive H2 experiment: real-only training FROM SCRATCH (no synthetic
pretraining), compared against the full curriculum on the identical real
evaluation split with a paired bootstrap CI.

Why this exists: the curriculum's 'stage 2 (real only)' CONTINUES from the
stage-1 synthetic checkpoint, so stage2-vs-final only measures the marginal
value of the combined stage 3 -- which multi-seed analysis showed to be
statistically null. The actual H2 question ('does synthetic data help under
real-data scarcity?') requires the comparison this script provides:

    deberta-v3-base --(real only, 8 epochs)--> scratch model
    vs.
    full curriculum final model (synthetic -> real -> combined)

Run per seed (minutes each; needs final_model[_seedS] to exist):
    SMOKE_TEST=0 SEED=0 python train_real_only_baseline.py
"""
from __future__ import annotations

import json
import os
import random

import numpy as np
import torch
import seqeval.metrics as sm
from transformers import (AutoTokenizer, AutoModelForTokenClassification,
                           TrainingArguments)

from train_linguistic_model import (BIODataset, WeightedTrainer, compute_class_weights,
                                     load_real, load_synthetic, LABEL_LIST, LABEL2ID,
                                     ID2LABEL)
from transformers import DataCollatorForTokenClassification
from eval_h2_ci import predict

SEED = int(os.environ.get("SEED", "0"))
# FINAL_TAG: suffix of the curriculum checkpoint to compare against, e.g.
# "_synth10k" to compare the (already trained) real-only model with the
# curriculum retrained at the 10^4 synthetic pool. Default "" = headline 10^6.
FINAL_TAG = os.environ.get("FINAL_TAG", "")
BASE = "microsoft/deberta-v3-base"


def splits(seed: int):
    rng = random.Random(seed)
    synthetic = load_synthetic("../data/linguistic/synthetic_corruption.json")
    real_path = "../data/linguistic/real_bio_combined.json"
    if not os.path.exists(real_path):
        real_path = "../data/linguistic/bea2019_bio_real.json"
    real = load_real(real_path)
    rng.shuffle(synthetic)
    rng.shuffle(real)
    cut = int(len(real) * 0.9)
    return real[:cut], real[cut:]


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(SEED)
    real_train, real_eval = splits(SEED)
    gold = [r["tags"] for r in real_eval]
    scratch_dir = f"./real_only_scratch_seed{SEED}"

    if not os.path.isdir(scratch_dir):
        print(f"Training real-only-from-scratch (seed {SEED}, {len(real_train)} sentences, 8 epochs)")
        tok = AutoTokenizer.from_pretrained(BASE)
        model = AutoModelForTokenClassification.from_pretrained(
            BASE, num_labels=len(LABEL_LIST), id2label=ID2LABEL, label2id=LABEL2ID,
            torch_dtype=torch.float32)
        args = TrainingArguments(output_dir="./out_scratch", per_device_train_batch_size=16,
                                  num_train_epochs=8, learning_rate=2e-5, warmup_ratio=0.1,
                                  max_grad_norm=1.0, bf16=False, logging_steps=20,
                                  save_strategy="no", report_to=[], disable_tqdm=True, seed=SEED)
        trainer = WeightedTrainer(model=model, args=args,
                                   train_dataset=BIODataset(real_train, tok),
                                   data_collator=DataCollatorForTokenClassification(tok),
                                   class_weights=compute_class_weights(real_train))
        trainer.train()
        nan_params = [n for n, p in model.named_parameters() if torch.isnan(p).any()]
        assert not nan_params, f"NaN params: {nan_params[:3]}"
        model.save_pretrained(scratch_dir); tok.save_pretrained(scratch_dir)

    final_dir = (f"./final_model_seed{SEED}" if SEED != 0 else "./final_model") + FINAL_TAG
    print(f"Evaluating {scratch_dir} vs {final_dir} on {len(real_eval)} real sentences")
    p_scratch = predict(scratch_dir, real_eval, device)
    p_final = predict(final_dir, real_eval, device)

    def f1(idx, pred):
        return sm.f1_score([gold[i] for i in idx], [pred[i] for i in idx])

    n = len(gold)
    all_idx = list(range(n))
    fs, ff = f1(all_idx, p_scratch), f1(all_idx, p_final)
    rng = np.random.default_rng(SEED)
    diffs = []
    for _ in range(2000):
        idx = rng.integers(0, n, n).tolist()
        diffs.append(f1(idx, p_final) - f1(idx, p_scratch))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    out = {"seed": SEED, "n": n, "final_tag": FINAL_TAG,
           "real_only_scratch_f1": round(fs, 4), "curriculum_final_f1": round(ff, 4),
           "diff": round(ff - fs, 4),
           "diff_ci95": [round(float(lo), 4), round(float(hi), 4)],
           "p_diff_leq_0": round(float(np.mean(np.array(diffs) <= 0)), 4)}
    print(json.dumps(out, indent=1))
    json.dump(out, open(f"../data/linguistic/h2_scratch_ci_seed{SEED}{FINAL_TAG}.json", "w"), indent=1)
