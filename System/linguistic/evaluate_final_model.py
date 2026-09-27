"""Post-training evaluation of ./final_model on REAL data only.

Why this exists: train_linguistic_model.py's stage-3 eval set is
synth_eval + real_eval, and synth_eval is ~145x larger -- so the stage-3
F1 (0.988 in the 2026-07-04 run) is dominated by synthetic data and CANNOT
be compared against stage 2's real-only F1 (0.60) to answer H2. The honest
H2 comparison is: stage-2 model vs stage-3 (final) model, both measured on
the SAME real eval split. Stage 2's number is already logged by the training
run; this script produces the final model's number on that identical split,
plus two held-out benchmarks the model never saw in any form:

  1. real_eval      -- same 10% real split as training (reproduced exactly)
  2. fce_test_bio   -- FCE test set (same domain as training's real data)
  3. holbrook_bio   -- struggling schoolchildren (DOMAIN SHIFT to the
                       thesis's actual target population)

Run on the GPU machine after training (takes a couple of minutes):
    python evaluate_final_model.py
"""
from __future__ import annotations

import json
import random

import torch
from transformers import (AutoTokenizer, AutoModelForTokenClassification,
                           TrainingArguments, Trainer, DataCollatorForTokenClassification)
import seqeval.metrics as seqeval_metrics

from train_linguistic_model import (BIODataset, compute_metrics, load_real,
                                     load_synthetic)

MODEL_DIR = "./final_model"


def reproduce_real_eval():
    """Replicates main()'s exact shuffling/split so real_eval here is
    token-for-token the same split stage 2 was evaluated on. NOTE: the same
    rng object shuffles synthetic FIRST, so we must too (shuffle consumes
    rng state as a function of list length)."""
    import os
    rng = random.Random(0)
    synthetic = load_synthetic("../data/linguistic/synthetic_corruption.json")
    real_path = "../data/linguistic/real_bio_combined.json"
    if not os.path.exists(real_path):
        real_path = "../data/linguistic/bea2019_bio_real.json"
    real = load_real(real_path)
    rng.shuffle(synthetic)
    rng.shuffle(real)
    split_at = int(len(real) * 0.9)
    return real[split_at:]


def evaluate(name: str, examples: list, tokenizer, model) -> dict:
    ds = BIODataset(examples, tokenizer)
    args = TrainingArguments(output_dir="./eval_tmp", per_device_eval_batch_size=32,
                              report_to=[], disable_tqdm=True)
    trainer = Trainer(model=model, args=args,
                      data_collator=DataCollatorForTokenClassification(tokenizer),
                      compute_metrics=compute_metrics)
    m = trainer.evaluate(eval_dataset=ds)
    print(f"\n=== {name} (n={len(examples)}) ===")
    print(f"  precision={m['eval_precision']:.4f} recall={m['eval_recall']:.4f} f1={m['eval_f1']:.4f}")
    return {"benchmark": name, "n_sentences": len(examples),
            "precision": m["eval_precision"], "recall": m["eval_recall"],
            "f1": m["eval_f1"], "pred_tag_counts": m["eval_pred_tag_counts"],
            "gold_tag_counts": m["eval_gold_tag_counts"]}


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForTokenClassification.from_pretrained(MODEL_DIR, torch_dtype=torch.float32)
    model.eval()

    results = []
    real_eval = reproduce_real_eval()
    results.append(evaluate("real_eval (same split as stage 2's 0.60 -- H2 comparison)",
                             real_eval, tokenizer, model))
    results.append(evaluate("fce_test (held-out, same domain)",
                             load_real("../data/linguistic/fce_test_bio.json"), tokenizer, model))
    results.append(evaluate("holbrook (held-out, DOMAIN SHIFT: struggling children)",
                             load_real("../data/linguistic/holbrook_bio.json"), tokenizer, model))

    with open("../data/linguistic/final_model_real_benchmarks.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("\nSaved -> ../data/linguistic/final_model_real_benchmarks.json")
    print("\nH2 read-out: if real_eval F1 here > stage 2's 0.60, the synthetic")
    print("curriculum helped on real data. The fce_test vs holbrook gap measures")
    print("transfer from L2-learner training data to the target population.")
