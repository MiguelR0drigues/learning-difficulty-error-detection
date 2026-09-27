"""Synthetic-scale ablation for Ch.5 (RUN_ON_GPU.md sec C).

Trains the identical 3-stage curriculum at several synthetic-data volumes
(default 10k / 100k / 1M) and measures the resulting span-F1 on two held-out
sets:
  * synth_eval  -- in-domain (synthetic) 10% split, grown with the scale
  * real_eval   -- the SAME real held-out split used by the seed-0 experiment
                   and the H2 CIs (eval_h2_ci.real_eval_split(0)), so the
                   real-domain numbers line up 1:1 with the rest of Ch.5.

The point of the curve is to separate *volume* from *type*: does piling on
more synthetic data keep helping on real text, or does it saturate / only help
in-domain? Read alongside the H2 result (combining ~ null on real eval).

Usage (System/linguistic/, venv active, GPU):
    SMOKE_TEST=0 python scale_ablation.py
    SMOKE_TEST=0 SCALES=10000,100000 python scale_ablation.py   # skip the 1M run
    SMOKE_TEST=1 python scale_ablation.py                        # CPU pipeline check

Resumable: a scale whose ./scale_model_{N}{TAG} dir already exists is not
retrained, only re-evaluated. The 1M point is the full seed-0 run; if you have
final_model_seed0 already, pass SCALES=10000,100000 and reuse the seed-0
real-eval F1 as the 1M point rather than paying for it twice.

Writes ../data/linguistic/scale_ablation_summary{TAG}.json.
"""
from __future__ import annotations

import json
import os
import random
import time

import torch
import seqeval.metrics as sm

from transformers import AutoTokenizer, AutoModelForTokenClassification

from train_linguistic_model import (
    load_synthetic, load_real, run_stage, MODEL_NAME, MODEL_TAG,
    LABEL_LIST, ID2LABEL, LABEL2ID,
)
from eval_h2_ci import real_eval_split, predict

SMOKE_TEST = os.environ.get("SMOKE_TEST", "1") == "1"
SEED = int(os.environ.get("SEED", "0"))
# SEEDS: multi-seed variant. Each seed writes its own tagged model dirs and a
# scale_ablation_summary_seed{S}.json, so the cheap 10k/100k points can be
# repeated to put an error bar on the curve without clobbering seed 0.
DEFAULT_SCALES = [1000] if SMOKE_TEST else [10000, 100000, 1000000]
SCALES = [int(x) for x in os.environ.get("SCALES", "").split(",") if x.strip()] or DEFAULT_SCALES

SYNTH_PATH = "../data/linguistic/synthetic_corruption.json"
# Seed 0 keeps the original unsuffixed names (so the completed seed-0 run is
# reused, not retrained); other seeds get a _seed{N} suffix.
SEED_TAG = "" if SEED == 0 else f"_seed{SEED}"


def split(lst, frac=0.9):
    cut = int(len(lst) * frac)
    return lst[:cut], lst[cut:]


def build_model():
    return AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME, num_labels=len(LABEL_LIST), id2label=ID2LABEL, label2id=LABEL2ID,
        torch_dtype=torch.float32)


def f1_on(model_dir, rows, device):
    gold = [r["tags"] for r in rows]
    preds = predict(model_dir, rows, device)
    return {
        "n": len(rows),
        "precision": round(float(sm.precision_score(gold, preds)), 4),
        "recall": round(float(sm.recall_score(gold, preds)), 4),
        "f1": round(float(sm.f1_score(gold, preds)), 4),
    }


def train_at_scale(scale, device):
    """Run the 3-stage curriculum with the synthetic pool capped at `scale`.
    Returns (model_dir, synth_eval_rows). Reuses the exact rng ordering of the
    main script so slices are deterministic and comparable across scales."""
    rng = random.Random(SEED)
    torch.manual_seed(SEED)

    synthetic = load_synthetic(SYNTH_PATH)
    real_path = "../data/linguistic/real_bio_combined.json"
    if not os.path.exists(real_path):
        real_path = "../data/linguistic/bea2019_bio_real.json"
    real = load_real(real_path)

    rng.shuffle(synthetic)
    rng.shuffle(real)

    eff = min(scale, len(synthetic))
    if eff < scale:
        print(f"  [scale {scale}] only {len(synthetic)} synthetic available; using {eff}")
    synthetic = synthetic[:eff]

    synth_train, synth_eval = split(synthetic)
    real_train, real_eval = split(real)
    combined_train = synth_train + real_train
    rng.shuffle(combined_train)

    if SMOKE_TEST:
        synth_train, synth_eval = synth_train[:16], synth_eval[:6]
        real_train = real_train[:16]
        combined_train = (synth_train + real_train)[:16]
        e1 = e2 = e3 = 1
        bs = 8
    else:
        e1, e2, e3 = 2, 8, 2
        bs = 16

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = build_model()

    model, _ = run_stage(f"scale{eff}_1_synth", synth_train, synth_eval, tokenizer,
                         model, "./out_scale_s1", epochs=e1, batch_size=bs, do_eval=False)
    model, _ = run_stage(f"scale{eff}_2_real", real_train, real_eval, tokenizer,
                         model, "./out_scale_s2", epochs=e2, batch_size=bs, do_eval=False)
    model, _ = run_stage(f"scale{eff}_3_comb", combined_train, synth_eval, tokenizer,
                         model, "./out_scale_s3", epochs=e3, batch_size=bs, do_eval=False)

    model_dir = f"./scale_model_{eff}{MODEL_TAG}{SEED_TAG}"
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    return model_dir, synth_eval, eff


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    real_eval = real_eval_split(SEED)  # same held-out real set as seed-0 / H2 CIs
    print(f"MODEL_NAME={MODEL_NAME} | device={device} | scales={SCALES}")
    print(f"real held-out eval sentences: {len(real_eval)}")

    results = []
    for scale in SCALES:
        t0 = time.time()
        eff = min(scale, len(load_synthetic(SYNTH_PATH)))
        model_dir = f"./scale_model_{eff}{MODEL_TAG}{SEED_TAG}"
        print(f"\n{'#'*70}\n# SCALE {scale} (effective {eff})\n{'#'*70}", flush=True)

        if os.path.isdir(model_dir):
            print(f"  checkpoint {model_dir} exists -> skip training, re-evaluate only")
            # reproduce the synth_eval slice for this scale without training
            rng = random.Random(SEED)
            synth = load_synthetic(SYNTH_PATH); rng.shuffle(synth)
            _, synth_eval = split(synth[:eff])
        else:
            model_dir, synth_eval, eff = train_at_scale(scale, device)

        synth_metrics = f1_on(model_dir, synth_eval, device)
        real_metrics = f1_on(model_dir, real_eval, device)
        row = {
            "scale_requested": scale,
            "scale_effective": eff,
            "synth_eval": synth_metrics,
            "real_eval": real_metrics,
            "wall_time_s": round(time.time() - t0, 1),
        }
        results.append(row)
        print(json.dumps(row, indent=1), flush=True)

    print(f"\n{'='*70}\nSCALE ABLATION CURVE\n{'='*70}")
    header = f"{'synth_scale':>12} {'synth_F1':>10} {'real_F1':>10} {'wall_s':>8}"
    print(header); print("-" * len(header))
    for r in results:
        print(f"{r['scale_effective']:>12} {r['synth_eval']['f1']:>10.4f} "
              f"{r['real_eval']['f1']:>10.4f} {r['wall_time_s']:>8.1f}")

    out_path = f"../data/linguistic/scale_ablation_summary{MODEL_TAG}{SEED_TAG}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
