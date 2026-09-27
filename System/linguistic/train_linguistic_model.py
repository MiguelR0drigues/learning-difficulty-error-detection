"""
Token-classification fine-tuning for the linguistic error detection module
(Thesis Ch.4). Implements the 3-stage GECToR-style curriculum from
model_training_plan.md: synthetic-only -> real-only -> combined.

Recommended production base model: microsoft/deberta-v3-base (see
model_training_plan.md for why -- DeBERTa-v3 retains a precision edge over
newer encoders like ModernBERT on NER/token-classification-style tasks).
This script defaults to deberta-v3-xsmall purely so the pipeline can be
smoke-tested end-to-end on a 2-core CPU sandbox in a couple of minutes;
swap MODEL_NAME below to deberta-v3-base (or larger) once running on a
GPU machine for real training.
"""
from __future__ import annotations

import json
import random
from collections import Counter

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (AutoTokenizer, AutoModelForTokenClassification,
                           TrainingArguments, Trainer, DataCollatorForTokenClassification)
import seqeval.metrics as seqeval_metrics

import argparse
import os

# SMOKE_TEST=1 (default) runs the tiny CPU-friendly sanity check that was
# already validated in this sandbox. SMOKE_TEST=0 runs the real, full-scale
# 3-stage curriculum on the complete datasets -- intended for Miguel's
# RTX 5070 (12GB VRAM), not this sandbox (no GPU here).
SMOKE_TEST = os.environ.get("SMOKE_TEST", "1") == "1"
# SEED: rerun the full curriculum with a different shuffle/init seed
# (multi-seed variance for Ch.5). SAVE_STAGE2=1 additionally saves the
# stage-2 (real-only) checkpoint, needed for the paired bootstrap CI of
# the H2 difference (stage-2 vs final on identical resamples).
RUN_SEED = int(os.environ.get("SEED", "0"))
SAVE_STAGE2 = os.environ.get("SAVE_STAGE2", "0") == "1"
# MODEL_NAME_OVERRIDE: swap the encoder without editing code (e.g. the
# BERT-base baseline run). Output dirs get a model tag so different encoders
# never overwrite each other's checkpoints.
MODEL_NAME = os.environ.get("MODEL_NAME_OVERRIDE") or \
    ("microsoft/deberta-v3-xsmall" if SMOKE_TEST else "microsoft/deberta-v3-base")
# SYNTH_LIMIT: cap the synthetic pool (after shuffle) to N examples. 0 = use
# all. Used for the H2-at-optimal-scale test: the scale ablation showed real
# F1 peaks at ~10k synthetic and falls with more, so re-running the paired
# stage2-vs-final H2 CI at SYNTH_LIMIT=10000 tests whether the curriculum
# beats the real-only baseline at the scale where synthetic actually helps.
# The real train/eval split is unaffected (synthetic is shuffled with the same
# rng state, then truncated, before real is shuffled), so eval_h2_ci and
# real_eval_split stay consistent across limits.
SYNTH_LIMIT = int(os.environ.get("SYNTH_LIMIT", "0"))
# RUN_TAG: extra suffix on every checkpoint/summary path so variant runs
# (e.g. the SYNTH_LIMIT sweep) never overwrite the headline 1M checkpoints.
RUN_TAG = os.environ.get("RUN_TAG", "")
MODEL_TAG = (("_" + MODEL_NAME.split("/")[-1]) if os.environ.get("MODEL_NAME_OVERRIDE") else "") + RUN_TAG
LABEL_LIST = ["O", "B-PHONO", "I-PHONO", "B-ORTHO", "I-ORTHO", "B-SEG", "I-SEG"]
LABEL2ID = {l: i for i, l in enumerate(LABEL_LIST)}
ID2LABEL = {i: l for l, i in LABEL2ID.items()}


def load_synthetic(path: str) -> list:
    data = json.load(open(path))
    out = []
    for row in data:
        out.append({"tokens": row["tokens"], "tags": row["bio_tags"]})
    return out


def load_real(path: str) -> list:
    data = json.load(open(path))
    return [{"tokens": row["tokens"], "tags": row["tags"]} for row in data]


class BIODataset(Dataset):
    def __init__(self, examples: list, tokenizer, max_length: int = 64):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        enc = self.tokenizer(ex["tokens"], is_split_into_words=True, truncation=True,
                              max_length=self.max_length)
        word_ids = enc.word_ids()
        labels = []
        prev_word_id = None
        for wid in word_ids:
            if wid is None:
                labels.append(-100)
            elif wid != prev_word_id:
                labels.append(LABEL2ID.get(ex["tags"][wid], 0))
            else:
                # subword continuation: reuse I- version of the tag if it was B-/I-
                tag = ex["tags"][wid]
                if tag.startswith("B-"):
                    tag = "I-" + tag[2:]
                labels.append(LABEL2ID.get(tag, 0))
            prev_word_id = wid
        enc["labels"] = labels
        return {k: torch.tensor(v) for k, v in enc.items()}


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=2)
    true_predictions, true_labels = [], []
    for pred_row, label_row in zip(predictions, labels):
        p_seq, l_seq = [], []
        for p, l in zip(pred_row, label_row):
            if l == -100:
                continue
            p_seq.append(ID2LABEL[p])
            l_seq.append(ID2LABEL[l])
        true_predictions.append(p_seq)
        true_labels.append(l_seq)
    # Diagnostic: raw predicted-tag distribution. If this stays 100% 'O'
    # across runs, the model literally never predicts an error span --
    # different from a low but nonzero F1, which would mean it's predicting
    # spans but getting them wrong/misaligned.
    pred_counts = Counter(t for seq in true_predictions for t in seq)
    gold_counts = Counter(t for seq in true_labels for t in seq)
    print(f"  [eval] predicted tag counts: {dict(pred_counts)}")
    print(f"  [eval] gold tag counts:      {dict(gold_counts)}")
    return {
        "precision": seqeval_metrics.precision_score(true_labels, true_predictions),
        "recall": seqeval_metrics.recall_score(true_labels, true_predictions),
        "f1": seqeval_metrics.f1_score(true_labels, true_predictions),
        "pred_tag_counts": dict(pred_counts),
        "gold_tag_counts": dict(gold_counts),
    }


def compute_class_weights(examples: list) -> torch.Tensor:
    """BIO tagging here is extremely imbalanced: most tokens in a sentence
    are 'O' (correct) and only 1-2 are the actual error span. Plain
    unweighted cross-entropy lets the model take the trivial shortcut of
    always predicting 'O' -- it gets low loss without ever learning to flag
    an error (observed: eval_precision/recall/f1 stuck at exactly 0.0 across
    every stage, with seqeval's "no predicted samples" warning). Standard
    fix: inverse-frequency class weighting (sklearn's 'balanced' formula),
    computed once per stage from that stage's own training label distribution."""
    counts = Counter()
    for ex in examples:
        for tag in ex["tags"]:
            counts[LABEL2ID.get(tag, 0)] += 1
    total = sum(counts.values()) or 1
    num_classes = len(LABEL_LIST)
    weights = []
    for i in range(num_classes):
        c = counts.get(i, 0)
        # Raw inverse-frequency (sklearn 'balanced' formula, ~133x ratio)
        # collapsed the model to predicting the priciest class 100% of the
        # time. Dampening it (sqrt, tighter cap) swung the other way: with
        # weights too mild, it collapsed straight back to predicting 'O'
        # 100% of the time. Static per-class weights alone are too blunt an
        # instrument here -- switched the loss itself to Focal Loss (see
        # FocalLoss below), which makes weight tuning much less brittle, so
        # this alpha only needs to be a mild nudge now, not the whole fix.
        w = (total / (num_classes * c)) ** 0.5 if c > 0 else 1.0
        weights.append(min(max(w, 0.5), 3.0))
    print(f"Label counts: { {LABEL_LIST[i]: counts.get(i, 0) for i in range(num_classes)} }")
    print(f"Class weights: { {LABEL_LIST[i]: round(weights[i], 2) for i in range(num_classes)} }")
    return torch.tensor(weights, dtype=torch.float)


class FocalLoss(torch.nn.Module):
    """Focal Loss (Lin et al. 2017, RetinaNet) for extreme class imbalance.
    Standard weighted cross-entropy applies a *fixed* multiplier to every
    token of a given class regardless of whether the model already gets it
    right -- with imbalance this severe (majority-class tokens ~40-90x more
    common than any single error tag) that fixed multiplier is impossible
    to tune well: too small and the model ignores rare classes entirely,
    too large and it starts always predicting the priciest one (both
    observed empirically here). Focal loss instead multiplies each token's
    loss by (1-p_t)^gamma, where p_t is the model's confidence in the
    *correct* label -- so once the model is confidently right about an easy
    'O' token, that token's contribution to the loss/gradient shrinks
    toward zero on its own, freeing up training signal for the tokens
    (mostly the rare error tags) it's still getting wrong. alpha (class
    weight) is kept, but only as a mild nudge now, not the primary lever."""

    def __init__(self, alpha: torch.Tensor = None, gamma: float = 2.0, ignore_index: int = -100):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, logits, labels):
        valid = labels != self.ignore_index
        if not valid.any():
            return logits.sum() * 0.0
        logits_v = logits[valid]
        labels_v = labels[valid]
        # pt = exp(-ce) is a common shortcut, but it's only mathematically
        # correct for *unweighted* CE -- with a class weight applied, ce is
        # scaled by alpha, so exp(-ce) no longer recovers the true
        # probability (it recovers p_y ** alpha instead), and empirically
        # this run's saved model came out with 100% NaN weights. Computing
        # pt directly from log_softmax avoids that distortion and reuses
        # PyTorch's numerically-stable log-sum-exp internally, instead of a
        # raw exp() that can misbehave under bf16's lower mantissa precision.
        log_probs = torch.nn.functional.log_softmax(logits_v, dim=-1)
        pt = log_probs.gather(1, labels_v.unsqueeze(1)).squeeze(1).exp().clamp(min=1e-6, max=1.0)
        ce = torch.nn.functional.nll_loss(log_probs, labels_v, weight=self.alpha, reduction="none")
        focal = ((1 - pt) ** self.gamma) * ce
        return focal.mean()


class WeightedTrainer(Trainer):
    """Trainer using Focal Loss (see FocalLoss) for token classification.
    Falls back to gamma=0 (== plain weighted cross-entropy) when no weights
    are given (e.g. smoke test)."""

    def __init__(self, *args, class_weights: torch.Tensor = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        weight = self.class_weights.to(device=logits.device, dtype=logits.dtype) if self.class_weights is not None else None
        loss_fct = FocalLoss(alpha=weight, gamma=2.0 if weight is not None else 0.0, ignore_index=-100)
        loss = loss_fct(logits.view(-1, logits.shape[-1]), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


def run_stage(name: str, train_examples: list, eval_examples: list, tokenizer,
              model, output_dir: str, epochs: float = 1.0, batch_size: int = 8,
              do_eval: bool = False):
    import time
    t0 = time.time()
    print(f"\n{'='*70}\nSTAGE: {name}  (train={len(train_examples)}, eval={len(eval_examples)})\n{'='*70}", flush=True)
    train_ds = BIODataset(train_examples, tokenizer)
    eval_ds = BIODataset(eval_examples, tokenizer) if (eval_examples and do_eval) else None
    collator = DataCollatorForTokenClassification(tokenizer)
    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        # DeBERTa-v3's disentangled attention is known to be numerically
        # unstable with the Trainer's default lr=5e-5 and no warmup -- loss
        # spikes then grad_norm goes NaN a few steps in (observed on real
        # run: loss 1.18->1.3->1.58 then NaN for the rest of training).
        # Lower LR + warmup + bf16 (native on Ampere+/Blackwell, wider
        # dynamic range than fp32's problematic intermediate matmuls here)
        # is the standard fix reported for this exact failure mode.
        learning_rate=2e-5,
        warmup_ratio=0.1,
        max_grad_norm=1.0,
        # bf16 was meant to help DeBERTa-v3's known instability, but the
        # final saved model came back with 100% NaN weights on this exact
        # setup (RTX 5070 / very new torch+transformers build) -- disabling
        # it trades some speed for numerical safety while we confirm
        # training is actually stable. Re-enable once a full run comes back
        # clean (no NaN check failures, see below).
        bf16=False,
        logging_steps=5,
        eval_strategy="epoch" if eval_ds else "no",
        save_strategy="no",
        report_to=[],
        disable_tqdm=True,
    )
    class_weights = compute_class_weights(train_examples)
    trainer = WeightedTrainer(model=model, args=args, train_dataset=train_ds, eval_dataset=eval_ds,
                               data_collator=collator, compute_metrics=compute_metrics if eval_ds else None,
                               class_weights=class_weights)
    train_result = trainer.train()
    wall_time = time.time() - t0
    print(f"[{name}] train_loss={train_result.training_loss:.4f}  wall_time={wall_time:.1f}s", flush=True)

    # Hard check: catch a NaN-corrupted model immediately instead of
    # silently saving/evaluating garbage (this is exactly what happened
    # last run -- eval_precision/recall/f1 were 0.0 not because of class
    # imbalance but because the saved model's weights were entirely NaN).
    nan_params = [n for n, p in model.named_parameters() if torch.isnan(p).any()]
    if nan_params:
        print(f"[{name}] *** WARNING: {len(nan_params)} parameter tensors are NaN after training! ***")
        print(f"[{name}] first few: {nan_params[:5]}")
    else:
        print(f"[{name}] NaN check passed: all parameters finite.")

    summary = {"stage": name, "train_loss": train_result.training_loss,
               "wall_time_s": round(wall_time, 1), "nan_params_count": len(nan_params)}
    if eval_ds:
        metrics = trainer.evaluate()
        print(f"[{name}] eval metrics:", metrics, flush=True)
        summary.update(metrics)
    return model, summary


def main():
    rng = random.Random(RUN_SEED)
    torch.manual_seed(RUN_SEED)
    synthetic = load_synthetic("../data/linguistic/synthetic_corruption.json")
    # Combined real corpus (BEA-2019 W&I+LOCNESS + FCE v2.1 train+dev, both
    # via the same m2_parser->taxonomy_map->BIO pipeline). Falls back to the
    # BEA-only file if the combined one hasn't been built. FCE *test* is kept
    # out of this pool entirely (data/linguistic/fce_test_bio.json) as an
    # optional held-out benchmark.
    real_path = "../data/linguistic/real_bio_combined.json"
    if not os.path.exists(real_path):
        real_path = "../data/linguistic/bea2019_bio_real.json"
    real = load_real(real_path)
    print(f"Loaded {len(synthetic)} synthetic examples, {len(real)} real examples ({os.path.basename(real_path)})")
    print(f"MODE: {'SMOKE TEST (tiny CPU sanity check)' if SMOKE_TEST else 'FULL TRAINING (GPU expected)'}")
    print(f"MODEL_NAME: {MODEL_NAME}")

    rng.shuffle(synthetic)
    if SYNTH_LIMIT and SYNTH_LIMIT < len(synthetic):
        print(f"SYNTH_LIMIT: capping synthetic pool {len(synthetic)} -> {SYNTH_LIMIT}")
        synthetic = synthetic[:SYNTH_LIMIT]
    rng.shuffle(real)

    if SMOKE_TEST:
        # Deliberately tiny slices: this is a CPU pipeline smoke test, not a
        # real training run.
        synth_train, synth_eval = synthetic[:16], synthetic[16:22]
        real_train, real_eval = real[:16], real[16:22]
        combined_train = (synth_train + real_train)[:8]
        rng.shuffle(combined_train)
        combined_eval = (synth_eval + real_eval)[:6]
        epochs_stage1 = epochs_stage2 = epochs_stage3 = 1
        batch_size = 8
    else:
        # Real run: full datasets, held-out eval split (90/10), more epochs.
        # Batch size 16 / max_length 64 comfortably fits deberta-v3-base on
        # a 12GB GPU with room to spare; bump to 32 if memory allows.
        split = lambda lst, frac=0.9: (lst[:int(len(lst) * frac)], lst[int(len(lst) * frac):])
        synth_train, synth_eval = split(synthetic)
        real_train, real_eval = split(real)
        combined_train = synth_train + real_train
        rng.shuffle(combined_train)
        combined_eval = synth_eval + real_eval
        # Stage 1/3 epochs pulled back down: the synthetic dataset grew from
        # 2k to ~1M examples (see synthetic_corruption.py), so 15 epochs
        # there is no longer "cheap" -- it's 500x more data per epoch than
        # when that number was picked. More data needs fewer passes over it,
        # not more, so stage 1/3 drop to 2. Stage 2 dropped 15 -> 8: the
        # real corpus grew ~3x (2.6k BEA-only -> 6.9k BEA+FCE), so fewer
        # passes are needed for the same number of gradient updates.
        epochs_stage1, epochs_stage2, epochs_stage3 = 2, 8, 2
        batch_size = 16

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME, num_labels=len(LABEL_LIST), id2label=ID2LABEL, label2id=LABEL2ID,
        # The checkpoint's own stored dtype was being used by default (crashed
        # with "expected scalar type Half but found Float" against our fp32
        # class-weight tensor) -- meaning training was silently happening in
        # fp16 with no Trainer-managed loss scaling (since fp16=True was never
        # set), which is a well-known recipe for the exact NaN weights seen
        # in every run so far. Force fp32 explicitly instead of trusting the
        # checkpoint's default dtype.
        torch_dtype=torch.float32)

    stage_summaries = []

    # Stage 1: synthetic only
    model, s1 = run_stage("1_synthetic_only", synth_train, synth_eval, tokenizer, model, "./out_stage1",
                           epochs=epochs_stage1, batch_size=batch_size, do_eval=not SMOKE_TEST)
    stage_summaries.append(s1)
    # Stage 2: real only (continue fine-tuning)
    model, s2 = run_stage("2_real_only", real_train, real_eval, tokenizer, model, "./out_stage2",
                           epochs=epochs_stage2, batch_size=batch_size, do_eval=not SMOKE_TEST)
    stage_summaries.append(s2)
    if SAVE_STAGE2:
        s2_dir = f"./stage2_model_seed{RUN_SEED}{MODEL_TAG}"
        model.save_pretrained(s2_dir); tokenizer.save_pretrained(s2_dir)
        print(f"Saved stage-2 checkpoint to {s2_dir}")
    # Stage 3: combined (this is the one we actually evaluate)
    model, s3 = run_stage("3_combined", combined_train, combined_eval, tokenizer, model, "./out_stage3",
                           epochs=epochs_stage3, batch_size=batch_size, do_eval=True)
    stage_summaries.append(s3)

    out_dir = "./smoke_test_final_model" if SMOKE_TEST else (f"./final_model_seed{RUN_SEED}{MODEL_TAG}" if (RUN_SEED != 0 or MODEL_TAG) else "./final_model")
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(f"Saved model checkpoint to {out_dir}")

    if SMOKE_TEST:
        print("\nSMOKE TEST COMPLETE: pipeline runs end-to-end (data -> tokenize/align -> "
              "3-stage train -> seqeval eval) on CPU with a tiny model/subset.")
        print("For real results: run with SMOKE_TEST=0 on a GPU machine (e.g. `SMOKE_TEST=0 python3 train_linguistic_model.py`).")
    else:
        print("\nFULL TRAINING COMPLETE. Compare stage 1 vs 2 vs 3 eval metrics to assess H2")
        print("(does combining synthetic + real data outperform either alone?).")

    # ---- End-of-run summary -------------------------------------------
    print(f"\n{'='*70}\nSUMMARY\n{'='*70}")
    if SMOKE_TEST:
        print("*** SMOKE TEST MODE: the metrics below are MEANINGLESS by design ***")
        print("*** (16 train / 6 eval sentences, xsmall model, 1 epoch -- this  ***")
        print("*** only checks the pipeline runs). For real results, run:       ***")
        print("***   SMOKE_TEST=0 python3 train_linguistic_model.py   (on GPU)  ***")
    header = (f"{'stage':<20} {'train_loss':>10} {'eval_loss':>10} {'precision':>10} "
              f"{'recall':>10} {'f1':>8} {'wall_s':>8} {'nan_params':>10}")
    print(header)
    print("-" * len(header))
    for s in stage_summaries:
        print(f"{s.get('stage', ''):<20} "
              f"{s.get('train_loss', float('nan')):>10.4f} "
              f"{s.get('eval_loss', float('nan')):>10.4f} "
              f"{s.get('eval_precision', float('nan')):>10.4f} "
              f"{s.get('eval_recall', float('nan')):>10.4f} "
              f"{s.get('eval_f1', float('nan')):>8.4f} "
              f"{s.get('wall_time_s', float('nan')):>8.1f} "
              f"{s.get('nan_params_count', 0):>10}")
    if stage_summaries and "eval_pred_tag_counts" in stage_summaries[-1]:
        print(f"\nFinal stage predicted tag distribution: {stage_summaries[-1]['eval_pred_tag_counts']}")
        print(f"Final stage gold tag distribution:      {stage_summaries[-1]['eval_gold_tag_counts']}")

    summary_path = f"../data/linguistic/training_run_summary_seed{RUN_SEED}{MODEL_TAG}.json" if not SMOKE_TEST else "./smoke_test_summary.json"
    try:
        with open(summary_path, "w") as f:
            json.dump(stage_summaries, f, indent=2, default=str)
        print(f"\nFull summary saved to {summary_path}")
    except OSError as e:
        print(f"\n(could not save summary JSON: {e})")


if __name__ == "__main__":
    main()
