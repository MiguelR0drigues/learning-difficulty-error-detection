#!/usr/bin/env bash
# Overnight queue for the scientific-rigor GPU runs (RUN_ON_GPU.md §1b).
# Runs each job when the previous finishes; safe to re-run (skips what's done).
#
# Usage (from System/linguistic/, venv active):
#   bash run_overnight.sh 2>&1 | tee overnight.log
# or fully detached:
#   nohup bash run_overnight.sh > overnight.log 2>&1 &
#
# In the morning, send me:
#   ../data/linguistic/h2_paired_ci_seed*.json
#   ../data/linguistic/training_run_summary_seed3.json
#   ../data/linguistic/training_run_summary_seed0_bert-base-cased.json
#   overnight_bert_eval.json

set -u
export SMOKE_TEST=0
STAMP() { date "+%Y-%m-%d %H:%M:%S"; }
say() { echo ""; echo "[$(STAMP)] ================ $* ================"; }

# ---- 1) paired H2 CIs for seeds already trained (minutes each) ----------
for S in 1 2; do
  if [ -f "../data/linguistic/h2_paired_ci_seed$S.json" ]; then
    say "H2 CI seed $S: already done, skipping"
  elif [ -d "stage2_model_seed$S" ]; then
    say "H2 CI seed $S: running eval_h2_ci.py"
    SEED=$S python eval_h2_ci.py || echo "[$(STAMP)] !! H2 CI seed $S FAILED (continuing)"
  else
    say "H2 CI seed $S: stage2_model_seed$S not found -- was SAVE_STAGE2=1 set? Skipping."
  fi
done

# ---- 2) seed 3: full curriculum + H2 CI (~4h) ----------------------------
if [ -f "../data/linguistic/h2_paired_ci_seed3.json" ]; then
  say "Seed 3: already done, skipping"
else
  if [ ! -d "final_model_seed3" ]; then
    say "Seed 3: training (approx. 4h)"
    SEED=3 SAVE_STAGE2=1 python train_linguistic_model.py || echo "[$(STAMP)] !! seed 3 training FAILED (continuing)"
  else
    say "Seed 3: checkpoint exists, skipping training"
  fi
  if [ -d "stage2_model_seed3" ] && [ -d "final_model_seed3" ]; then
    say "Seed 3: H2 CI"
    SEED=3 python eval_h2_ci.py || echo "[$(STAMP)] !! H2 CI seed 3 FAILED (continuing)"
  fi
fi

# ---- 3) BERT-base encoder baseline (~2-3h) + eval on real_eval ----------
BERT_DIR="final_model_seed0_bert-base-cased"
if [ ! -d "$BERT_DIR" ]; then
  say "BERT-base baseline: training"
  SEED=0 MODEL_NAME_OVERRIDE=bert-base-cased python train_linguistic_model.py \
    || echo "[$(STAMP)] !! BERT training FAILED (continuing)"
else
  say "BERT-base baseline: checkpoint exists, skipping training"
fi
if [ -d "$BERT_DIR" ] && [ ! -f "overnight_bert_eval.json" ]; then
  say "BERT-base: evaluating on real_eval (same split as DeBERTa seed 0)"
  python - << 'PYEOF' || echo "!! BERT eval FAILED"
import json
import torch
import seqeval.metrics as sm
from eval_h2_ci import real_eval_split, predict

device = "cuda" if torch.cuda.is_available() else "cpu"
rows = real_eval_split(0)
gold = [r["tags"] for r in rows]
preds = predict("./final_model_seed0_bert-base-cased", rows, device)
out = {"model": "bert-base-cased", "n": len(rows),
       "precision": sm.precision_score(gold, preds),
       "recall": sm.recall_score(gold, preds),
       "f1": sm.f1_score(gold, preds)}
print(json.dumps(out, indent=1))
json.dump(out, open("overnight_bert_eval.json", "w"), indent=1)
PYEOF
fi

say "QUEUE FINISHED"
echo "Results to send back:"
ls -la ../data/linguistic/h2_paired_ci_seed*.json 2>/dev/null
ls -la ../data/linguistic/training_run_summary_seed*.json 2>/dev/null
ls -la overnight_bert_eval.json 2>/dev/null
