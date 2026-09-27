#!/usr/bin/env bash
# Overnight runner for the multi-seed H2 experiment (RUN_ON_GPU.md §1b-A).
# Runs seeds sequentially: train (3-stage curriculum, ~4h/seed on the 5070)
# then the paired H2 bootstrap CI. Safe to re-run: seeds whose results
# already exist are skipped, so an interrupted night resumes where it left.
#
# Usage (from System/linguistic/, venv active):
#   bash run_multiseed.sh            # seeds 1 2 3
#   bash run_multiseed.sh 1 2        # just these seeds
#   nohup bash run_multiseed.sh > multiseed_nohup.log 2>&1 &   # detach & sleep
set -u

SEEDS=("${@:-}")
if [ ${#SEEDS[@]} -eq 0 ] || [ -z "${SEEDS[0]}" ]; then SEEDS=(1 2 3); fi

LOGDIR="./multiseed_logs"
mkdir -p "$LOGDIR"

echo "=== run_multiseed: seeds ${SEEDS[*]} | started $(date) ==="

for S in "${SEEDS[@]}"; do
  CI_OUT="../data/linguistic/h2_paired_ci_seed${S}.json"
  FINAL_DIR="./final_model_seed${S}"
  TRAIN_LOG="$LOGDIR/train_seed${S}.log"
  CI_LOG="$LOGDIR/ci_seed${S}.log"

  if [ -f "$CI_OUT" ]; then
    echo "[seed $S] $CI_OUT already exists -- skipping seed entirely."
    continue
  fi

  if [ -d "$FINAL_DIR" ] && [ -f "$FINAL_DIR/model.safetensors" ]; then
    echo "[seed $S] final model already trained -- skipping training."
  else
    echo "[seed $S] training started $(date) (log: $TRAIN_LOG)"
    SMOKE_TEST=0 SEED="$S" SAVE_STAGE2=1 python train_linguistic_model.py > "$TRAIN_LOG" 2>&1
    RC=$?
    if [ $RC -ne 0 ]; then
      echo "[seed $S] TRAINING FAILED (exit $RC) -- see $TRAIN_LOG. Moving to next seed."
      continue
    fi
    echo "[seed $S] training finished $(date)"
  fi

  echo "[seed $S] paired H2 CI started $(date) (log: $CI_LOG)"
  SEED="$S" python eval_h2_ci.py > "$CI_LOG" 2>&1
  RC=$?
  if [ $RC -ne 0 ]; then
    echo "[seed $S] CI EVAL FAILED (exit $RC) -- see $CI_LOG. Moving to next seed."
    continue
  fi
  echo "[seed $S] done $(date) -> $CI_OUT"
done

echo ""
echo "=== SUMMARY $(date) ==="
python - << 'PYEOF'
import glob, json
files = sorted(glob.glob("../data/linguistic/h2_paired_ci_seed*.json"))
if not files:
    print("No CI results found.")
else:
    diffs = []
    for f in files:
        d = json.load(open(f))
        diffs.append(d["diff"])
        print(f"seed {d['seed']}: stage2={d['stage2_f1']:.4f} final={d['final_f1']:.4f} "
              f"diff=+{d['diff']:.4f} CI95={d['diff_ci95']} p<=0: {d['p_diff_leq_0']}")
    if len(diffs) > 1:
        import statistics as st
        print(f"\nacross {len(diffs)} seeds: mean diff=+{st.mean(diffs):.4f} sd={st.stdev(diffs):.4f}")
PYEOF
echo "=== run_multiseed complete ==="
