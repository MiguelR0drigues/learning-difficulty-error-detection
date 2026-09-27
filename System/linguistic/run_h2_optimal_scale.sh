#!/usr/bin/env bash
# H2 at the optimal synthetic scale (Ch.5 follow-up).
# The scale ablation showed real F1 peaks near 10k synthetic and falls with
# more. This retrains the curriculum at SYNTH_LIMIT=10000 for several seeds
# (saving the stage-2 / real-only checkpoint) and runs the paired stage-2-vs-
# final H2 CI on TWO sets:
#   * real  -- the seed's real held-out split (was used to pick the scale, so
#              treat a positive here as suggestive, not confirmatory)
#   * fce   -- the FCE test set, never used for scale selection -> the honest
#              confirmation set for the H2-at-optimal-scale claim
#
# Non-destructive: everything is tagged _synth10k, so the headline 1M
# checkpoints and h2_paired_ci_seed*.json are untouched.
#
# Usage (System/linguistic/, venv active, GPU):
#   bash run_h2_optimal_scale.sh 2>&1 | tee h2_optscale.log
# override seeds / scale:
#   SEEDS="1 2 3" LIMIT=10000 bash run_h2_optimal_scale.sh
#
# Send back: ../data/linguistic/h2_paired_ci_seed*_synth10k_*.json
set -u
export SMOKE_TEST=0
: "${SEEDS:=1 2 3}"
: "${LIMIT:=10000}"
TAG="_synth${LIMIT%000}k"; [ "$LIMIT" = 10000 ] && TAG="_synth10k"
STAMP() { date "+%F %T"; }

for S in $SEEDS; do
  echo ""; echo "[$(STAMP)] ===== seed $S: train @ SYNTH_LIMIT=$LIMIT (tag $TAG) ====="
  if [ -d "final_model_seed${S}${TAG}" ] && [ -d "stage2_model_seed${S}${TAG}" ]; then
    echo "  checkpoints exist, skipping training"
  else
    SEED=$S SYNTH_LIMIT=$LIMIT SAVE_STAGE2=1 RUN_TAG=$TAG python train_linguistic_model.py \
      || { echo "!! seed $S training FAILED"; continue; }
  fi
  echo "[$(STAMP)] seed $S: H2 CI on real split"
  SEED=$S RUN_TAG=$TAG EVAL_SET=real python eval_h2_ci.py || echo "!! seed $S real CI FAILED"
  echo "[$(STAMP)] seed $S: H2 CI on FCE test (independent confirmation)"
  SEED=$S RUN_TAG=$TAG EVAL_SET=fce  python eval_h2_ci.py || echo "!! seed $S fce CI FAILED"
done

echo ""; echo "[$(STAMP)] ===== DONE ====="
ls -la ../data/linguistic/h2_paired_ci_seed*${TAG}_*.json 2>/dev/null
