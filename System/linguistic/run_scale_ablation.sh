#!/usr/bin/env bash
# Synthetic-scale ablation for Ch.5 (RUN_ON_GPU.md sec C).
# Trains the 3-stage curriculum at 10k / 100k / 1M synthetic examples and
# reports span-F1 on both the in-domain synthetic eval and the real held-out
# split (same one as seed-0 / the H2 CIs). Resumable: existing scale_model_*
# dirs are re-evaluated, not retrained.
#
# Usage (from System/linguistic/, venv active):
#   bash run_scale_ablation.sh 2>&1 | tee scale_ablation.log
# detached:
#   nohup bash run_scale_ablation.sh > scale_ablation.log 2>&1 &
#
# The 1M point == the full seed-0 run (~4h). If final_model_seed0 already
# exists you can skip it and reuse that run's real-eval F1 as the 1M point:
#   SCALES=10000,100000 bash run_scale_ablation.sh
#
# Send me back: ../data/linguistic/scale_ablation_summary*.json
set -u
export SMOKE_TEST=0
: "${SCALES:=10000,100000,1000000}"
export SCALES
echo "[$(date '+%F %T')] scale ablation starting: SCALES=$SCALES"
python scale_ablation.py || { echo "!! scale_ablation.py FAILED"; exit 1; }
echo "[$(date '+%F %T')] done"
ls -la ../data/linguistic/scale_ablation_summary*.json 2>/dev/null
