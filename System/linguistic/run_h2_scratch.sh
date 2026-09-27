#!/usr/bin/env bash
# The decisive H2 runs: real-only-from-scratch vs curriculum, seeds 0-3.
# Each seed takes minutes. Usage: bash run_h2_scratch.sh 2>&1 | tee h2_scratch.log
set -u
export SMOKE_TEST=0
for S in 0 1 2 3; do
  if [ -f "../data/linguistic/h2_scratch_ci_seed$S.json" ]; then
    echo "seed $S: done, skipping"
  else
    echo "=== seed $S ==="
    SEED=$S python train_real_only_baseline.py || echo "!! seed $S failed"
  fi
done
echo "Send back: ../data/linguistic/h2_scratch_ci_seed*.json"
