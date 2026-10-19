#!/usr/bin/env bash
# Experiment 38 (server B, after exp36): the hybrid with the CTC-3B log-likelihood instead of the
# CTC-1B v3 one (same perturbed-view pool and features; 3 from-scratch weight searches), compared
# with the full hybrid; and CTC-3B greedy vs CTC-1B greedy as single models.
set -euo pipefail
until grep -q "EXP36 DONE" /workspace/simbig/logs/exp36.log; do sleep 300; done
cd /workspace/simbig/repo
PY=/workspace/simbig/envs/omni/.venv/bin/python; R=/workspace/simbig/repo/scripts
$PY $R/hybrid_ablation.py --cands pool_dev_aug --post ctc3b_v3_soup_wise0.85 --only full \
    --workers 8 --out reports/abl_ctc3b.csv
$PY $R/compare_runs.py --a abl:abl_ctc3b:full --b abl:hybrid_ablation:full
echo "EXP38 DONE"
