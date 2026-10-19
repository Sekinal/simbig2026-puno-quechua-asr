#!/usr/bin/env bash
# Experiment 33 (server A, CPU): re-run the hybrid ablation, the views-vs-samples control and the
# soup-scorer test with per-clip outputs saved (hybrid_ablation.py now writes
# artifacts/preds/ablation_<out>.parquet), so every ablation row gets a paired confidence interval in
# scripts/paper_stats.py. Same seeds and settings as before, so the means reproduce.
set -euo pipefail
cd /workspace/simbig/repo
PY=/workspace/simbig/envs/omni/.venv/bin/python; R=/workspace/simbig/repo/scripts
$PY $R/hybrid_ablation.py --cands pool_dev_aug --workers 4 --out reports/hybrid_ablation.csv
for p in base aug smp; do
  $PY $R/hybrid_ablation.py --cands pool_ctrl_$p --only full --workers 4 --out reports/ctrl_$p.csv
done
for p in pool_dev_aug_soup pool_dev_aug_soup_only; do
  $PY $R/hybrid_ablation.py --cands $p --only full --workers 4 --out reports/abl_$p.csv
done
echo "EXP33 DONE"
