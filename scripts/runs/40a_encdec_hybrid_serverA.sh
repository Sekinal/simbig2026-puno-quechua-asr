#!/usr/bin/env bash
# Experiment 40a (server A, CPU): hybrid tests for the encoder+decoder LoRA (exp40 pool).
# full: ed candidates + ed scores; score-only: candidates proposed only by ed removed. Both vs the
# hybrid on the same clips without ed (ctrl_aug); 3 from-scratch weight searches each.
set -euo pipefail
cd /workspace/simbig/repo
PY=/workspace/simbig/envs/omni/.venv/bin/python; R=/workspace/simbig/repo/scripts
$PY - <<'PY'
import polars as pl
from qxp_asr.paths import ARTIFACTS
C = ARTIFACTS / "cands"
p = pl.read_parquet(C / "pool_ctrl_ed.parquet")
non_ed = pl.col("sources").list.eval(~pl.element().str.starts_with("ed")).list.any()
p.filter(non_ed).write_parquet(C / "pool_ctrl_ed_scoreonly.parquet")
print(p.height, "candidates;", p.filter(non_ed).height, "not only from ed")
PY
for t in pool_ctrl_ed pool_ctrl_ed_scoreonly; do
  $PY $R/hybrid_ablation.py --cands $t --only full --workers 4 --out reports/abl_$t.csv
  $PY $R/compare_runs.py --a abl:abl_$t:full --b abl:ctrl_aug:full
done
echo "EXP40A DONE"
