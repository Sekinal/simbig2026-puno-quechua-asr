#!/usr/bin/env bash
# Experiment 48a (server A, CPU): the hybrid's 4th scorer = rank-64 enc+dec LoRA (exp48 scores)
# instead of rank 16, same score-only pool, 3 from-scratch searches, vs the rank-16 version (exp40a).
set -euo pipefail
cd /workspace/simbig/repo
PY=/workspace/simbig/envs/omni/.venv/bin/python; R=/workspace/simbig/repo/scripts
$PY - <<'PY'
import polars as pl
from qxp_asr.paths import ARTIFACTS
C = ARTIFACTS / "cands"
p = pl.read_parquet(C / "pool_ctrl_ed_scoreonly.parquet").drop("lp_ed")
s = pl.read_parquet(C / "pool_ctrl_r64.parquet").select("clip_id", "text", "lp_r64")
p.join(s, on=["clip_id", "text"], how="left").write_parquet(C / "pool_ctrl_r64_scoreonly.parquet")
PY
$PY $R/hybrid_ablation.py --cands pool_ctrl_r64_scoreonly --only full --workers 8 \
    --out reports/abl_pool_ctrl_r64_scoreonly.csv
$PY $R/compare_runs.py --a abl:abl_pool_ctrl_r64_scoreonly:full --b abl:abl_pool_ctrl_ed_scoreonly:full
echo "EXP48A DONE"
