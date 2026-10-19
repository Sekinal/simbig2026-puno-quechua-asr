#!/usr/bin/env bash
# Experiment 43a (server A, CPU): the hybrid's 4th scorer = soup of the two enc+dec seeds (exp43)
# instead of seed 0 (same score-only pool), 3 from-scratch searches, vs the seed-0 version (exp40a).
set -euo pipefail
cd /workspace/simbig/repo
PY=/workspace/simbig/envs/omni/.venv/bin/python; R=/workspace/simbig/repo/scripts
$PY - <<'PY'
import polars as pl
from qxp_asr.paths import ARTIFACTS
C = ARTIFACTS / "cands"
p = pl.read_parquet(C / "pool_ctrl_ed_scoreonly.parquet").drop("lp_ed")
s = pl.read_parquet(C / "pool_ctrl_edsoup.parquet").select("clip_id", "text", "lp_edsoup")
p.join(s, on=["clip_id", "text"], how="left").write_parquet(C / "pool_ctrl_edsoup_scoreonly.parquet")
PY
$PY $R/hybrid_ablation.py --cands pool_ctrl_edsoup_scoreonly --only full --workers 4 \
    --out reports/abl_pool_ctrl_edsoup_scoreonly.csv
$PY $R/compare_runs.py --a abl:abl_pool_ctrl_edsoup_scoreonly:full \
    --b abl:abl_pool_ctrl_ed_scoreonly:full
echo "EXP43A DONE"
