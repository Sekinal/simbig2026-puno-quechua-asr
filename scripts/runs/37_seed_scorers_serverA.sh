#!/usr/bin/env bash
# Experiment 37 (server A, CPU): LoRA v2 seeds 1 and 2 (exp32/35) as extra hybrid scorers, added to
# the full hybrid (lp_s1, lp_s2); 3 from-scratch weight searches, compared with the full hybrid.
set -euo pipefail
cd /workspace/simbig/repo
PY=/workspace/simbig/envs/omni/.venv/bin/python; R=/workspace/simbig/repo/scripts
$PY - <<'PY'
import polars as pl
from qxp_asr.paths import ARTIFACTS
C = ARTIFACTS / "cands"
pool = pl.read_parquet(C / "pool_dev_aug.parquet")
seeds = pl.read_parquet(C / "pool_dev_seeds.parquet").select("clip_id", "text", "lp_s1", "lp_s2")
both = pool.join(seeds, on=["clip_id", "text"], how="left")
both.write_parquet(C / "pool_dev_aug_seeds.parquet")
print(both.height, "candidates;", both["lp_s2"].null_count(), "without seed scores")
PY
$PY $R/hybrid_ablation.py --cands pool_dev_aug_seeds --only full --workers 4 \
    --out reports/abl_pool_dev_aug_seeds.csv
$PY $R/compare_runs.py --a abl:abl_pool_dev_aug_seeds:full --b abl:hybrid_ablation:full
$PY $R/compare_runs.py --a llm7b_v2_seedsoup --b llm7b_lora_v2
echo "EXP37 DONE"
