#!/usr/bin/env bash
# Experiment 31 (server A): the exact LoRA soup (exp28, rank 64: v1-v4) is the best single model
# (itest unseen 19.2 vs 20.2). Does it help as a hybrid scorer? Score the perturbed-view pool with it,
# then re-tune the hybrid from scratch (3 seeds) with the soup added, and with the soup replacing
# LoRA v1 + v2 (scripts/hybrid_ablation.py --only full).
set -euo pipefail
cd /workspace/simbig/envs/omni
export HF_HOME=/workspace/simbig/cache/hf PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=/workspace/simbig/repo/scripts; A=/workspace/simbig/artifacts
uv run python $R/llm_candidates.py --ctc pool_dev_aug --name pool_dev_soup --nbest none --rank 64 \
    --adapter soup=$A/runs/llm7b_soup4/best_trainable.pt
uv run python - <<'PY'
import polars as pl
from qxp_asr.paths import ARTIFACTS
C = ARTIFACTS / "cands"
pool = pl.read_parquet(C / "pool_dev_aug.parquet")
soup = pl.read_parquet(C / "pool_dev_soup.parquet").select("clip_id", "text", "lp_soup")
both = pool.join(soup, on=["clip_id", "text"], how="left")
both.write_parquet(C / "pool_dev_aug_soup.parquet")
both.drop("lp_v1", "lp_v2").write_parquet(C / "pool_dev_aug_soup_only.parquet")
print(both.height, "candidates;", both["lp_soup"].null_count(), "without a soup score")
PY
cd /workspace/simbig/repo
for p in pool_dev_aug_soup pool_dev_aug_soup_only; do
  /workspace/simbig/envs/omni/.venv/bin/python $R/hybrid_ablation.py --cands $p --only full \
      --workers 6 --out reports/abl_$p.csv
done
echo "EXP31 DONE"
