#!/usr/bin/env bash
# Experiment 30 (server A, after exp27a2): controls requested by the internal review.
# 1. Matched-count control for perturbed views: 6 temperature samples (T=1) per adapter (zs, v2) of
#    the CLEAN audio vs 6 perturbed views, on the evaluation clips only (unseen-sentence read speech,
#    spontaneous dev/itest, OOD); every pool is rescored by the same hybrid, tuned from scratch with
#    3 seeds (scripts/hybrid_ablation.py --only full).
# 2. Re-decode zero-shot and LoRA v1 with pause-based chunking: their stored predictions predate the
#    chunking fix (4631bb8), which inflated their WER on the 15 dev-spont clips > 39 s.
set -euo pipefail
# runs alone on the GPU: started next to exp27a it ran out of memory (and took exp27a down)
until [ -f /workspace/simbig/artifacts/preds/hyb_pool_f0.parquet ]; do sleep 120; done  # exp27a2 GPU stage
cd /workspace/simbig/envs/omni
export HF_HOME=/workspace/simbig/cache/hf PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=/workspace/simbig/repo/scripts; A=/workspace/simbig/artifacts
uv run python - <<'PY'
import polars as pl
from qxp_asr.paths import ARTIFACTS, MANIFESTS
man = pl.read_parquet(MANIFESTS / "all.parquet")
ev = man.filter(
    (pl.col("role").is_in(["dev_scripted", "itest_scripted"]) & pl.col("unseen_sentence"))
    | pl.col("role").is_in(["dev_spont", "itest_spont", "ood"])
)["clip_id"]
pool = pl.read_parquet(ARTIFACTS / "cands" / "pool_dev_aug.parquet").filter(pl.col("clip_id").is_in(ev))
pool.write_parquet(ARTIFACTS / "cands" / "pool_ctrl_aug.parquet")
base = pool.with_columns(
    sources=pl.col("sources").list.eval(pl.element().filter(~pl.element().str.contains("@aug")))
).filter(pl.col("sources").list.len() > 0)
base.write_parquet(ARTIFACTS / "cands" / "pool_ctrl_base.parquet")
print(pool["clip_id"].n_unique(), "clips;", pool.height, "candidates with views;", base.height, "without")
PY
uv run python $R/llm_candidates.py --ctc pool_ctrl_base --name pool_ctrl_smp --nbest none \
    --samples 6 --sample-temp 1.0 --view-adapters zs,v2 --batch-seconds 30 --adapter zs \
    --adapter v1=$A/runs/llm7b_lora_v1/best_trainable.pt \
    --adapter v2=$A/runs/llm7b_lora_v2/best_trainable.pt
cd /workspace/simbig/repo
for p in base aug smp; do
  /workspace/simbig/envs/omni/.venv/bin/python $R/hybrid_ablation.py --cands pool_ctrl_$p \
      --only full --workers 6 --out reports/ctrl_$p.csv
done
cd /workspace/simbig/envs/omni
uv run python $R/infer_llm.py --name zs_llm7b_pc --card omniASR_LLM_7B_v2
uv run python $R/infer_llm.py --name llm7b_lora_v1_pc --card omniASR_LLM_7B_v2 --decoder-lora 16 \
    --weights $A/runs/llm7b_lora_v1/best_trainable.pt
echo "EXP30 DONE"
