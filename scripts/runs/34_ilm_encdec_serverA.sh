#!/usr/bin/env bash
# Experiment 34 (server A).
# (1) Internal-LM / density-ratio subtraction for the hybrid verifier: every candidate of the
#     perturbed-view pool is scored by zs, v1 and v2 on 1 s of near-silence (the adapters' text
#     prior); the negated priors are added as features (lp_ilm_*), so a positive weight subtracts
#     the prior. The hybrid is re-tuned from scratch (3 seeds) and compared with the full hybrid.
#     Motivation: the tuned read-speech weight of the zero-shot likelihood is already negative.
# (2) Encoder+decoder LoRA (exp08 recipe, never run): acoustic adaptation for new speakers and
#     domains; evaluated as a single model against LoRA v2 (same data).
set -euo pipefail
cd /workspace/simbig/envs/omni
export HF_HOME=/workspace/simbig/cache/hf PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=/workspace/simbig/repo/scripts; A=/workspace/simbig/artifacts; P=$A/pseudo
uv run python $R/llm_candidates.py --ctc pool_dev_aug --name pool_dev_ilm --nbest none --blank-audio \
    --adapter zs --adapter v1=$A/runs/llm7b_lora_v1/best_trainable.pt \
    --adapter v2=$A/runs/llm7b_lora_v2/best_trainable.pt
uv run python - <<'PY'
import polars as pl
from qxp_asr.paths import ARTIFACTS
C = ARTIFACTS / "cands"
pool = pl.read_parquet(C / "pool_dev_aug.parquet")
ilm = pl.read_parquet(C / "pool_dev_ilm.parquet").select(
    "clip_id", "text", *[(-pl.col(f"lp_{a}")).alias(f"lp_ilm_{a}") for a in ("zs", "v1", "v2")]
)
both = pool.join(ilm, on=["clip_id", "text"], how="left")
both.write_parquet(C / "pool_dev_aug_ilm.parquet")
print(both.height, "candidates;", both["lp_ilm_v2"].null_count(), "without a prior score")
PY
cd /workspace/simbig/repo
/workspace/simbig/envs/omni/.venv/bin/python $R/hybrid_ablation.py --cands pool_dev_aug_ilm \
    --only full --workers 6 --out reports/abl_pool_dev_aug_ilm.csv
/workspace/simbig/envs/omni/.venv/bin/python $R/compare_runs.py --a abl:abl_pool_dev_aug_ilm:full \
    --b abl:hybrid_ablation:full
echo "EXP34 ILM DONE"
cd /workspace/simbig/envs/omni
uv run python $R/train_llm.py --run llm7b_lora_encdec \
    --card omniASR_LLM_7B_v2 --encoder lora --decoder lora --rank 16 \
    --sources scripted:1,spont_gold:3,pseudo_old:3,pseudo_rest:1 \
    --pseudo pseudo_old=$P/pseudo_v1_old.parquet,pseudo_rest=$P/pseudo_v1_rest.parquet \
    --steps 1500 --warmup 100 --lr-lora 2e-4 --max-batch-seconds 60 --grad-accum 4 \
    --eval-every 300 --eval-clips 150 --workers 8
uv run python $R/infer_llm.py --name llm7b_lora_encdec --card omniASR_LLM_7B_v2 --decoder-lora 16 \
    --encoder-lora 16 --weights $A/runs/llm7b_lora_encdec/best_trainable.pt
cd /workspace/simbig/repo
/workspace/simbig/envs/omni/.venv/bin/python $R/compare_runs.py --a llm7b_lora_encdec --b llm7b_lora_v2
echo "EXP34 DONE"
