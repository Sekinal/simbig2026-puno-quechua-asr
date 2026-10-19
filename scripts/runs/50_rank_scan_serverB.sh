#!/usr/bin/env bash
# Experiment 50 (server B, after exp48): LoRA rank scan for the enc+dec recipe (exp34b). Rank 16 gives
# itest unseen 16.0 / 15.3 (two seeds), rank 64 14.8 (exp47, n.s. vs 16). Rank 128 first (could still
# reach the submission), then rank 32 (completes the curve). One seed per rank: a trend, not a test.
set -euo pipefail
until grep -q "EXP48 DONE" /workspace/simbig/logs/exp48.log; do sleep 120; done
cd /workspace/simbig/envs/omni
export HF_HOME=/workspace/simbig/cache/hf PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=/workspace/simbig/repo/scripts; A=/workspace/simbig/artifacts; P=$A/pseudo
for r in 128 32; do
  uv run python $R/train_llm.py --run llm7b_lora_encdec_r$r --card omniASR_LLM_7B_v2 \
      --encoder lora --decoder lora --rank $r \
      --sources scripted:1,spont_gold:3,pseudo_old:3,pseudo_rest:1 \
      --pseudo pseudo_old=$P/pseudo_v1_old.parquet,pseudo_rest=$P/pseudo_v1_rest.parquet \
      --steps 1500 --warmup 100 --lr-lora 2e-4 --max-batch-seconds 60 --grad-accum 4 \
      --eval-every 300 --eval-clips 150 --workers 8
  uv run python $R/infer_llm.py --name llm7b_lora_encdec_r$r --card omniASR_LLM_7B_v2 \
      --decoder-lora $r --encoder-lora $r --weights $A/runs/llm7b_lora_encdec_r$r/best_trainable.pt
  echo "EXP50 RANK $r DONE"
done
echo "EXP50 DONE"
