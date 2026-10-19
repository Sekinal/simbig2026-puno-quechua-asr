#!/usr/bin/env bash
# Experiment 45 (server B, after exp43): encoder adaptation aimed at spontaneous (elderly) speech. The
# enc+dec LoRA gains nothing on spontaneous speech, which has little clean in-domain audio; here the
# untranscribed pool enters with the cross-fitted hybrid labels (exp27: 16 % of labels changed vs
# pseudo_v1; elderly "old" part 2.9 h) and spontaneous sources are up-weighted.
set -euo pipefail
until grep -q "EXP43 DONE" /workspace/simbig/logs/exp43.log; do sleep 120; done
cd /workspace/simbig/envs/omni
export HF_HOME=/workspace/simbig/cache/hf PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=/workspace/simbig/repo/scripts; A=/workspace/simbig/artifacts; P=$A/pseudo
uv run python $R/train_llm.py --run llm7b_lora_encdec_sp --card omniASR_LLM_7B_v2 \
    --encoder lora --decoder lora --rank 16 \
    --sources scripted:1,spont_gold:4,pseudo_old:4,pseudo_rest:2 \
    --pseudo pseudo_old=$P/hyb_pool_xfit_old.parquet,pseudo_rest=$P/hyb_pool_xfit_rest.parquet \
    --steps 1500 --warmup 100 --lr-lora 2e-4 --max-batch-seconds 60 --grad-accum 4 \
    --eval-every 300 --eval-clips 150 --workers 8
uv run python $R/infer_llm.py --name llm7b_lora_encdec_sp --card omniASR_LLM_7B_v2 \
    --decoder-lora 16 --encoder-lora 16 --weights $A/runs/llm7b_lora_encdec_sp/best_trainable.pt
echo "EXP45 DONE"
