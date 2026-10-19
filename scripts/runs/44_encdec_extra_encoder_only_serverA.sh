#!/usr/bin/env bash
# Experiment 44 (server A, after exp41b). What else should be adapted, and what does what?
# (a) "x": the enc+dec LoRA recipe (exp34b) plus full training of the never-trained small modules:
#     the encoder->LLM connector (encoder_proj, 8.4 M), the language embeddings (only the qxp row gets
#     gradient) and every Layer/RMSNorm (1.2 M); lr 5e-5 for these.
# (b) encoder-only LoRA (decoder frozen): does the decoder LoRA still matter once the encoder is
#     adapted? (acoustics vs language decomposition)
set -euo pipefail
until grep -q "EXP41B DONE" /workspace/simbig/logs/exp41b.log; do sleep 120; done
cd /workspace/simbig/envs/omni
export HF_HOME=/workspace/simbig/cache/hf PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=/workspace/simbig/repo/scripts; A=/workspace/simbig/artifacts; P=$A/pseudo
COMMON="--card omniASR_LLM_7B_v2 --rank 16 --sources scripted:1,spont_gold:3,pseudo_old:3,pseudo_rest:1
  --pseudo pseudo_old=$P/pseudo_v1_old.parquet,pseudo_rest=$P/pseudo_v1_rest.parquet
  --steps 1500 --warmup 100 --lr-lora 2e-4 --max-batch-seconds 60 --grad-accum 4
  --eval-every 300 --eval-clips 150 --workers 8"
uv run python $R/train_llm.py --run llm7b_lora_encdec_x --encoder lora --decoder lora $COMMON \
    --extra encoder_proj,lang_embeddings,norms --lr-extra 5e-5
uv run python $R/infer_llm.py --name llm7b_lora_encdec_x --card omniASR_LLM_7B_v2 --decoder-lora 16 \
    --encoder-lora 16 --weights $A/runs/llm7b_lora_encdec_x/best_trainable.pt
echo "EXP44A DONE"
uv run python $R/train_llm.py --run llm7b_lora_enc --encoder lora --decoder frozen $COMMON
uv run python $R/infer_llm.py --name llm7b_lora_enc --card omniASR_LLM_7B_v2 \
    --encoder-lora 16 --weights $A/runs/llm7b_lora_enc/best_trainable.pt
echo "EXP44 DONE"
