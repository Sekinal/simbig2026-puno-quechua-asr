#!/usr/bin/env bash
# Experiment 51 (server A): the enc+dec LoRA recipe (exp34b) with the "strong" SimCLR-style
# distortion set (qxp_asr.augment: continuous speed 0.8-1.25, noise on half the clips at 5-30 dB,
# synthetic reverb, random band-limiting) instead of the mild one. The adapted encoder gains nothing
# on OOD radio and spontaneous speech, which differ from Common Voice in channel, room and voice:
# does distortion robustness help there (and keep the read-speech gain)?
set -euo pipefail
cd /workspace/simbig/envs/omni
export HF_HOME=/workspace/simbig/cache/hf PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=/workspace/simbig/repo/scripts; A=/workspace/simbig/artifacts; P=$A/pseudo
uv run python $R/train_llm.py --run llm7b_lora_encdec_aug --card omniASR_LLM_7B_v2 \
    --encoder lora --decoder lora --rank 16 --augment strong \
    --sources scripted:1,spont_gold:3,pseudo_old:3,pseudo_rest:1 \
    --pseudo pseudo_old=$P/pseudo_v1_old.parquet,pseudo_rest=$P/pseudo_v1_rest.parquet \
    --steps 1500 --warmup 100 --lr-lora 2e-4 --max-batch-seconds 60 --grad-accum 4 \
    --eval-every 300 --eval-clips 150 --workers 8
uv run python $R/infer_llm.py --name llm7b_lora_encdec_aug --card omniASR_LLM_7B_v2 \
    --decoder-lora 16 --encoder-lora 16 --weights $A/runs/llm7b_lora_encdec_aug/best_trainable.pt
cd /workspace/simbig/repo
/workspace/simbig/envs/omni/.venv/bin/python $R/compare_runs.py --a llm7b_lora_encdec_aug \
    --b llm7b_lora_encdec
echo "EXP51 DONE"
