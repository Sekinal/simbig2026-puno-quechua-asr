#!/usr/bin/env bash
# Experiment 18 (server B, after exp16): ASR "back-translation" — synthetic speech for external text.
# 20k respelled external sentences (dev-mode LM text, no dev/itest sentences) were synthesised with
# MMS-TTS Cusco Quechua (scripts/tts_synth.py, speaking rate 0.8-1.1; ~28 h). Round trip: our CTC v3
# transcribes the synthetic voice as well as human Puno speech on dev unseen sentences
# (14.92 vs 14.64 WER). The recipe is the text-CPT LoRA (llm7b_textcpt) plus the synthetic pairs at
# half weight, so the comparison with llm7b_textcpt isolates the paired synthetic audio
# (Hayashi 2018; Rosenberg 2019; Rossenbach 2020: synthetic speech helps most for unseen text).
set -euo pipefail
until grep -q "ALL DONE" /workspace/simbig/logs/exp16.log; do sleep 120; done
until [ -f /workspace/simbig/artifacts/tts/ext20k/manifest.parquet ]; do sleep 120; done
cd /workspace/simbig/envs/omni
export HF_HOME=/workspace/simbig/cache/hf PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=/workspace/simbig/repo/scripts; A=/workspace/simbig/artifacts; P=$A/pseudo
uv run python $R/train_llm.py --run llm7b_tts --card omniASR_LLM_7B_v2 --encoder frozen \
    --decoder lora --rank 16 --sources scripted:1,spont_gold:3,pseudo_old:3,pseudo_rest:1,tts:0.5 \
    --pseudo pseudo_old=$P/pseudo_v1_old.parquet,pseudo_rest=$P/pseudo_v1_rest.parquet,tts=$A/tts/ext20k/manifest.parquet \
    --steps 1500 --warmup 100 --lr-lora 3e-4 --max-batch-seconds 120 --grad-accum 2 \
    --eval-every 300 --eval-clips 150 --workers 8 \
    --text-corpus $A/lm/dev_v2/all.txt --text-weight 0.3 --text-batch 16 --grad-ckpt
uv run python $R/infer_llm.py --name llm7b_tts --card omniASR_LLM_7B_v2 --decoder-lora 16 \
    --weights $A/runs/llm7b_tts/best_trainable.pt
echo "TTS DONE"
