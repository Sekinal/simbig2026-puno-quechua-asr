#!/usr/bin/env bash
# Experiment 40 (server B, GPU part): the encoder+decoder LoRA ("ed", exp34) in the hybrid, dev mode,
# on the evaluation clips of the perturbed-view pool (pool_ctrl_aug, 1,298 clips): ed adds beam-8
# candidates and 6 perturbed-view decodes, and every candidate is scored by zs, v1, v2 and ed.
# The hybrid test (3 from-scratch weight searches) runs on server A (exp40a).
set -euo pipefail
cd /workspace/simbig/envs/omni
export HF_HOME=/workspace/simbig/cache/hf PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=/workspace/simbig/repo/scripts; A=/workspace/simbig/artifacts
uv run python $R/llm_candidates.py --ctc pool_ctrl_aug --name pool_ctrl_ed --nbest ed \
    --views 6 --view-adapters ed --encoder-rank 16 --adapter zs \
    --adapter v1=$A/runs/llm7b_lora_v1/best_trainable.pt \
    --adapter v2=$A/runs/llm7b_lora_v2/best_trainable.pt \
    --adapter ed=$A/runs/llm7b_lora_encdec/best_trainable.pt
echo "EXP40 DONE"
