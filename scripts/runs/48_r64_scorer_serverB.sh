#!/usr/bin/env bash
# Experiment 48 (server B): the rank-64 enc+dec LoRA (exp47) scored on the dev score-only pool, to
# test it as the hybrid's 4th scorer instead of the rank-16 one (hybrid test on server A).
set -euo pipefail
cd /workspace/simbig/envs/omni
export HF_HOME=/workspace/simbig/cache/hf PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=/workspace/simbig/repo/scripts; A=/workspace/simbig/artifacts
uv run python $R/llm_candidates.py --ctc pool_ctrl_ed_scoreonly --name pool_ctrl_r64 --nbest none \
    --rank 64 --encoder-rank 64 --adapter r64=$A/runs/llm7b_lora_encdec_r64/best_trainable.pt
echo "EXP48 DONE"
