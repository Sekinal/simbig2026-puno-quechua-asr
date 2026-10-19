#!/usr/bin/env bash
# Experiment 43 (server B): exact soup of the two encoder+decoder LoRA seeds (exp34b, exp42; rank 32
# on encoder and decoder) scored on the dev score-only pool, to test it as the hybrid's 4th scorer
# instead of the single seed (hybrid test on server A).
set -euo pipefail
cd /workspace/simbig/envs/omni
export HF_HOME=/workspace/simbig/cache/hf PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=/workspace/simbig/repo/scripts; A=/workspace/simbig/artifacts
uv run python $R/lora_soup.py --out $A/runs/llm7b_encdec_soup2/best_trainable.pt \
    --adapter $A/runs/llm7b_lora_encdec/best_trainable.pt \
    --adapter $A/runs/llm7b_lora_encdec_s1/best_trainable.pt
uv run python $R/llm_candidates.py --ctc pool_ctrl_ed_scoreonly --name pool_ctrl_edsoup --nbest none \
    --rank 32 --encoder-rank 32 --adapter edsoup=$A/runs/llm7b_encdec_soup2/best_trainable.pt
echo "EXP43 DONE"
