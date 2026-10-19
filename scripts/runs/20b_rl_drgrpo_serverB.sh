#!/usr/bin/env bash
# Experiment 20b (server B, after exp20): 4th arm of the sentence cross-fit, Dr. GRPO token-sum
# policy gradient (LITERATURE §g.12: in beam groups the shared prefixes cancel, so credit lands on
# the tokens where hypotheses diverge). Same clips, steps and lr as exp20.
set -euo pipefail
until grep -q "EXP20 DONE" /workspace/simbig/logs/exp20.log; do sleep 120; done
cd /workspace/simbig/envs/omni
export HF_HOME=/workspace/simbig/cache/hf PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=/workspace/simbig/repo/scripts; A=/workspace/simbig/artifacts
V2=$A/runs/llm7b_lora_v2/best_trainable.pt
uv run python $R/train_rl.py --run rl20_drgrpo --init $V2 --sentence-half 0 --mode org \
    --steps 400 --clips 4 --group 4 --eval-every 100 --objective drgrpo
echo "EXP20B DONE"
