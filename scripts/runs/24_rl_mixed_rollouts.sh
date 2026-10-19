#!/usr/bin/env bash
# Experiment 24 (short runs): RL with exploration that actually finds better transcripts.
# Exploration sweep (scripts/pass_at_k.py): beam-8 beats its own top-1 on 6 % of held-out-sentence
# clips (2.8 distinct of 8); temperature-1 samples 14 %, greedy decodes of perturbed audio views
# 16.7 %, their union 24 % (oracle WER 8.35 vs top-1 14.86). Groups now mix beam, samples and
# perturbed-view decodes; every candidate is scored under the CLEAN audio, so the policy learns to
# produce from the original input what it found under perturbation. Sentence cross-fit protocol
# (exp20): updates on held-out sentences of half 0 only, evaluation on half 1, dev_spont, OOD.
# Usage: 24_rl_mixed_rollouts.sh <objective> [rollouts]
set -euo pipefail
OBJ=${1:-mwer}; MIX=${2:-beam:2,sample:3,aug:3}
cd /workspace/simbig/envs/omni
export HF_HOME=/workspace/simbig/cache/hf PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=/workspace/simbig/repo/scripts; A=/workspace/simbig/artifacts
uv run python $R/train_rl.py --run rl24_${OBJ}_${MIX//[:,]/_} \
    --init $A/runs/llm7b_lora_v2/best_trainable.pt --sentence-half 0 --mode org \
    --steps 150 --clips 2 --group 8 --rollouts $MIX --scripted-frac 1.0 \
    --objective $OBJ --eval-every 50
echo "EXP24 $OBJ DONE"
