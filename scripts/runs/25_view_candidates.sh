#!/usr/bin/env bash
# Experiment 25: perturbed-view candidates in the provided-data hybrid. Exploration sweep: greedy
# decodes of perturbed audio views find better transcripts far more often than beam search, and
# view agreement alone beats the top-1 on dev unseen (11.95 vs 13.26). Here zs and v2 decode 6
# views per clip (llm_candidates --views); the new candidates are scored under every adapter on
# the CLEAN audio and the hybrid gets a view-agreement feature (joint_rescore2 --view-feature),
# searched from the hybrid_dev_in weights. Usage: 25_view_candidates.sh dev|final
set -euo pipefail
cd /workspace/simbig/envs/omni
export HF_HOME=/workspace/simbig/cache/hf PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=/workspace/simbig/repo/scripts; A=/workspace/simbig/artifacts
if [ "$1" = dev ]; then
  uv run python $R/llm_candidates.py --ctc pool_dev_in --name pool_dev_aug --nbest none \
      --views 6 --view-adapters zs,v2 --adapter zs \
      --adapter v1=$A/runs/llm7b_lora_v1/best_trainable.pt \
      --adapter v2=$A/runs/llm7b_lora_v2/best_trainable.pt
  uv run python $R/joint_rescore2.py --cands pool_dev_aug --post ctc1b_v3_soup_wise0.85 \
      --lm dev_v2 --lm-text indomain --view-feature --init-weights hybrid_dev_in.weights.json \
      --name hybrid_dev_aug --workers 8
  cd /workspace/simbig/repo && UV_LINK_MODE=copy uv run python $R/oracle_gap.py \
      --cands pool_dev_aug --pred hybrid_dev_aug
else
  uv run python $R/llm_candidates.py --ctc pool_final_in --name pool_final_aug --nbest none \
      --views 6 --view-adapters zs,v2 --adapter zs \
      --adapter v1=$A/runs/llm7b_final_v1/last_trainable.pt \
      --adapter v2=$A/runs/llm7b_final_v2/last_trainable.pt
fi
echo "EXP25 $1 DONE"
