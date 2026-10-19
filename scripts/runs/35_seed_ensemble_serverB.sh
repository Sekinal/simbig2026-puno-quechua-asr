#!/usr/bin/env bash
# Experiment 35 (server B): use the training-seed variance (exp32: LoRA v2 recipe, seeds 0/1/2).
# (1) score the perturbed-view pool with seeds 1 and 2 (extra hybrid scorers; the hybrid test runs on
#     server A with scripts/hybrid_ablation.py); (2) an exact soup of the three seeds (rank 48) as a
#     single model.
set -euo pipefail
cd /workspace/simbig/envs/omni
export HF_HOME=/workspace/simbig/cache/hf PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=/workspace/simbig/repo/scripts; A=/workspace/simbig/artifacts
uv run python $R/llm_candidates.py --ctc pool_dev_aug --name pool_dev_seeds --nbest none \
    --adapter s1=$A/runs/llm7b_lora_v2_s1/best_trainable.pt \
    --adapter s2=$A/runs/llm7b_lora_v2_s2/best_trainable.pt
uv run python $R/lora_soup.py --out $A/runs/llm7b_v2_seedsoup/best_trainable.pt \
    --adapter $A/runs/llm7b_lora_v2/best_trainable.pt \
    --adapter $A/runs/llm7b_lora_v2_s1/best_trainable.pt \
    --adapter $A/runs/llm7b_lora_v2_s2/best_trainable.pt
uv run python $R/infer_llm.py --name llm7b_v2_seedsoup --card omniASR_LLM_7B_v2 --decoder-lora 48 \
    --weights $A/runs/llm7b_v2_seedsoup/best_trainable.pt
echo "EXP35 DONE"
