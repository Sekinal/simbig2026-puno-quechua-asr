#!/usr/bin/env bash
# Experiment 28 (server A, after exp27a): exact soup of the four decoder-LoRA adapters (v1 silver,
# v2 ROVER pseudo-labels, v3 hybrid labels, v4 hybrid + perturbed-view labels) as one rank-64
# adapter (scripts/lora_soup.py: concatenated factors = mean of the B@A updates; model soups,
# Wortsman et al. 2022). Single-model evaluation against the individual adapters.
set -euo pipefail
until grep -qE "EXP27A DONE" /workspace/simbig/logs/exp27a.log; do sleep 120; done
cd /workspace/simbig/envs/omni
export HF_HOME=/workspace/simbig/cache/hf PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=/workspace/simbig/repo/scripts; A=/workspace/simbig/artifacts
mkdir -p $A/runs/llm7b_lora_v4
scp -q -P <SERVER_B_PORT> root@<SERVER_B_HOST>:$A/runs/llm7b_lora_v4/best_trainable.pt $A/runs/llm7b_lora_v4/
uv run python $R/lora_soup.py --out $A/runs/llm7b_soup4/best_trainable.pt \
    --adapter $A/runs/llm7b_lora_v1/best_trainable.pt --adapter $A/runs/llm7b_lora_v2/best_trainable.pt \
    --adapter $A/runs/llm7b_lora_v3/best_trainable.pt --adapter $A/runs/llm7b_lora_v4/best_trainable.pt
uv run python $R/infer_llm.py --name llm7b_soup4 --card omniASR_LLM_7B_v2 --decoder-lora 64 \
    --weights $A/runs/llm7b_soup4/best_trainable.pt
cd /workspace/simbig/repo
/workspace/simbig/envs/omni/.venv/bin/python $R/ladder.py llm7b_lora_v1 llm7b_lora_v2 llm7b_lora_v3 llm7b_soup4
echo "EXP28 DONE"
