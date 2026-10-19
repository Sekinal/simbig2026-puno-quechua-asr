#!/usr/bin/env bash
# Experiment 15 (server A): hybrid rescoring with FOUR LLM scorers (zero-shot, LoRA v1, LoRA v2,
# LoRA v2 + decoder text-CPT — the best OOD single model). Weights tuned on dev-mode systems, applied
# to final systems (spontaneous re-tuned on dev_spont). Then the RL pilot (exp11, memory-fixed).
set -euo pipefail
cd /workspace/simbig/envs/omni
export HF_HOME=/workspace/simbig/cache/hf PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=/workspace/simbig/repo/scripts; A=/workspace/simbig/artifacts
uv run python $R/llm_candidates.py --ctc ctc_v3 --name pool_dev_tc --adapter zs \
    --adapter v1=$A/runs/llm7b_lora_v1/best_trainable.pt \
    --adapter v2=$A/runs/llm7b_lora_v2/best_trainable.pt \
    --adapter tc=$A/runs/llm7b_textcpt/best_trainable.pt
uv run python $R/joint_rescore2.py --cands pool_dev_tc --post ctc1b_v3_soup_wise0.85 --name hybrid_dev_tc
uv run python $R/llm_candidates.py --ctc ctc_final --name pool_final_tc --adapter zs \
    --adapter v1=$A/runs/llm7b_final_v1/last_trainable.pt \
    --adapter v2=$A/runs/llm7b_final_v2/last_trainable.pt \
    --adapter tc=$A/runs/llm7b_final_tc/last_trainable.pt
uv run python $R/joint_rescore2.py --cands pool_final_tc --post ctc1b_final --lm final_v1 \
    --weights hybrid_dev_tc.weights.json --retune-spont --name hybrid_final_tc
cd /workspace/simbig/repo
UV_LINK_MODE=copy uv run python $R/make_submission.py --pred hybrid_final_tc --pred hybrid_final \
    --pred joint_final --name qxp_final_hybrid_tc
echo "HYBRID TC DONE"
/workspace/simbig/repo/scripts/runs/11_post_final.sh
