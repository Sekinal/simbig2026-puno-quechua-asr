#!/usr/bin/env bash
# Experiment 12: hybrid CTC + multi-LLM likelihood rescoring (joint_rescore2), applied to the
# final systems. Weights are tuned on dev-mode systems (dev sentences unseen) and transferred to
# the final-mode systems; spontaneous weights are re-tuned on dev_spont (speakers still unseen).
# Afterwards the follow-up experiments of 11_post_final.sh run.
set -euo pipefail
until grep -q "ALL DONE" /workspace/simbig/logs/exp10.log; do sleep 120; done
cd /workspace/simbig/envs/omni
export HF_HOME=/workspace/simbig/cache/hf PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=/workspace/simbig/repo/scripts; A=/workspace/simbig/artifacts

# dev-mode systems: tune the feature weights
uv run python $R/llm_candidates.py --ctc ctc_v3 --name pool_dev --adapter zs \
    --adapter v1=$A/runs/llm7b_lora_v1/best_trainable.pt \
    --adapter v2=$A/runs/llm7b_lora_v2/best_trainable.pt
uv run python $R/joint_rescore2.py --cands pool_dev --post ctc1b_v3_soup_wise0.85 --name hybrid_dev

# final-mode systems: transfer scripted weights, re-tune spontaneous
uv run python $R/ctc_nbest.py --post ctc1b_final --lm final_v1 --name ctc_final
uv run python $R/llm_candidates.py --ctc ctc_final --name pool_final --adapter zs \
    --adapter v1=$A/runs/llm7b_final_v1/last_trainable.pt \
    --adapter v2=$A/runs/llm7b_final_v2/last_trainable.pt
uv run python $R/joint_rescore2.py --cands pool_final --post ctc1b_final --lm final_v1 \
    --weights hybrid_dev.weights.json --retune-spont --name hybrid_final
cd /workspace/simbig/repo
UV_LINK_MODE=copy uv run python $R/make_submission.py --pred hybrid_final --pred joint_final \
    --pred joint_v3 --name qxp_final_hybrid
echo "HYBRID DONE"
/workspace/simbig/repo/scripts/runs/11_post_final.sh
