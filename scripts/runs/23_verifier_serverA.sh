#!/usr/bin/env bash
# Experiment 23 (server A, after exp17/exp11 and exp22 on server B): "Verifier" stage of the
# Spectrum -> Verifier -> Signal program. A learned outcome reward model for ASR candidates
# (scripts/train_verifier.py: listwise MWER on cross-fit candidate lists, init LoRA v2), then
#  a) as a 4th LLM scorer in the provided-data hybrid (hybrid_dev_ver), and
#  b) in the reward bench (pairwise ranking accuracy vs the ~64-70 % of reference-free rewards).
# Stage 3 (RL on unlabelled audio with this reward) only if b) clearly beats the existing rewards.
set -euo pipefail
until grep -q "ALL DONE" /workspace/simbig/logs/exp19.log; do sleep 120; done
B="ssh -p <SERVER_B_PORT> root@<SERVER_B_HOST>"
until $B 'grep -q "EXP22 DONE" /workspace/simbig/logs/exp22.log'; do sleep 300; done
for k in 0 1; do
  scp -q -P <SERVER_B_PORT> root@<SERVER_B_HOST>:/workspace/simbig/artifacts/cands/crossfit$k.parquet \
      /workspace/simbig/artifacts/cands/
done
cd /workspace/simbig/envs/omni
export HF_HOME=/workspace/simbig/cache/hf PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=/workspace/simbig/repo/scripts; A=/workspace/simbig/artifacts
uv run python $R/train_verifier.py --run verifier_v1 --init $A/runs/llm7b_lora_v2/best_trainable.pt \
    --cands crossfit0,crossfit1 --steps 1500 --eval-every 250
VER=$(ls $A/runs/verifier_v1/step*_trainable.pt | sort -V | tail -1)
uv run python $R/llm_candidates.py --ctc pool_dev_in --name pool_dev_ver --nbest none \
    --adapter zs --adapter v1=$A/runs/llm7b_lora_v1/best_trainable.pt \
    --adapter v2=$A/runs/llm7b_lora_v2/best_trainable.pt --adapter ver=$VER
uv run python $R/joint_rescore2.py --cands pool_dev_ver --post ctc1b_v3_soup_wise0.85 \
    --lm dev_v2 --lm-text indomain --name hybrid_dev_ver --workers 8
cd /workspace/simbig/repo
/workspace/simbig/envs/omni/.venv/bin/python scripts/reward_bench.py --cands pool_dev_ver \
    --post ctc1b_v3_soup_wise0.85 --lm-text indomain --weights hybrid_dev_ver.weights.json \
    --out reward_bench_ver
echo "EXP23 DONE"
