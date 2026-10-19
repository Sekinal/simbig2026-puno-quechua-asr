#!/usr/bin/env bash
# Experiment 11 (after the final submission exists): decoding and training efficiency/quality follow-ups.
#  a) LLM beam size at inference (official pipeline decodes greedily, nbest=1)
#  b) sequence-level RL (GSPO, then MWER) on top of LoRA v2, gold rewards only
#  c) research runs 08 (encoder+decoder LoRA) and 09 (organizer-equivalent protocol)
#  Expert iteration (exp17, the top-ranked RL experiment of reports/rl_experiment_plan.md) runs first.
set -euo pipefail
/workspace/simbig/repo/scripts/runs/17_expert_iteration.sh
cd /workspace/simbig/envs/omni
export HF_HOME=/workspace/simbig/cache/hf PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=/workspace/simbig/repo/scripts; A=/workspace/simbig/artifacts
V2=$A/runs/llm7b_lora_v2/best_trainable.pt
for b in 4 8; do
  uv run python $R/infer_llm.py --name llm7b_lora_v2_beam$b --decoder-lora 16 --weights $V2 \
      --beam $b --roles dev_scripted,dev_spont,itest_scripted,itest_spont,ood,test
done
# RL pilots on training-set gold moved to exp20 (server B, sentence cross-fit protocol): the
# reward bench shows training clips give almost no reward variance.
echo "ALL DONE"
