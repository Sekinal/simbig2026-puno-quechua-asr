#!/usr/bin/env bash
# Experiment 58 (server A, after exp51): where in its own spectrum does the encoder update work?
# Weight surgery on trained adapters (scripts/lora_surgery.py, built beforehand on CPU), inference
# only, no test set:
# * encoder-only LoRA (exp44b) cut to its top 1/2/4/8 singular directions per projection (energy
#   0.61 / 0.74 / 0.85 / 0.93 of the update);
# * the exact average of the two enc+dec seeds (rank 32), that average cut back to rank 16
#   (the part the seeds share), and the average with the encoder cut to rank 4.
set -euo pipefail
until grep -q "EXP51 DONE" /workspace/simbig/logs/exp51.log; do sleep 120; done
cd /workspace/simbig/envs/omni
export HF_HOME=/workspace/simbig/cache/hf PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=/workspace/simbig/repo/scripts; A=/workspace/simbig/artifacts
ROLES=dev_scripted,dev_spont,itest_scripted,itest_spont,ood
for v in surg_enc_top1:1:0 surg_enc_top2:2:0 surg_enc_top4:4:0 surg_enc_top8:8:0 \
         surg_ed_avg:32:32 surg_ed_avg_top16:16:16 surg_ed_avg_enctop4:32:32; do
  IFS=: read -r name er dr <<< "$v"
  uv run python $R/infer_llm.py --name llm7b_$name --card omniASR_LLM_7B_v2 --roles $ROLES \
      --encoder-lora $er --decoder-lora $dr --weights $A/runs/$name/best_trainable.pt
  echo "EXP58 $name DONE"
done
echo "EXP58 DONE"
