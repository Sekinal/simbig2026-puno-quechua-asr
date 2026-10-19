#!/usr/bin/env bash
# Experiment 59 (server A, after exp60; the alignment posteriors live on A): information or
# plasticity? Contrast probes and CKA on the pretrained
# 300M and 1B encoders (and their encoder-only LoRA, exp54), as for 7B in exp56. If the small
# encoders encode the contrasts as well as the 7B one, the scale effect of exp54/57 is about how
# the update can use the information, not about the information.
set -euo pipefail
cd /workspace/simbig/repo
export HF_HOME=/workspace/simbig/cache/hf PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY="env PYTHONPATH=src /workspace/simbig/envs/omni/.venv/bin/python"
for size in 300M 1B; do
  tag=$(echo $size | tr 'A-Z' 'a-z')
  $PY scripts/encoder_probe.py --card omniASR_LLM_${size}_v2 --adapted llm${tag}_lora_enc \
      --adapted-seed "" --tag _$tag
  echo "EXP59 $size DONE"
done
echo "EXP59 DONE"
