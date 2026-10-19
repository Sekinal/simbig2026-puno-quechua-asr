#!/usr/bin/env bash
# Experiment 01: zero-shot omniASR baselines and the first CTC-1B fine-tune (dev protocol).
set -euo pipefail
cd /workspace/simbig/envs/omni
export HF_HOME=/workspace/simbig/cache/hf PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=/workspace/simbig/repo/scripts
uv run python $R/omni_infer.py omniASR_CTC_1B_v2 --roles dev_scripted,dev_spont,test --name zs_ctc1b
uv run python $R/omni_infer.py omniASR_LLM_7B_v2 --roles dev_scripted,dev_spont,test --name zs_llm7b
uv run python $R/train_ctc.py --run ctc1b_v2 --steps 6000 --eval-every 500 --workers 8 \
    --sources scripted:1,spont_gold:3,spont_silver:1     # lr 3e-5, 200 s batches, tri-stage
