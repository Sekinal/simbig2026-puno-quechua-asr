#!/usr/bin/env bash
# Experiment 05: hypotheses on the ~30 h unlabelled spontaneous pool for agreement-filtered
# self-training (docs/LITERATURE.md §c/§g: noisy student, IPL, multi-system agreement).
# Voters: LoRA LLM-7B (best system), CTC-1B WiSE-0.85, and the organizer silver labels
# (= zero-shot omniASR-LLM-7B), giving three partly independent systems.
set -euo pipefail
cd /workspace/simbig/envs/omni
export HF_HOME=/workspace/simbig/cache/hf PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=/workspace/simbig/repo/scripts
uv run python $R/infer_ctc.py --name pool_ctc1b_v2_wise0.85 --pool \
    --checkpoint /workspace/simbig/artifacts/merged/ctc1b_v2_wise0.85.pt
uv run python $R/infer_llm.py --name pool_llm7b_lora_v1 --pool --card omniASR_LLM_7B_v2 \
    --decoder-lora 16 --weights /workspace/simbig/artifacts/runs/llm7b_lora_v1/best_trainable.pt
echo "ALL DONE"
