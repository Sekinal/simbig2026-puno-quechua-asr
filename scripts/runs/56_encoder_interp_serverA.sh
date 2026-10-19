#!/usr/bin/env bash
# Experiment 56 (server A, after exp53): inside the adapted encoder.
# (a) Per-layer CKA (pretrained vs enc+dec LoRA, and across its two seeds) and phonetic-contrast
#     probes (ejective / aspirated / q-vs-k) on force-aligned stops: scripts/encoder_probe.py.
# (b) Causal block ablation of the encoder-only LoRA (exp44b): keep its update in one block of
#     layers, zero it elsewhere (infer_llm --keep-encoder-layers). Quarters, halves (to set against
#     exp53's trained halves) and three-quarter ranges (= one end quarter removed). No test set.
set -euo pipefail
until grep -q "EXP53 DONE" /workspace/simbig/logs/exp53.log; do sleep 120; done
cd /workspace/simbig/envs/omni
export HF_HOME=/workspace/simbig/cache/hf PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=/workspace/simbig/repo/scripts; A=/workspace/simbig/artifacts
probe() { (cd /workspace/simbig/repo && PYTHONPATH=src /workspace/simbig/envs/omni/.venv/bin/python $R/encoder_probe.py "$@"); }
if probe --limit 40; then probe && echo "EXP56 PROBE DONE" || echo "EXP56 PROBE FAILED"
else echo "EXP56 PROBE SMOKE FAILED"; fi
for range in 0:32 32:64 64:96 96:128 0:64 64:128 32:128 0:96; do
  uv run python $R/infer_llm.py --name llm7b_lora_enc_keep${range/:/_} --card omniASR_LLM_7B_v2 \
      --encoder-lora 16 --weights $A/runs/llm7b_lora_enc/best_trainable.pt \
      --roles dev_scripted,dev_spont,itest_scripted,itest_spont,ood --keep-encoder-layers $range
  echo "EXP56 keep $range DONE"
done
echo "EXP56 DONE"
