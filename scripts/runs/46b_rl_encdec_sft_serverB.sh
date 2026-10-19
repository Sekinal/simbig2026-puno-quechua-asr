#!/usr/bin/env bash
# Experiment 46b (server B): rerun of exp46's SFT control, which ran out of memory because the frozen
# encoder LoRA was left with requires_grad=True (SFT backpropagates through the encoder; fixed in
# train_rl.py). The GSPO arm of exp46 finished and is valid (its encoder pass is cached without
# gradients and the optimizer held decoder parameters only). Then both arms are re-decoded.
set -euo pipefail
cd /workspace/simbig/envs/omni
export HF_HOME=/workspace/simbig/cache/hf PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=/workspace/simbig/repo/scripts; A=/workspace/simbig/artifacts
ED=$A/runs/llm7b_lora_encdec/best_trainable.pt
rm -rf $A/runs/rl46_sft
uv run python $R/train_rl.py --run rl46_sft --init $ED --encoder-rank 16 --sentence-half 0 \
    --mode org --steps 400 --clips 4 --group 4 --eval-every 400 --objective sft
for a in gspo sft; do
  uv run python $R/infer_llm.py --name rl46_${a}_s400 --decoder-lora 16 --encoder-lora 16 \
      --weights $A/runs/rl46_$a/step400_trainable.pt --roles dev_scripted,itest_scripted,dev_spont,ood
done
echo "EXP46B DONE"
