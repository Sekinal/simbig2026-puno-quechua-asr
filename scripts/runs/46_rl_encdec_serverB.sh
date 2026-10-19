#!/usr/bin/env bash
# Experiment 46 (server B): does RL work once the acoustic model is adapted? Same sentence cross-fit
# protocol as exp20 (half 0 updates, 400 steps, 4 clips x group 4), but starting from the enc+dec
# LoRA (exp34b) with its encoder LoRA frozen: GSPO vs the SFT control. Then re-decoded with infer_llm
# and compared with the enc+dec start (scripts/paper_stats.py designs).
set -euo pipefail
cd /workspace/simbig/envs/omni
export HF_HOME=/workspace/simbig/cache/hf PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=/workspace/simbig/repo/scripts; A=/workspace/simbig/artifacts
ED=$A/runs/llm7b_lora_encdec/best_trainable.pt
COMMON="--init $ED --encoder-rank 16 --sentence-half 0 --mode org --steps 400 --clips 4 --group 4
  --eval-every 400"
uv run python $R/train_rl.py --run rl46_gspo $COMMON --objective gspo
uv run python $R/train_rl.py --run rl46_sft $COMMON --objective sft
for a in gspo sft; do
  uv run python $R/infer_llm.py --name rl46_${a}_s400 --decoder-lora 16 --encoder-lora 16 \
      --weights $A/runs/rl46_$a/step400_trainable.pt --roles dev_scripted,itest_scripted,dev_spont,ood
done
echo "EXP46 DONE"
