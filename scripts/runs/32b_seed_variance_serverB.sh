#!/usr/bin/env bash
# Experiment 32b (server B): training-seed variance. The LoRA v2 recipe (exp06) retrained with
# seed 1; with the original (seed 0) this gives the spread due to training alone, against which the
# single-model differences in the paper are judged.
# Then the RL arms of exp20/20b/24 are re-decoded with infer_llm (same decoding as LoRA v2) so their
# changes get paired per-clip confidence intervals (their training logs kept only corpus WER).
set -euo pipefail
cd /workspace/simbig/envs/omni
export HF_HOME=/workspace/simbig/cache/hf PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=/workspace/simbig/repo/scripts; A=/workspace/simbig/artifacts; P=$A/pseudo
uv run python $R/train_llm.py --run llm7b_lora_v2_s1 --seed 1 \
    --card omniASR_LLM_7B_v2 --encoder frozen --decoder lora --rank 16 \
    --sources scripted:1,spont_gold:3,pseudo_old:3,pseudo_rest:1 \
    --pseudo pseudo_old=$P/pseudo_v1_old.parquet,pseudo_rest=$P/pseudo_v1_rest.parquet \
    --steps 1500 --warmup 100 --lr-lora 3e-4 --max-batch-seconds 120 --grad-accum 2 \
    --eval-every 300 --eval-clips 150 --workers 8
uv run python $R/infer_llm.py --name llm7b_lora_v2_s1 --decoder-lora 16 \
    --weights $A/runs/llm7b_lora_v2_s1/best_trainable.pt
for run in rl20_gspo:400 rl20_sft:400 rl20_mwer:400 rl20_drgrpo:400 rl24_mwer_beam_2_sample_3_aug_3:150; do
  name=${run%%:*}; step=${run##*:}
  uv run python $R/infer_llm.py --name ${name}_s$step --decoder-lora 16 \
      --weights $A/runs/$name/step${step}_trainable.pt --roles dev_scripted,itest_scripted,dev_spont,ood
done
# control: LoRA v2 itself decoded again (decoding nondeterminism, the floor for the RL deltas)
uv run python $R/infer_llm.py --name llm7b_lora_v2_redecode --decoder-lora 16 \
    --weights $A/runs/llm7b_lora_v2/best_trainable.pt --roles dev_scripted,itest_scripted,dev_spont,ood
echo "EXP32 DONE"
