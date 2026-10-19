#!/usr/bin/env bash
# Experiment 20 (server B): does sequence-level RL on NEW sentences generalise to other new
# sentences better than supervised fine-tuning on the same clips? (provided data only)
# Sentence cross-fit: the 400 held-out sentences are split in two halves (md5 of sentence_id).
# Policy updates use training speakers reading half-0 sentences (role holdout_sent; the LoRA v2
# init has never seen these texts, so N-best rewards vary) + gold spontaneous clips; evaluation
# uses dev + itest clips of half-1 sentences (unseen speakers AND unseen sentences), dev_spont, OOD.
# Arms, all from LoRA v2, same clips/steps/lr: GSPO (GLM-5-style masked trust region), MWER, SFT.
set -euo pipefail
cd /workspace/simbig/envs/omni
export HF_HOME=/workspace/simbig/cache/hf PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=/workspace/simbig/repo/scripts; A=/workspace/simbig/artifacts
V2=$A/runs/llm7b_lora_v2/best_trainable.pt
COMMON="--init $V2 --sentence-half 0 --mode org --steps 400 --clips 4 --group 4 --eval-every 100"
uv run python $R/train_rl.py --run rl20_gspo $COMMON --objective gspo
uv run python $R/train_rl.py --run rl20_sft $COMMON --objective sft
uv run python $R/train_rl.py --run rl20_mwer $COMMON --objective mwer
echo "EXP20 DONE"
