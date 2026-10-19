#!/usr/bin/env bash
# Experiment 21 (server B, after exp20b): RL with a concentrated reward signal. In exp20 only ~25 %
# of groups carried signal (half of every step were gold spontaneous clips the SFT model already
# transcribes; many read clips are solved by every beam). Here (LITERATURE §g.12):
#  * difficulty pre-pass (rl_difficulty.py): beam-8 under LoRA v2; sample read clips by top-1 error
#    if the beam holds a better answer, with a 20 % uniform floor (GLM-4.5 / Kimi k1.5 curriculum);
#  * all clips per step are held-out-sentence read speech (--scripted-frac 1);
#  * groups of 8.
# The objective is the best exp20 arm (set OBJ); same sentence cross-fit evaluation.
set -euo pipefail
until grep -q "EXP20B DONE" /workspace/simbig/logs/exp20b.log; do sleep 120; done
cd /workspace/simbig/envs/omni
export HF_HOME=/workspace/simbig/cache/hf PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=/workspace/simbig/repo/scripts; A=/workspace/simbig/artifacts
V2=$A/runs/llm7b_lora_v2/best_trainable.pt
OBJ=${OBJ:-drgrpo}
uv run python $R/rl_difficulty.py --init $V2 --sentence-half 0 --out rl_difficulty_h0
uv run python $R/train_rl.py --run rl21_${OBJ}_conc --init $V2 --sentence-half 0 --mode org \
    --steps 400 --clips 4 --group 8 --eval-every 100 --objective $OBJ --scripted-frac 1.0 \
    --difficulty $A/runs/rl_difficulty_h0.parquet
echo "EXP21 DONE"
