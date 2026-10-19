#!/usr/bin/env bash
# Experiment 22 (server B, after exp20): "Spectrum" stage of the Spectrum -> Verifier -> Signal
# program (LITERATURE §g.12; VibeThinker's spectrum-to-signal principle, arXiv:2511.06221).
# A learned verifier needs candidate lists with REALISTIC errors on labelled audio; decoding the
# training set with our models gives memorised, error-free lists. Two cross-fit LoRA folds: fold-k
# trains without fold-k speakers and without fold-k sentences (qxp_asr.splits), then decodes the
# fold-k gold clips (voice and text both unseen) with beam 8, together with the zero-shot model.
# Then the deferred RL arms (exp20b Dr. GRPO, exp21 concentrated signal).
set -euo pipefail
until grep -q "EXP20 DONE" /workspace/simbig/logs/exp20.log; do sleep 120; done
cd /workspace/simbig/envs/omni
export HF_HOME=/workspace/simbig/cache/hf PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=/workspace/simbig/repo/scripts; A=/workspace/simbig/artifacts; P=$A/pseudo
for k in 0 1; do
  uv run python $R/train_llm.py --run llm7b_fold$k --card omniASR_LLM_7B_v2 --encoder frozen \
      --decoder lora --rank 16 --sources scripted:1,spont_gold:3,pseudo_old:3,pseudo_rest:1 \
      --pseudo pseudo_old=$P/pseudo_v1_old.parquet,pseudo_rest=$P/pseudo_v1_rest.parquet \
      --steps 1500 --warmup 100 --lr-lora 3e-4 --max-batch-seconds 120 --grad-accum 2 \
      --eval-every 500 --eval-clips 150 --workers 8 --exclude-fold $k
  uv run python $R/llm_candidates.py --crossfit-fold $k --max-clips 3000 --name crossfit$k \
      --beam 8 --no-score --adapter zs --adapter f$k=$A/runs/llm7b_fold$k/best_trainable.pt
done
echo "EXP22 DONE"
/workspace/simbig/repo/scripts/runs/20b_rl_drgrpo_serverB.sh > /workspace/simbig/logs/exp20b.log 2>&1
OBJ=drgrpo /workspace/simbig/repo/scripts/runs/21_rl_concentrated_serverB.sh \
    > /workspace/simbig/logs/exp21.log 2>&1
