#!/usr/bin/env bash
# Experiment 27b (server B): fold-1 half of the cross-fitted search-then-distill (see 27a), then
# LoRA v5 on the merged cross-fitted labels (LoRA v2 recipe, only the label source changes).
set -euo pipefail
until grep -q "EXP26 DONE" /workspace/simbig/logs/exp26.log; do sleep 120; done
cd /workspace/simbig/envs/omni
export HF_HOME=/workspace/simbig/cache/hf PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=/workspace/simbig/repo/scripts; A=/workspace/simbig/artifacts; P=$A/pseudo
until [ -f $A/cands/ctc_pool_f1.parquet ]; do sleep 60; done
uv run python $R/llm_candidates.py --ctc ctc_pool_f1 --name pool_unl_f1 --beam 4 --nbest zs,fold \
    --views 3 --view-adapters zs,fold --adapter zs --adapter fold=$A/runs/llm7b_fold1/best_trainable.pt
until [ -f $A/cands/hybrid_dev_fold.weights.json ]; do sleep 60; done
uv run python $R/joint_rescore2.py --cands pool_unl_f1 --post pool_ctc1b_v3 --scorers zs,fold \
    --ctc-weight 0 --view-feature --lm dev_v2 --lm-text indomain \
    --weights hybrid_dev_fold.weights.json --apply-pool --name hyb_pool_f1 --workers 8
touch $A/preds/hyb_pool_f1.parquet.done
until [ -f $P/hyb_pool_xfit.done ]; do sleep 120; done
uv run python $R/train_llm.py --run llm7b_lora_v5 --card omniASR_LLM_7B_v2 --encoder frozen \
    --decoder lora --rank 16 --sources scripted:1,spont_gold:3,pseudo_old:3,pseudo_rest:1 \
    --pseudo pseudo_old=$P/hyb_pool_xfit_old.parquet,pseudo_rest=$P/hyb_pool_xfit_rest.parquet \
    --steps 1500 --warmup 100 --lr-lora 3e-4 --max-batch-seconds 120 --grad-accum 2 \
    --eval-every 300 --eval-clips 150 --workers 8
uv run python $R/infer_llm.py --name llm7b_lora_v5 --card omniASR_LLM_7B_v2 --decoder-lora 16 \
    --weights $A/runs/llm7b_lora_v5/best_trainable.pt
echo "EXP27B DONE"
