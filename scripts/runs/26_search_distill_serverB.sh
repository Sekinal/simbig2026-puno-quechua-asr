#!/usr/bin/env bash
# Experiment 26 (server B): search-then-distill on the unlabelled spontaneous pool (expert iteration
# with a search operator that finds NEW transcripts). exp17 failed because the hybrid only re-ranked
# candidates the memorising members already produced (labels 94 % unchanged). Now the search adds
# greedy decodes of perturbed audio views (exp25: better candidates surface, clean-audio hybrid
# verifies them). Steps: views -> hybrid labels (dev-tuned weights, view feature) -> filtering and
# label-change report vs pseudo_v1 -> LoRA v4 (v2 recipe, new labels) -> single-model evaluation.
set -euo pipefail
cd /workspace/simbig/envs/omni
export HF_HOME=/workspace/simbig/cache/hf PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=/workspace/simbig/repo/scripts; A=/workspace/simbig/artifacts; P=$A/pseudo
uv run python $R/llm_candidates.py --ctc pool_unl --name pool_unl_aug --nbest none \
    --views 3 --view-adapters zs,v2 --adapter zs \
    --adapter v1=$A/runs/llm7b_lora_v1/best_trainable.pt \
    --adapter v2=$A/runs/llm7b_lora_v2/best_trainable.pt
uv run python $R/joint_rescore2.py --cands pool_unl_aug --post pool_ctc1b_v3 --lm dev_v2 \
    --lm-text indomain --view-feature --weights hybrid_dev_aug.weights.json --apply-pool \
    --name hyb_pool_aug --workers 8
uv run python $R/pseudo_label.py --name hyb_pool_aug --hybrid hyb_pool_aug --compare pseudo_v1
echo "LABELS DONE"
uv run python $R/train_llm.py --run llm7b_lora_v4 --card omniASR_LLM_7B_v2 --encoder frozen \
    --decoder lora --rank 16 --sources scripted:1,spont_gold:3,pseudo_old:3,pseudo_rest:1 \
    --pseudo pseudo_old=$P/hyb_pool_aug_old.parquet,pseudo_rest=$P/hyb_pool_aug_rest.parquet \
    --steps 1500 --warmup 100 --lr-lora 3e-4 --max-batch-seconds 120 --grad-accum 2 \
    --eval-every 300 --eval-clips 150 --workers 8
uv run python $R/infer_llm.py --name llm7b_lora_v4 --card omniASR_LLM_7B_v2 --decoder-lora 16 \
    --weights $A/runs/llm7b_lora_v4/best_trainable.pt
echo "EXP26 DONE"
