#!/usr/bin/env bash
# Experiment 17 (server A, after exp19): RL-1 of reports/rl_experiment_plan.md, expert iteration
# (ExIt / ReST-EM, LITERATURE §g.11), provided data only. The hybrid CTC + multi-LLM rescorer — our
# strongest "policy", ~7% rel. better than the ROVER consensus that made pseudo_v1 — re-labels the
# unlabelled spontaneous pool; a fresh LoRA (v3 = the LoRA v2 recipe, only the label source
# changes) is trained on those labels and joins the hybrid as a 4th LLM scorer. Dev protocol.
set -euo pipefail
cd /workspace/simbig/envs/omni
export HF_HOME=/workspace/simbig/cache/hf PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=/workspace/simbig/repo/scripts; A=/workspace/simbig/artifacts; P=$A/pseudo
ADAPTERS="--adapter zs --adapter v1=$A/runs/llm7b_lora_v1/best_trainable.pt
    --adapter v2=$A/runs/llm7b_lora_v2/best_trainable.pt"

# 1) candidates for the pool: CTC v3 word-LM N-best + beam-4 N-best of zs and v2, scored by all 3
uv run python $R/infer_ctc.py --name pool_ctc1b_v3 --pool \
    --checkpoint $A/merged/ctc1b_v3_soup_wise0.85.pt
uv run python $R/ctc_nbest.py --post pool_ctc1b_v3 --name ctc_pool --pool --keep 8 --workers 8 \
    --lm dev_v2 --lm-text indomain
uv run python $R/llm_candidates.py --ctc ctc_pool --name pool_unl --beam 4 --nbest zs,v2 $ADAPTERS
# 2) hybrid labels with the dev-tuned spontaneous weights, then confidence filtering
uv run python $R/joint_rescore2.py --cands pool_unl --post pool_ctc1b_v3 --lm dev_v2 \
    --lm-text indomain --weights hybrid_dev_in.weights.json --apply-pool --name hyb_pool --workers 8
uv run python $R/pseudo_label.py --name hyb_pool --hybrid hyb_pool --compare pseudo_v1
echo "LABELS DONE"

# 3) LoRA v3 from the base model: LoRA v2 recipe with the new labels
uv run python $R/train_llm.py --run llm7b_lora_v3 --card omniASR_LLM_7B_v2 --encoder frozen \
    --decoder lora --rank 16 --sources scripted:1,spont_gold:3,pseudo_old:3,pseudo_rest:1 \
    --pseudo pseudo_old=$P/hyb_pool_old.parquet,pseudo_rest=$P/hyb_pool_rest.parquet \
    --steps 1500 --warmup 100 --lr-lora 3e-4 --max-batch-seconds 120 --grad-accum 2 \
    --eval-every 300 --eval-clips 150 --workers 8
V3=$A/runs/llm7b_lora_v3/best_trainable.pt
uv run python $R/infer_llm.py --name llm7b_lora_v3 --card omniASR_LLM_7B_v2 --decoder-lora 16 \
    --weights $V3
# 4) v3 joins the dev hybrid (existing candidates and scores are reused)
uv run python $R/llm_candidates.py --ctc pool_dev_in --name pool_dev_v3 --nbest v3 $ADAPTERS \
    --adapter v3=$V3
uv run python $R/joint_rescore2.py --cands pool_dev_v3 --post ctc1b_v3_soup_wise0.85 \
    --lm dev_v2 --lm-text indomain --name hybrid_dev_v3 --workers 8
echo "EXPERT ITERATION DONE"
