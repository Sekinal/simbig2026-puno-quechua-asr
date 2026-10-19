#!/usr/bin/env bash
# Experiment 07: CTC-1B v3 — pseudo-labelled spontaneous pool (agreement-filtered) instead of raw
# silver, older speakers up-weighted, 2x longer schedule (v2 still improving at 6k steps),
# checkpoint soup ingredients from 60 % of training (§g.3).
set -euo pipefail
until grep -q "ALL DONE" /workspace/simbig/logs/exp06.log; do sleep 60; done
cd /workspace/simbig/envs/omni
export HF_HOME=/workspace/simbig/cache/hf PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
P=/workspace/simbig/artifacts/pseudo
R=/workspace/simbig/repo/scripts
uv run python $R/train_ctc.py --run ctc1b_v3 --steps 12000 --eval-every 1000 --workers 8 \
    --sources scripted:1,spont_gold:3,pseudo_old:3,pseudo_rest:1 \
    --pseudo pseudo_old=$P/pseudo_v1_old.parquet,pseudo_rest=$P/pseudo_v1_rest.parquet
RUN=/workspace/simbig/artifacts/runs/ctc1b_v3
SOUP=$(ls $RUN/step*.pt | sort -V | tail -4 | sed 's/$/:1/' | sed 's/^/--ckpt /' | tr '\n' ' ')
uv run python $R/merge_weights.py --out ctc1b_v3_soup.pt $SOUP
uv run python $R/merge_weights.py --out ctc1b_v3_soup_wise0.85.pt \
    --ckpt /workspace/simbig/artifacts/merged/ctc1b_v3_soup.pt:0.85 --zeroshot-weight 0.15
for m in ctc1b_v3_soup ctc1b_v3_soup_wise0.85; do
  uv run python $R/infer_ctc.py --name $m --checkpoint /workspace/simbig/artifacts/merged/$m.pt
done
echo "ALL DONE"
