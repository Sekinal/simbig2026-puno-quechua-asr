#!/usr/bin/env bash
# Experiment 36 (server B, after exp35): a stronger CTC scorer. The CTC log-likelihood is the hybrid's
# most important read-speech feature (ablation: +1.1 dev / +1.7 itest unseen without it), and
# zero-shot omniASR-CTC-3B is far better than the 1B on spontaneous speech (23.5 vs 31.5 WER).
# CTC v3 recipe on the 3B model; the lowest 75% of the 60 encoder layers are frozen so the fp32 AdamW
# state and activations fit on one L40S (half froze: OOM); 50 s batches x 4 accumulation;
# 5k steps (0.094 min/step: 10k would not leave time for a final-mode run before the deadline).
set -euo pipefail
until grep -q "EXP35 DONE" /workspace/simbig/logs/exp35.log; do sleep 120; done
cd /workspace/simbig/envs/omni
export HF_HOME=/workspace/simbig/cache/hf PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=/workspace/simbig/repo/scripts; A=/workspace/simbig/artifacts; P=$A/pseudo
uv run python $R/train_ctc.py --run ctc3b_v3 --card omniASR_CTC_3B_v2 --freeze-frac 0.75 \
    --steps 5000 --eval-every 1000 --workers 8 --max-batch-seconds 50 --grad-accum 4 \
    --sources scripted:1,spont_gold:3,pseudo_old:3,pseudo_rest:1 \
    --pseudo pseudo_old=$P/pseudo_v1_old.parquet,pseudo_rest=$P/pseudo_v1_rest.parquet
RUN=$A/runs/ctc3b_v3
SOUP=$(ls $RUN/step*.pt | sort -V | tail -4 | sed 's/$/:1/' | sed 's/^/--ckpt /' | tr '\n' ' ')
uv run python $R/merge_weights.py --card omniASR_CTC_3B_v2 --out ctc3b_v3_soup.pt $SOUP
uv run python $R/merge_weights.py --card omniASR_CTC_3B_v2 --out ctc3b_v3_soup_wise0.85.pt \
    --ckpt $A/merged/ctc3b_v3_soup.pt:0.85 --zeroshot-weight 0.15
uv run python $R/infer_ctc.py --name ctc3b_v3_soup_wise0.85 --card omniASR_CTC_3B_v2 \
    --checkpoint $A/merged/ctc3b_v3_soup_wise0.85.pt
echo "EXP36 DONE"
