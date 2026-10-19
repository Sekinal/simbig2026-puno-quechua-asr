#!/usr/bin/env bash
# Experiment 16 (server B): CTC fine-tuning from the LoRA-CPT encoder (memory-fixed init), longer
# schedule because it starts from an SSL encoder (Meta's from-encoder recipe uses 30k steps).
set -euo pipefail
cd /workspace/simbig/envs/omni
export HF_HOME=/workspace/simbig/cache/hf PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=/workspace/simbig/repo/scripts; A=/workspace/simbig/artifacts; P=$A/pseudo
uv run python $R/train_ctc.py --run ctc1b_cpt --steps 20000 --eval-every 1000 --workers 8 \
    --sources scripted:1,spont_gold:3,pseudo_old:3,pseudo_rest:1 \
    --pseudo pseudo_old=$P/pseudo_v1_old.parquet,pseudo_rest=$P/pseudo_v1_rest.parquet \
    --init-encoder $A/cpt/cpt_w2v1b_lora64/encoder.pt --lr 5e-5
RUN=$A/runs/ctc1b_cpt
SOUP=$(ls $RUN/step*.pt | sort -V | tail -4 | sed 's/$/:1/' | sed 's/^/--ckpt /' | tr '\n' ' ')
uv run python $R/merge_weights.py --out ctc1b_cpt_soup.pt $SOUP
uv run python $R/infer_ctc.py --name ctc1b_cpt_soup --checkpoint $A/merged/ctc1b_cpt_soup.pt
echo "ALL DONE"
