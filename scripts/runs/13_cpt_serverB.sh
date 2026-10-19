#!/usr/bin/env bash
# Experiment 13 (server B): continued pretraining, both branches, dev protocol (comparable to v2/v3).
#  a) decoder text-CPT: LoRA v2 recipe + text-only LM loss on dev-mode LM text (dev_v2/all.txt,
#     in-domain transcripts + 1.3M-token respelled external corpus; no dev/itest sentences)
#  b) encoder audio CPT: LoRA (r=64) wav2vec2 SSL on all training-speaker audio, then CTC fine-tuning
#     from the CPT encoder with the v3 recipe; compared with ctc1b_v3.
set -euo pipefail
cd /workspace/simbig/envs/omni
export HF_HOME=/workspace/simbig/cache/hf PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=/workspace/simbig/repo/scripts; A=/workspace/simbig/artifacts; P=$A/pseudo
PSEUDO="--pseudo pseudo_old=$P/pseudo_v1_old.parquet,pseudo_rest=$P/pseudo_v1_rest.parquet"

# a) decoder text-CPT (same data/steps as LoRA v2 + text LM loss)
uv run python $R/train_llm.py --run llm7b_textcpt --card omniASR_LLM_7B_v2 --encoder frozen \
    --decoder lora --rank 16 --sources scripted:1,spont_gold:3,pseudo_old:3,pseudo_rest:1 $PSEUDO \
    --steps 1500 --warmup 100 --lr-lora 3e-4 --max-batch-seconds 120 --grad-accum 2 \
    --eval-every 300 --eval-clips 150 --workers 8 \
    --text-corpus $A/lm/dev_v2/all.txt --text-weight 0.3 --text-batch 16 --grad-ckpt
uv run python $R/infer_llm.py --name llm7b_textcpt --card omniASR_LLM_7B_v2 --decoder-lora 16 \
    --weights $A/runs/llm7b_textcpt/best_trainable.pt
echo "TEXTCPT DONE"

# b) encoder audio CPT (LoRA r=64) -> CTC fine-tuning with the v3 recipe
uv run python $R/cpt_w2v.py --run cpt_w2v1b_lora64 --card omniASR_W2V_1B --lora 64 --steps 8000
uv run python $R/train_ctc.py --run ctc1b_cpt --steps 12000 --eval-every 1000 --workers 8 \
    --sources scripted:1,spont_gold:3,pseudo_old:3,pseudo_rest:1 $PSEUDO \
    --init-encoder $A/cpt/cpt_w2v1b_lora64/encoder.pt
RUN=$A/runs/ctc1b_cpt
SOUP=$(ls $RUN/step*.pt | sort -V | tail -4 | sed 's/$/:1/' | sed 's/^/--ckpt /' | tr '\n' ' ')
uv run python $R/merge_weights.py --out ctc1b_cpt_soup.pt $SOUP
uv run python $R/merge_weights.py --out ctc1b_cpt_soup_wise0.85.pt \
    --ckpt $A/merged/ctc1b_cpt_soup.pt:0.85 --zeroshot-weight 0.15
for m in ctc1b_cpt_soup ctc1b_cpt_soup_wise0.85; do
  uv run python $R/infer_ctc.py --name $m --checkpoint $A/merged/$m.pt
done
echo "ALL DONE"
