#!/usr/bin/env bash
# Experiment 10: FINAL systems. Recipes are chosen automatically on dev (scripts/select_best.py),
# then retrained in "final" mode (all labelled data except the dev roles, which stay held out for
# spontaneous tuning/monitoring), decoded, jointly rescored and written as the submission.
set -euo pipefail
until grep -q "ALL DONE" /workspace/simbig/logs/exp07.log; do sleep 60; done
OMNI=/workspace/simbig/envs/omni; REPO=/workspace/simbig/repo; R=$REPO/scripts
A=/workspace/simbig/artifacts; P=$A/pseudo
export HF_HOME=/workspace/simbig/cache/hf PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd $REPO
CTC_BEST=$(UV_LINK_MODE=copy uv run python $R/select_best.py ctc1b_v2_wise0.85_greedy \
    ctc1b_v3_soup_greedy ctc1b_v3_soup_wise0.85_greedy 2>/dev/null | tail -1)
echo "selected CTC recipe: $CTC_BEST (both LLM recipes are trained: they are complementary, joint_v3)"
PSEUDO="--pseudo pseudo_old=$P/pseudo_v1_old.parquet,pseudo_rest=$P/pseudo_v1_rest.parquet"
case "$CTC_BEST" in
  ctc1b_v3*) CTC_SRC="scripted:1,spont_gold:3,pseudo_old:3,pseudo_rest:1"; CTC_STEPS=12000; CTC_PS="$PSEUDO";;
  *)         CTC_SRC="scripted:1,spont_gold:3,spont_silver:1"; CTC_STEPS=6000; CTC_PS="";;
esac
case "$CTC_BEST" in *wise*) WISE=1;; *) WISE=0;; esac

# final language models (all non-dev text)
UV_LINK_MODE=copy uv run python $R/build_lm.py --name final_v1 --mode final
L=$A/lm/final_v1; K=/workspace/simbig/envs/kenlm/build/bin
$K/lmplz -o 10 --discount_fallback -S 20% < $L/all.chars.txt > $L/all.char10.arpa 2>/dev/null
$K/build_binary trie $L/all.char10.arpa $L/all.char10.bin >/dev/null && rm $L/all.char10.arpa

cd $OMNI
LLM_COMMON="--mode final --card omniASR_LLM_7B_v2 --encoder frozen --decoder lora --rank 16 \
    --warmup 100 --lr-lora 3e-4 --max-batch-seconds 120 --grad-accum 2 --eval-every 500 \
    --eval-clips 150 --workers 8"
# recipe v1 (silver labels) and v2 (pseudo labels, 50+ up-weighted): complementary (joint_v3)
uv run python $R/train_llm.py --run llm7b_final_v1 $LLM_COMMON --steps 1200 \
    --sources scripted:1,spont_gold:3,spont_silver:1
uv run python $R/train_llm.py --run llm7b_final_v2 $LLM_COMMON --steps 1500 \
    --sources scripted:1,spont_gold:3,pseudo_old:3,pseudo_rest:1 $PSEUDO
for v in v1 v2; do
  uv run python $R/infer_llm.py --name llm7b_final_$v --card omniASR_LLM_7B_v2 --decoder-lora 16 \
      --weights $A/runs/llm7b_final_$v/last_trainable.pt --roles dev_scripted,dev_spont,ood,test
done

uv run python $R/train_ctc.py --run ctc1b_final --mode final --steps $CTC_STEPS \
    --eval-every 1000 --workers 8 --sources $CTC_SRC $CTC_PS
RUN=$A/runs/ctc1b_final
SOUP=$(ls $RUN/step*.pt | sort -V | tail -4 | sed 's/$/:1/' | sed 's/^/--ckpt /' | tr '\n' ' ')
uv run python $R/merge_weights.py --out ctc1b_final_soup.pt $SOUP
CKPT=$A/merged/ctc1b_final_soup.pt
if [ "$WISE" = 1 ]; then
  uv run python $R/merge_weights.py --out ctc1b_final_wise.pt --ckpt $CKPT:0.85 --zeroshot-weight 0.15
  CKPT=$A/merged/ctc1b_final_wise.pt
fi
uv run python $R/infer_ctc.py --name ctc1b_final --checkpoint $CKPT --roles dev_scripted,dev_spont,ood,test

# joint CTC/LLM rescoring; scripted weights fixed from dev-mode tuning (dev sentences now seen)
uv run python $R/joint_rescore.py --post ctc1b_final --llm llm7b_final_v1 --llm llm7b_final_v2 \
    --llm zs_llm7b_all \
    --lm final_v1 --fixed-scripted 1.0,0.5,10 --name joint_final
cd $REPO
UV_LINK_MODE=copy uv run python $R/make_submission.py --pred joint_final --pred joint_v3 \
    --pred llm7b_final_v2 --name qxp_final
echo "ALL DONE"
