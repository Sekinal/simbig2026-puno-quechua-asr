#!/usr/bin/env bash
# Experiment 19 (server A): provided-data-only systems. From 2026-10-10 on we use only the
# shared-task data (Common Voice Puno Quechua scripted + spontaneous, incl. the organizers' silver
# labels) plus pretrained models; no external text. Everything that touched external text is
# rebuilt: CTC N-best with the in-domain word 4-gram, the hybrid with the in-domain char 10-gram,
# no text-CPT scorer. LLM beam candidates and their scores (zs, v1, v2: in-domain) are reused.
# Then expert iteration (exp17, in-domain) and the RL pilots (exp11).
set -euo pipefail
OMNI=/workspace/simbig/envs/omni; R=/workspace/simbig/repo/scripts; A=/workspace/simbig/artifacts
export HF_HOME=/workspace/simbig/cache/hf PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
K=/workspace/simbig/envs/kenlm/build/bin
for L in $A/lm/dev_v2 $A/lm/final_v1; do  # in-domain char 10-gram (dev_v2 already has one)
  [ -f $L/indomain.char10.bin ] && continue
  $K/lmplz -o 10 --discount_fallback -S 20% < $L/indomain.chars.txt > $L/indomain.char10.arpa 2>/dev/null
  $K/build_binary trie $L/indomain.char10.arpa $L/indomain.char10.bin >/dev/null
  rm $L/indomain.char10.arpa
done
cd $OMNI
# dev-mode systems: tune the hybrid weights
uv run python $R/ctc_nbest.py --post ctc1b_v3_soup_wise0.85 --lm dev_v2 --lm-text indomain \
    --name ctc_v3_in --workers 8
uv run python $R/llm_candidates.py --ctc ctc_v3_in --base pool_dev --name pool_dev_in --nbest none \
    --adapter zs --adapter v1=$A/runs/llm7b_lora_v1/best_trainable.pt \
    --adapter v2=$A/runs/llm7b_lora_v2/best_trainable.pt
uv run python $R/joint_rescore2.py --cands pool_dev_in --post ctc1b_v3_soup_wise0.85 \
    --lm dev_v2 --lm-text indomain --name hybrid_dev_in --workers 8
# final-mode systems: transfer scripted weights, re-tune spontaneous on dev_spont
uv run python $R/ctc_nbest.py --post ctc1b_final --lm final_v1 --lm-text indomain \
    --name ctc_final_in --workers 8
uv run python $R/llm_candidates.py --ctc ctc_final_in --base pool_final --name pool_final_in \
    --nbest none --adapter zs --adapter v1=$A/runs/llm7b_final_v1/last_trainable.pt \
    --adapter v2=$A/runs/llm7b_final_v2/last_trainable.pt
uv run python $R/joint_rescore2.py --cands pool_final_in --post ctc1b_final --lm final_v1 \
    --lm-text indomain --weights hybrid_dev_in.weights.json --retune-spont --name hybrid_final_in \
    --workers 8
cd /workspace/simbig/repo
UV_LINK_MODE=copy uv run python $R/make_submission.py --pred hybrid_final_in \
    --pred llm7b_final_v2 --pred ctc1b_final_greedy --name qxp_final_indomain
echo "INDOMAIN DONE"
/workspace/simbig/repo/scripts/runs/11_post_final.sh
