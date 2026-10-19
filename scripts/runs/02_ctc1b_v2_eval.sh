#!/usr/bin/env bash
# Experiment 02: evaluate ctc1b_v2 (best + WiSE-FT interpolations with zero-shot weights).
# Runs on the GPU box inside envs/omni. Outputs: artifacts/posteriors/<name>, artifacts/preds/.
set -euo pipefail
cd /workspace/simbig/envs/omni
export HF_HOME=/workspace/simbig/cache/hf PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=/workspace/simbig/repo/scripts
RUN=/workspace/simbig/artifacts/runs/ctc1b_v2

# 1) fine-tuned model on every evaluation role (+ test) -> posteriors for LM decoding
uv run python $R/infer_ctc.py --name ctc1b_v2 --checkpoint $RUN/best.pt

# 2) WiSE-FT sweep (robustness to distribution shift); dev + OOD only
for a in 0.5 0.7 0.85; do
  uv run python $R/merge_weights.py --out ctc1b_v2_wise$a.pt --ckpt $RUN/best.pt:$a \
      --zeroshot-weight "$(python3 -c "print(round(1-$a, 2))")"
  uv run python $R/infer_ctc.py --name ctc1b_v2_wise$a \
      --checkpoint /workspace/simbig/artifacts/merged/ctc1b_v2_wise$a.pt \
      --roles dev_scripted,dev_spont,ood
done

# 3) best/last average (cheap soup)
uv run python $R/merge_weights.py --out ctc1b_v2_bestlast.pt --ckpt $RUN/best.pt:1 --ckpt $RUN/last.pt:1
uv run python $R/infer_ctc.py --name ctc1b_v2_bestlast \
    --checkpoint /workspace/simbig/artifacts/merged/ctc1b_v2_bestlast.pt --roles dev_scripted,dev_spont,ood
echo "ALL DONE"
