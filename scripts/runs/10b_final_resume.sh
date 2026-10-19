#!/usr/bin/env bash
# Resume of experiment 10 after the joint-rescoring step failed: posteriors for the itest roles
# (in final-mode training, but iterated by joint_rescore) were missing. Same checkpoint/settings.
set -euo pipefail
cd /workspace/simbig/envs/omni
export HF_HOME=/workspace/simbig/cache/hf PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=/workspace/simbig/repo/scripts; A=/workspace/simbig/artifacts
uv run python $R/infer_ctc.py --name ctc1b_final --checkpoint $A/merged/ctc1b_final_wise.pt \
    --roles itest_scripted,itest_spont
uv run python $R/joint_rescore.py --post ctc1b_final --llm llm7b_final_v1 --llm llm7b_final_v2 \
    --llm zs_llm7b_all --lm final_v1 --fixed-scripted 1.0,0.5,10 --name joint_final
cd /workspace/simbig/repo
UV_LINK_MODE=copy uv run python $R/make_submission.py --pred joint_final --pred joint_v3 \
    --pred llm7b_final_v2 --name qxp_final
echo "ALL DONE" >> /workspace/simbig/logs/exp10.log
