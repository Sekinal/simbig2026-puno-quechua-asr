#!/usr/bin/env bash
# Experiment 41a (server A, CPU): prepare a submission with the encoder+decoder LoRA as a 4th scorer.
# (1) Reproduce the current final hybrid (hybrid_final_aug -> qxp_final_indomain_aug.tsv) from its
#     saved weights, to make sure the final pipeline below is the one that made the submission.
# (2) Tune the dev-mode weights with the ed score (score-only pool of exp40), starting from the
#     current weights (the ed weight starts at 0), and compare with hybrid_dev_aug on the same clips.
set -euo pipefail
cd /workspace/simbig/envs/omni
R=/workspace/simbig/repo/scripts; A=/workspace/simbig/artifacts
uv run python $R/joint_rescore2.py --cands pool_final_aug --post ctc1b_final --lm final_v1 \
    --lm-text indomain --view-feature --weights hybrid_dev_aug.weights.json \
    --name hybrid_final_aug_repro --workers 6
cd /workspace/simbig/repo
UV_LINK_MODE=copy uv run python $R/make_submission.py --pred hybrid_final_aug_repro \
    --pred llm7b_final_v2 --pred ctc1b_final_greedy --name repro_check
if cmp -s $A/submission/repro_check.tsv $A/submission/qxp_final_indomain_aug.tsv; then
  echo "REPRO OK: identical to qxp_final_indomain_aug.tsv"
else
  echo "REPRO DIFFERS:"; diff $A/submission/repro_check.tsv $A/submission/qxp_final_indomain_aug.tsv | wc -l
fi
rm -f $A/submission/repro_check.tsv
cd /workspace/simbig/envs/omni
uv run python $R/joint_rescore2.py --cands pool_ctrl_ed_scoreonly --post ctc1b_v3_soup_wise0.85 \
    --lm dev_v2 --lm-text indomain --view-feature --init-weights hybrid_dev_aug.weights.json \
    --name hybrid_dev_ed --workers 6
cat $A/cands/hybrid_dev_ed.weights.json
cd /workspace/simbig/repo
/workspace/simbig/envs/omni/.venv/bin/python $R/compare_runs.py --a hybrid_dev_ed --b hybrid_dev_aug
echo "EXP41A DONE"
