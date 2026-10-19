#!/usr/bin/env bash
# Experiment 27a, resumed (server A): the fold-0 pool labelling of exp27a ran out of GPU memory in its
# view stage because exp30 was started on the same GPU. The dev-side outputs (pool_dev_fold, the
# unbiased hybrid weights) were saved; this reruns exp27a from the fold-0 pool step unchanged.
set -euo pipefail
cd /workspace/simbig/envs/omni
export HF_HOME=/workspace/simbig/cache/hf PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=/workspace/simbig/repo/scripts; A=/workspace/simbig/artifacts; B="root@<SERVER_B_HOST>"
uv run python $R/llm_candidates.py --ctc ctc_pool_f0 --name pool_unl_f0 --beam 4 --nbest zs,fold \
    --views 3 --view-adapters zs,fold --adapter zs --adapter fold=$A/runs/llm7b_fold0/best_trainable.pt
uv run python $R/joint_rescore2.py --cands pool_unl_f0 --post pool_ctc1b_v3 --scorers zs,fold \
    --ctc-weight 0 --view-feature --lm dev_v2 --lm-text indomain \
    --weights hybrid_dev_fold.weights.json --apply-pool --name hyb_pool_f0 --workers 8
# 3) merge with fold 1 (server B), filter, label-change report, ship to B for training
until ssh -p <SERVER_B_PORT> $B "test -f $A/preds/hyb_pool_f1.parquet.done"; do sleep 120; done
scp -q -P <SERVER_B_PORT> $B:$A/preds/hyb_pool_f1.parquet $A/preds/
uv run python -c "
import polars as pl
P = '$A/preds/'
pl.concat([pl.read_parquet(P + f'hyb_pool_f{k}.parquet') for k in (0, 1)]).write_parquet(P + 'hyb_pool_xfit.parquet')"
uv run python $R/pseudo_label.py --name hyb_pool_xfit --hybrid hyb_pool_xfit --compare pseudo_v1
for f in hyb_pool_xfit hyb_pool_xfit_old hyb_pool_xfit_rest; do
  scp -q -P <SERVER_B_PORT> $A/pseudo/$f.parquet $B:$A/pseudo/
done
ssh -p <SERVER_B_PORT> $B "touch $A/pseudo/hyb_pool_xfit.done"
echo "EXP27A DONE"
