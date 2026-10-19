#!/usr/bin/env bash
# Experiment 27a (server A): cross-fitted search-then-distill. exp17/exp26 showed the hybrid keeps
# the old pseudo-labels on the unlabelled pool (93 % unchanged) because its scorers (LoRA v1/v2,
# CTC v3) were trained on those very clips. Here every scorer is unbiased: the zero-shot model and a
# cross-fit fold LoRA that never saw the speakers being labelled (exp22, --exclude-fold); CTC v3
# only proposes candidates (--ctc-weight 0). Steps: tune the unbiased hybrid on dev (fold 0 as
# "fold"), label fold-0 pool clips here and fold-1 clips on server B, merge, report label change.
set -euo pipefail
cd /workspace/simbig/envs/omni
export HF_HOME=/workspace/simbig/cache/hf PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=/workspace/simbig/repo/scripts; A=/workspace/simbig/artifacts; B="root@<SERVER_B_HOST>"
for k in 0 1; do  # fold LoRAs live on server B
  mkdir -p $A/runs/llm7b_fold$k
  scp -q -P <SERVER_B_PORT> $B:$A/runs/llm7b_fold$k/best_trainable.pt $A/runs/llm7b_fold$k/
done
# per-fold CTC candidate tables of the unlabelled pool (speaker folds of qxp_asr.splits)
uv run python - <<'PY'
import polars as pl
from qxp_asr.paths import ARTIFACTS, MANIFESTS
from qxp_asr.splits import with_crossfit_folds
folds = with_crossfit_folds(pl.read_parquet(MANIFESTS / "all.parquet")).select("clip_id", "spk_fold")
ctc = pl.read_parquet(ARTIFACTS / "cands" / "ctc_pool.parquet").join(folds, on="clip_id")
for k in (0, 1):
    part = ctc.filter(pl.col("spk_fold") == k).drop("spk_fold")
    part.write_parquet(ARTIFACTS / "cands" / f"ctc_pool_f{k}.parquet")
    print(k, part["clip_id"].n_unique(), "clips")
PY
scp -q -P <SERVER_B_PORT> $A/cands/ctc_pool_f1.parquet $B:$A/cands/
# 1) unbiased hybrid on dev: fold-0 beam + views added, scored by zs and fold only
uv run python $R/llm_candidates.py --ctc pool_dev_aug --name pool_dev_fold --nbest fold \
    --views 6 --view-adapters fold --adapter zs --adapter fold=$A/runs/llm7b_fold0/best_trainable.pt
uv run python $R/joint_rescore2.py --cands pool_dev_fold --post ctc1b_v3_soup_wise0.85 \
    --scorers zs,fold --ctc-weight 0 --view-feature --lm dev_v2 --lm-text indomain \
    --name hybrid_dev_fold --workers 8
scp -q -P <SERVER_B_PORT> $A/cands/hybrid_dev_fold.weights.json $B:$A/cands/
# 2) label fold-0 pool clips with zs + fold-0 (search: beams + 3 views each)
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
