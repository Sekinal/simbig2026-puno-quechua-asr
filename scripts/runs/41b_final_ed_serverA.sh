#!/usr/bin/env bash
# Experiment 41b (server A, after exp39): final-mode submission with the encoder+decoder LoRA as a
# 4th scorer, mirroring the dev-mode "score-only" variant (exp40a/41a): ed decodes 6 perturbed views
# (they count in the view-agreement feature) and scores every candidate; candidates proposed only by
# ed are dropped. Restricted to the clips the submission and the clean checks need (test, dev_spont,
# OOD); weights from hybrid_dev_ed (scripted: tuned with lp_ed; spontaneous: lp_ed weight 0).
set -euo pipefail
until grep -q "EXP39 DONE" /workspace/simbig/logs/exp39.log; do sleep 120; done
cd /workspace/simbig/envs/omni
export HF_HOME=/workspace/simbig/cache/hf PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=/workspace/simbig/repo/scripts; A=/workspace/simbig/artifacts
uv run python - <<'PY'
import polars as pl
from qxp_asr.paths import ARTIFACTS, MANIFESTS
man = pl.read_parquet(MANIFESTS / "all.parquet")
keep = man.filter(pl.col("role").is_in(["test", "dev_spont", "ood"]))["clip_id"]
pool = pl.read_parquet(ARTIFACTS / "cands" / "pool_final_aug.parquet")
sub = pool.filter(pl.col("clip_id").is_in(keep.implode()))
sub.write_parquet(ARTIFACTS / "cands" / "pool_final_aug_sub.parquet")
print(sub["clip_id"].n_unique(), "clips,", sub.height, "candidates")
PY
uv run python $R/llm_candidates.py --ctc pool_final_aug_sub --name pool_final_ed --nbest none \
    --views 6 --view-adapters ed --encoder-rank 16 --adapter zs \
    --adapter v1=$A/runs/llm7b_final_v1/last_trainable.pt \
    --adapter v2=$A/runs/llm7b_final_v2/last_trainable.pt \
    --adapter ed=$A/runs/llm7b_final_encdec/last_trainable.pt
uv run python - <<'PY'
import polars as pl
from qxp_asr.paths import ARTIFACTS
C = ARTIFACTS / "cands"
p = pl.read_parquet(C / "pool_final_ed.parquet")
non_ed = pl.col("sources").list.eval(~pl.element().str.starts_with("ed")).list.any()
p.filter(non_ed).write_parquet(C / "pool_final_ed_scoreonly.parquet")
print(p.height, "candidates;", p.filter(non_ed).height, "kept (not only from ed)")
PY
uv run python $R/joint_rescore2.py --cands pool_final_ed_scoreonly --post ctc1b_final --lm final_v1 \
    --lm-text indomain --view-feature --weights hybrid_dev_ed.weights.json \
    --name hybrid_final_ed --workers 6
cd /workspace/simbig/repo
UV_LINK_MODE=copy uv run python $R/make_submission.py --pred hybrid_final_ed \
    --pred llm7b_final_v2 --pred ctc1b_final_greedy --name qxp_final_ed
echo "test lines changed vs qxp_final_indomain_aug:"
diff $A/submission/qxp_final_ed.tsv $A/submission/qxp_final_indomain_aug.tsv | grep -c "^<" || true
/workspace/simbig/envs/omni/.venv/bin/python $R/compare_runs.py --a hybrid_final_ed \
    --b hybrid_final_aug --sets dev_spont,ood
/workspace/simbig/envs/omni/.venv/bin/python $R/compare_runs.py --a llm7b_final_encdec \
    --b llm7b_final_v2 --sets dev_spont,ood
echo "EXP41B DONE"
