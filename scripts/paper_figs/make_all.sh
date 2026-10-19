#!/usr/bin/env bash
# Regenerate every figure used in docs/paper/main.tex into docs/paper/figs/ (reads only reports/).
#   bash scripts/paper_figs/make_all.sh
# fig_sentence_reuse.py and fig_verifier.py are kept for an extended version; they are not in the
# 8-page paper and are not run here.
set -euo pipefail
cd "$(dirname "$0")/../.."
for name in where_to_adapt plasticity layers_rank; do
    echo "== fig_${name}.py"
    uv run python "scripts/paper_figs/fig_${name}.py"
done
