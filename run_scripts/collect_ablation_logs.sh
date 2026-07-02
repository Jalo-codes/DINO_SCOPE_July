#!/usr/bin/env bash
# Collect ablation sweep LOGS into the repo for committing — run ON THE BOX.
#
# Copies ONLY the small text artifacts (orchestrator.log, eval_summary.json,
# rollup *.csv) out of each sweep run_root into results/ablation/<basename>,
# preserving the per-cell directory structure. Checkpoints, per-item dumps, and
# images are filtered out by the rsync include/exclude rules, so the committed
# tree stays tiny and the .gitignore exception (results/ablation/**/*.{log,json,csv})
# tracks exactly what lands here.
#
# Usage (on the box, repo cwd):
#   bash run_scripts/collect_ablation_logs.sh                       # default roots
#   bash run_scripts/collect_ablation_logs.sh /media/ssd/runs/foo … # explicit roots
#
# Then commit + push (also on the box):
#   git add results/ablation && git commit -m "ablation: sweep logs + rollups" && git push
set -euo pipefail
cd "$(dirname "$0")/.."                          # repo root
DST="results/ablation"

# Default source run_roots (override by passing paths as args). Add/trim freely;
# missing roots are skipped with a warning rather than aborting.
if [ "$#" -gt 0 ]; then
  SRCS=("$@")
else
  SRCS=(
    /media/ssd/runs/ablation_eval        # all eval run_roots (rank, res, e4, nozoom, deep)
    /media/ssd/runs/ablation             # training run_roots (rank/res sweep), if present
  )
fi

mkdir -p "$DST"
for src in "${SRCS[@]}"; do
  if [ ! -d "$src" ]; then
    echo "[collect] skip (not found): $src" >&2
    continue
  fi
  base="$(basename "$src")"
  echo "[collect] $src  ->  $DST/$base"
  rsync -av --prune-empty-dirs \
    --include='*/' \
    --include='orchestrator.log' \
    --include='eval_summary.json' \
    --include='*.csv' \
    --exclude='*' \
    "$src/" "$DST/$base/"
done

echo
echo "[collect] done. Review what will be committed:"
echo "  git add -n results/ablation && git status --short results/ablation"
echo "[collect] then:"
echo "  git add results/ablation && git commit -m 'ablation: sweep logs + rollups' && git push"
