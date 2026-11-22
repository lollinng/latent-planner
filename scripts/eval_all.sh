#!/usr/bin/env bash
# Evaluate every trained run on the 100 test tasks (planner settings locked from the tuning set) and run the
# probes. Waits for scripts/train_all.sh to finish first. Skips runs already in results/eval.csv for this tag.
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
TAG=${TAG:-main}
while ! grep -q ALL_DONE runs/train_all.log 2>/dev/null; do sleep 30; done
for d in runs/lam0.09_s? runs/lam0_s? runs/fs4_lam0.09_s?; do
  [ -f "$d/summary.json" ] || continue
  if grep -q ",$TAG,test,latent_cem,$d," results/eval.csv 2>/dev/null || grep -q "^$TAG,test,latent_cem,$d," results/eval.csv 2>/dev/null; then
    echo "skip eval $d"
  else
    $PY -m latent_planner.eval --runs "$d" --taskset test --tag "$TAG" --running-w 1.0 2>&1 | grep -v -i warn
  fi
  if ! grep -q "^$(basename "$d")," results/analysis.csv 2>/dev/null; then
    $PY -m latent_planner.analyze --runs "$d" 2>&1 | grep -v -i warn
  fi
done
echo EVAL_DONE
