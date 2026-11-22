#!/usr/bin/env bash
# Scene 2 (Push): frameskip-4 models, lambda 0.09, 3 seeds, trained one after another. Skips finished runs.
set -u
cd "$(dirname "$0")/.."
for s in 0 1 2; do
  name=push_fs4_s$s
  if [ -f "runs/$name/summary.json" ]; then echo "skip $name"; continue; fi
  mkdir -p "runs/$name"
  echo "$(date +%T) start $name"
  .venv/bin/python -m latent_planner.train --lam 0.09 --seed $s --frameskip 4 --train data/push_train \
    --val data/push_val --out "runs/$name" > "runs/$name/stdout.txt" 2>&1 || echo "FAILED $name"
  echo "$(date +%T) done $name"
done
echo PUSH_DONE
