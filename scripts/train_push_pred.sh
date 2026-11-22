#!/usr/bin/env bash
# Push: does a bigger predictor fix the contact dynamics? (True simulator + latent cost reaches 86% on the tuning
# tasks, the learned predictor 32%, so the predictor is the weak part.) Seed 0, same data, tuning tasks.
set -u
cd "$(dirname "$0")/.."
for c in "512 2" "512 3" "1024 2" "1024 3"; do
  set -- $c
  name=push_p$1x$2_s0
  if [ ! -f "runs/$name/summary.json" ]; then
    mkdir -p "runs/$name"
    echo "$(date +%T) start $name"
    .venv/bin/python -m latent_planner.train --lam 0.09 --seed 0 --frameskip 4 --pred-hidden $1 --pred-layers $2 \
      --train data/push_train --val data/push_val --out "runs/$name" > "runs/$name/stdout.txt" 2>&1 || echo "FAILED $name"
  fi
  .venv/bin/python -m latent_planner.eval_push --runs "runs/$name" --taskset tune --tag push_pred || echo "FAILED eval $name"
  nvidia-smi --query-gpu=temperature.gpu,clocks.sm --format=csv,noheader | sed 's/^/gpu /'
done
echo PRED_DONE
