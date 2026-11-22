#!/usr/bin/env bash
# Does more CEM search help the learned push model the way it helps the true simulator (oracle 57% -> 81% on the
# tuning tasks with K8xh4 S1000 I15)? Tuning tasks only; test is run once on the chosen setting.
set -u
cd "$(dirname "$0")/.."
for run in runs/push_fs4_s0 runs/pushv2_ln_s0; do
  for c in "5 4 1000 15" "8 4 1000 15" "5 8 1000 15" "8 4 2000 20"; do
    set -- $c
    .venv/bin/python -m latent_planner.eval_push --runs $run --taskset tune --tag push_search \
      --knots $1 --hold $2 --samples $3 --iters $4 || echo "FAILED $run $c"
    nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader | sed 's/^/gpu temp /'
  done
done
echo SEARCH_DONE
