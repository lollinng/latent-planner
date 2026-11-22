#!/usr/bin/env bash
# After the 3 push seeds, train one push variant locally (lambda 0.01, 15k steps) and evaluate it on the tuning set.
set -u
cd "$(dirname "$0")/.."
while ! grep -q PUSH_DONE runs/train_push.log 2>/dev/null; do sleep 15; done
name=pushv_l0.01_fs4_15k_s0
mkdir -p runs/$name
echo "$(date +%T) start $name"
.venv/bin/python -m latent_planner.train --lam 0.01 --seed 0 --frameskip 4 --steps 15000 --train data/push_train \
  --val data/push_val --out runs/$name > runs/$name/stdout.txt 2>&1 || echo "FAILED $name"
echo "$(date +%T) done $name"
.venv/bin/python -m latent_planner.eval_push --runs runs/$name --taskset tune --tag push_dev 2>&1 | grep -v -i warn
echo VARIANT_DONE
