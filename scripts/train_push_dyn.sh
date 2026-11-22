#!/usr/bin/env bash
# Push dynamics levers (seed 0 unless noted, tuning tasks), after the predictor-size sweep:
#  1. 512x3 predictor over 3 seeds (is its 40% on the tuning tasks real?) + test eval
#  2. longer open-loop rollout loss (12 predictor steps = 48 env steps, the longest that fits 49-frame episodes)
#  3. frameskip 1 (finer contact steps), 16-step rollout loss
#  4. 3x data, 90% pushing, 18k steps
set -u
cd "$(dirname "$0")/.."
train() {  # name, extra args...
  local name=$1; shift
  if [ ! -f "runs/$name/summary.json" ]; then
    mkdir -p "runs/$name"; echo "$(date +%T) start $name"
    .venv/bin/python -m latent_planner.train --lam 0.09 --val data/push_val --out "runs/$name" "$@" \
      > "runs/$name/stdout.txt" 2>&1 || echo "FAILED $name"
  fi
}
ev() { .venv/bin/python -m latent_planner.eval_push --runs "$@" --taskset tune --tag push_dyn || echo "FAILED eval $*"; }
P="--pred-hidden 512 --pred-layers 3 --frameskip 4"
for s in 1 2; do train push_p512x3_s$s --seed $s $P --train data/push_train; ev runs/push_p512x3_s$s; done
.venv/bin/python -m latent_planner.eval_push --runs runs/push_p512x3_s0 runs/push_p512x3_s1 runs/push_p512x3_s2 \
  --taskset test --tag push_dyn || echo "FAILED test eval"
train push_p512x3_r12_s0 --seed 0 $P --T 12 --rollout 12 --train data/push_train; ev runs/push_p512x3_r12_s0
train push_p512x3_fs1_s0 --seed 0 --pred-hidden 512 --pred-layers 3 --frameskip 1 --T 16 --rollout 16 \
  --train data/push_train; ev runs/push_p512x3_fs1_s0
if [ ! -f data/push3_train/meta.json ]; then
  .venv/bin/python -m latent_planner.data --scene push --episodes 6000 --seed 11 --policy-frac 0.9 --out data/push3_train \
    || echo "FAILED data"
fi
train push3_p512x3_18k_s0 --seed 0 $P --steps 18000 --train data/push3_train; ev runs/push3_p512x3_18k_s0
nvidia-smi --query-gpu=temperature.gpu,clocks.sm --format=csv,noheader | sed 's/^/gpu /'
echo DYN_DONE
