#!/usr/bin/env bash
# Runs after scripts/train_push_dyn.sh: DAgger-style predictor retraining on the demo model (push_fs4_s0) and on
# the 512x3 predictor model, 3 rounds of 2000 planner episodes each.
set -u
cd "$(dirname "$0")/.."
until grep -q DYN_DONE runs/train_push_dyn.log; do sleep 30; done
.venv/bin/python scripts/push_dagger.py --run runs/push_fs4_s0 --out runs/push_dagger_s0 --rounds 3 || echo "FAILED dagger fs4"
nvidia-smi --query-gpu=temperature.gpu,clocks.sm --format=csv,noheader | sed 's/^/gpu /'
.venv/bin/python scripts/push_dagger.py --run runs/push_p512x3_s0 --out runs/push_dagger_p512x3_s0 --rounds 3 || echo "FAILED dagger p512x3"
echo DAGGER_DONE
