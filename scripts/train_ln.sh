#!/usr/bin/env bash
# Next-step experiment: LayerNorm instead of BatchNorm in the projector. Two-room frameskip 4 and push, 3 seeds each,
# trained one after another, then evaluated on the same test tasks as the BatchNorm models. Skips finished work.
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
train() {  # name, args...
  local name=$1; shift
  if [ -f "runs/$name/summary.json" ]; then echo "skip $name"; return; fi
  mkdir -p "runs/$name"; echo "$(date +%T) start $name"
  $PY -m latent_planner.train --out "runs/$name" "$@" > "runs/$name/stdout.txt" 2>&1 || echo "FAILED $name"
  echo "$(date +%T) done $name"
}
for s in 0 1 2; do train ln_fs4_lam0.09_s$s --lam 0.09 --seed $s --frameskip 4 --proj-norm ln; done
for s in 0 1 2; do train push_ln_fs4_s$s --lam 0.09 --seed $s --frameskip 4 --proj-norm ln \
  --train data/push_train --val data/push_val; done
$PY -m latent_planner.eval --runs runs/ln_fs4_lam0.09_s0 runs/ln_fs4_lam0.09_s1 runs/ln_fs4_lam0.09_s2 \
  --taskset test --tag main --running-w 1.0 2>&1 | grep -v -i warn
$PY -m latent_planner.analyze --runs runs/ln_fs4_lam0.09_s0 runs/ln_fs4_lam0.09_s1 runs/ln_fs4_lam0.09_s2 2>&1 | grep -v -i warn
$PY -m latent_planner.eval_push --runs runs/push_ln_fs4_s0 runs/push_ln_fs4_s1 runs/push_ln_fs4_s2 \
  --taskset test --tag push 2>&1 | grep -v -i warn
echo LN_DONE
