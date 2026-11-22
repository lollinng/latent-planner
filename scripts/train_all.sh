#!/usr/bin/env bash
# Train the experiment matrix sequentially (one run fits in the 4 GB GPU at a time). Skips finished runs.
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
run() {  # name, extra args...
  local name=$1; shift
  if [ -f "runs/$name/summary.json" ]; then echo "skip $name"; return; fi
  mkdir -p "runs/$name"
  echo "$(date +%T) start $name"
  $PY -m latent_planner.train --out "runs/$name" "$@" > "runs/$name/stdout.txt" 2>&1 || echo "FAILED $name"
  echo "$(date +%T) done $name"
}
for s in 0 1 2; do run "lam0.09_s$s" --lam 0.09 --seed $s; done
for s in 0 1 2; do run "lam0_s$s" --lam 0 --seed $s; done
for s in 0 1 2; do run "fs4_lam0.09_s$s" --lam 0.09 --seed $s --frameskip 4; done
echo ALL_DONE
