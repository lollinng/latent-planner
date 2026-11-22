#!/usr/bin/env bash
# Horizon limitation and latent-waypoint subgoals (results/horizon/SUBGOAL.md). One GPU job at a time.
# Plain latent CEM at K = 2, 3, 5 on the test tasks comes from latent_planner.eval (see SUBGOAL.md for the command).
set -u
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
RUNS="runs/fs4_lam0.09_s0 runs/fs4_lam0.09_s1 runs/fs4_lam0.09_s2"
sg() { $PY -m latent_planner.subgoal --runs $RUNS "$@" 2>&1 | grep -v -i warn; }
# sanity: the plain mode of the new entry point reproduces latent_planner.eval (K=2, seed 0)
$PY -m latent_planner.subgoal --runs runs/fs4_lam0.09_s0 --knots 2 --taskset test --tag check_plain_K2 2>&1 | grep -v -i warn
# test tasks: K=1 plain, subgoal at K=1, 2, 3, 5
sg --knots 1 --taskset test --tag plain_K1
for K in 1 2 3 5; do sg --knots $K --taskset test --tag subgoal_K$K --subgoal; done
# far tasks (cross-room, shortest path >= 70 px): plain and subgoal at K=1, 2, 3, 5
for K in 1 2 3 5; do
  sg --knots $K --taskset far --tag plain_K$K
  sg --knots $K --taskset far --tag subgoal_K$K --subgoal
done
# pick rule chosen on the tuning tasks at K=1: "within" (first version) vs "beyond"
sg --knots 1 --taskset tune --tag plain_K1
sg --knots 1 --taskset tune --tag subgoal_K1 --subgoal
sg --knots 1 --taskset tune --tag subgoal_beyond_K1 --subgoal --pick beyond
# the chosen rule ("beyond") on the test and far tasks
for TS in test far; do
  for K in 1 2 3 5; do sg --knots $K --taskset $TS --tag subgoal_beyond_K$K --subgoal --pick beyond; done
done
echo SUBGOAL_DONE
