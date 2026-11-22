# Longer tasks: horizon limitation and latent-waypoint subgoals

Frameskip-4 models (`runs/fs4_lam0.09_s{0,1,2}`), CEM as in the main results (300 samples, 30 elites, 10 iterations,
running cost weight 1.0, hold 4, execute 1 knot, re-plan), 50-step budget, success = within 4 px. K knots x 4
env steps of lookahead, so K = 1, 2, 3, 5 means 4, 8, 12, 20 env steps (at most 20, 40, 60, 100 px of travel).
Mean ± std over the 3 training seeds, one planning seed each. All numbers are in `eval.csv` and `eval/<tag>/`
in this folder.

Task sets:
- **test**: the 100 fixed test tasks (seed 20261122, half cross-room; shortest paths up to 92 px, median 33 px).
- **far**: 100 new cross-room tasks whose shortest path through the door is at least 70 px (seed 20261123,
  median 77 px, max 91 px; `tasks_far.json`). I built this set after step 1 showed the test tasks have little
  headroom at K >= 2. Positions are used only to choose the tasks.
- **tune**: the 100 tuning tasks (seed 777), used once, to choose between two subgoal rules (below).

## 1. The limitation

Plain latent CEM (`latent_planner.eval` for K = 2, 3, 5 on test; `latent_planner.subgoal` without `--subgoal`
for the rest; the two give identical results, checked at K = 2 seed 0):

| Task set | Horizon | Success | Same room | Other room | Per-seed success |
|---|---|---|---|---|---|
| test | K=1 (4 steps) | 81 ± 3% | 88 ± 2% | 73 ± 8% | 0.77, 0.85, 0.80 |
| test | K=2 (8 steps) | 99 ± 0% | 100 ± 0% | 99 ± 1% | 0.99, 0.99, 1.00 |
| test | K=3 (12 steps) | 99 ± 0% | 100 ± 0% | 97 ± 1% | 0.98, 0.99, 0.99 |
| test | K=5 (20 steps, reference) | 100 ± 0% | 100 ± 0% | 100 ± 0% | 1.00, 1.00, 1.00 |
| far | K=1 | 50 ± 3% | - | 50 ± 3% | 0.54, 0.51, 0.46 |
| far | K=2 | 83 ± 4% | - | 83 ± 4% | 0.79, 0.88, 0.83 |
| far | K=3 | 94 ± 2% | - | 94 ± 2% | 0.94, 0.92, 0.96 |
| far | K=5 | 99 ± 1% | - | 99 ± 1% | 1.00, 0.98, 0.99 |

The limitation is real but milder than WRITEUP finding 2 says. With 8 steps of lookahead (at most 40 px) the
planner still solves 99% of the test tasks, whose paths reach 92 px. The far field of the cost is not flat:
for start/goal pairs in different rooms, the median latent distance keeps rising with shortest-path distance,
7.1 at 20-30 px, 7.3 at 30-45, 8.1 at 45-60, 8.6 at 60-80 and 9.4 at 80-110 px (seed 0; seeds 1 and 2 are
within 0.3; `latent_vs_px.txt`, diagnostic only). Inside the basin the slope is ~0.4 per px, out here ~0.03
per px, weak but enough for CEM to follow. Success drops clearly only at K = 1, or with short horizons on
long tasks (far set: 50% at K = 1, 83% at K = 2). Failed runs end stuck: the agent moved a median 0.7 px over
its last 20 steps (test, K = 1, seed 0).

## 2. The fix: subgoals from a latent waypoint graph

`latent_planner/subgoal.py`, option `--subgoal` on a new entry point; `latent_planner.eval` is unchanged.

- **Bank**: every 4th frame of the first 400 training episodes, encoded: 5,200 nodes. Frames only.
- **Edges**: consecutive bank frames of an episode (4 env steps = one predictor step, so known reachable),
  cost 1; plus each node's 8 nearest latent neighbours closer than the 25th percentile of those transition
  distances (threshold ~3.0; median transition 5.0), cost d / median transition. All 8 neighbours fell under
  the threshold for every node, giving 4,800 + 41,600 edges, one connected component. Checked with positions
  afterwards (diagnostic only): no neighbour edge joins the two sides of the wall outside the doorway, and
  neighbour edges are short (median 0.7 px, max 3.9 px).
- **Subgoal**: at every MPC step, the node nearest the current embedding, the shortest path from it to the node
  nearest the goal embedding (read from a Dijkstra tree rooted at the goal node, the same path cost as
  re-running the search), and CEM's target is a node one horizon (K predictor steps of path cost) ahead; the
  goal embedding itself once the rest of the path is within one horizon.

**First rule, "within" (the farthest node whose path cost fits in the horizon), failed at K = 1**: 44% on test
and 2% on far, far below plain CEM. At K = 1 the horizon is one step of path cost, and a single temporal edge
already costs 1, so the farthest node that fit was often a neighbour edge ~1 px away. The agent reached it,
the next subgoal was ~1 px away again, and the agent stopped (failed runs moved a median 0.4 px over their
last 20 steps). **Second rule, "beyond"**: the first node at or past one horizon, so every subgoal makes
progress. I picked between the two on the tuning tasks at K = 1 only: plain 83 ± 2%, within 42 ± 2%,
beyond 95 ± 2%. Then I ran both rules on test and far.

## 3. Results

Test tasks:

| Planner | Horizon | Success | Same room | Other room | Steps (successes) | Per-seed success |
|---|---|---|---|---|---|---|
| plain latent CEM | K=1 (4) | 81 ± 3% | 88 ± 2% | 73 ± 8% | 8.2 | 0.77, 0.85, 0.80 |
| subgoals, within | K=1 (4) | 44 ± 3% | 66 ± 2% | 23 ± 5% | 15.0 | 0.49, 0.43, 0.41 |
| **subgoals, beyond** | K=1 (4) | **98 ± 1%** | 100 ± 0% | 95 ± 2% | 8.5 | 0.96, 0.98, 0.99 |
| plain latent CEM | K=2 (8) | 99 ± 0% | 100 ± 0% | 99 ± 1% | 7.9 | 0.99, 0.99, 1.00 |
| subgoals, within | K=2 (8) | 100 ± 0% | 100 ± 0% | 100 ± 0% | 7.9 | 1.00, 1.00, 1.00 |
| subgoals, beyond | K=2 (8) | 100 ± 0% | 100 ± 0% | 100 ± 0% | 7.5 | 1.00, 1.00, 1.00 |
| plain latent CEM | K=3 (12) | 99 ± 0% | 100 ± 0% | 97 ± 1% | 8.3 | 0.98, 0.99, 0.99 |
| subgoals, within | K=3 (12) | 99 ± 0% | 100 ± 0% | 98 ± 0% | 8.1 | 0.99, 0.99, 0.99 |
| subgoals, beyond | K=3 (12) | 99 ± 0% | 100 ± 0% | 98 ± 0% | 8.2 | 0.99, 0.99, 0.99 |
| plain latent CEM | K=5 (20) | 100 ± 0% | 100 ± 0% | 100 ± 0% | 9.4 | 1.00, 1.00, 1.00 |
| subgoals, within | K=5 (20) | 100 ± 0% | 100 ± 0% | 100 ± 0% | 9.4 | 1.00, 1.00, 1.00 |
| subgoals, beyond | K=5 (20) | 100 ± 0% | 100 ± 0% | 100 ± 0% | 9.3 | 1.00, 1.00, 1.00 |

Far tasks (all cross-room, shortest path >= 70 px):

| Planner | Horizon | Success | Steps (successes) | Per-seed success |
|---|---|---|---|---|
| plain latent CEM | K=1 (4) | 50 ± 3% | 19.5 | 0.54, 0.51, 0.46 |
| subgoals, within | K=1 (4) | 2 ± 1% | 36.0 | 0.01, 0.01, 0.03 |
| **subgoals, beyond** | K=1 (4) | **83 ± 7%** | 19.9 | 0.73, 0.85, 0.90 |
| plain latent CEM | K=2 (8) | 83 ± 4% | 18.2 | 0.79, 0.88, 0.83 |
| subgoals, within | K=2 (8) | 95 ± 0% | 17.9 | 0.95, 0.95, 0.96 |
| **subgoals, beyond** | K=2 (8) | **100 ± 0%** | 16.1 | 1.00, 0.99, 1.00 |
| plain latent CEM | K=3 (12) | 94 ± 2% | 17.6 | 0.94, 0.92, 0.96 |
| subgoals, within | K=3 (12) | 99 ± 0% | 16.4 | 0.99, 0.99, 0.98 |
| subgoals, beyond | K=3 (12) | 97 ± 0% | 16.7 | 0.97, 0.97, 0.98 |
| plain latent CEM | K=5 (20) | 99 ± 1% | 18.4 | 1.00, 0.98, 0.99 |
| subgoals, within | K=5 (20) | 99 ± 1% | 18.5 | 1.00, 0.98, 0.98 |
| subgoals, beyond | K=5 (20) | 99 ± 1% | 18.4 | 1.00, 0.97, 0.99 |

GPU peak 0.13 GB, host RSS peak 1.2 GB; building the graph takes 0.4-2.5 s.

## Interpretation

- Subgoals help exactly where the horizon is short relative to the task: K = 1 goes from 81% to 98% on test and
  from 50% to 83% on far, and far tasks at K = 2 from 83% to 100%. They also shorten successful paths slightly
  on far tasks (16.1 vs 18.2 steps at K = 2).
- They don't hurt at K = 5 (identical within one task per seed), and they change nothing where plain CEM
  already succeeds (test at K >= 2). At K = 5 the rest of the graph path fits in one horizon at the first step
  for 98% of test tasks and 80% of far tasks (100% from the second step), so the planner aims at the goal
  itself almost from the start.
- The subgoal rule matters more than the graph: the same graph with "within" is far worse than no graph at
  K = 1 (2% on far). Subgoals have to be far enough ahead to force progress.
- At K = 3 on far, "within" (99%) edges out "beyond" (97%); with one planning seed, a gap of 2 tasks per seed is
  near the noise. "Beyond" was chosen on the tuning tasks at K = 1 and kept for all horizons.

## Limitations

- The world is tiny: 5,200 bank frames cover both rooms at about 2 frames per px² of reachable area, and the bank comes
  from the training data of the same environment. Nearest-node lookup is then easy. A larger or partly
  observed world would need a sparser bank, and graph errors (wrong-room shortcuts, missing links) would matter.
- The far task set was designed after seeing step 1, and the K = 2/3 within-vs-beyond comparison on far is a
  look at evaluation data, not a tuned choice. Only the K = 1 rule choice used the tuning tasks.
- Path cost is in predictor steps only for temporal edges; neighbour edges use a latent-distance fraction, a
  heuristic. The threshold (25th percentile) and k = 8 were set once from the transition-distance statistics
  and never tuned.
- One planning seed per model; the ± is training-seed variance only.
- ms/replan (in `eval.csv`) was measured while another training job shared the GPU and varies a lot run to run
  (110 to 313 ms for the same configuration across seeds), so it is not a measure of subgoal overhead.

## Reproduce

```bash
# plain CEM at K = 2, 3, 5 on test
for K in 2 3 5; do .venv/bin/python -m latent_planner.eval --runs runs/fs4_lam0.09_s{0,1,2} --taskset test \
  --tag horizon_K$K --knots $K --hold 4 --running-w 1.0 --results results/horizon; done
scripts/eval_subgoal.sh                  # everything else (writes results/horizon/eval.csv)
.venv/bin/python scripts/latent_vs_px.py > results/horizon/latent_vs_px.txt
.venv/bin/python scripts/subgoal_table.py
```
