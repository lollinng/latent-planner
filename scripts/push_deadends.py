"""Count push failures that are provably unrecoverable ("dead ends") from saved eval trajectories.

The agent can only push. To push the block away from a wall it must stand between block and wall, which needs a
gap of 2*AGENT_R. So once the block centre is within that band of a wall (x < 12 or x > 52 px, same for y), that
coordinate can only move toward the wall. If the goal is more than SUCCESS_PX the other way, the task can no longer
be solved, whatever the planner does. This check is sufficient, not necessary: it finds only this kind of dead end.

    python scripts/push_deadends.py results/eval/push_dagger_r1/test__latent_cem__kdagger_r1drop_s*_round1.json
"""

from __future__ import annotations

import json
import sys

import numpy as np

sys.path.insert(0, ".")
from latent_planner import env_push as P  # noqa: E402
from latent_planner.eval_push import SUCCESS_PX  # noqa: E402

LO = P.BORDER + 2 * P.AGENT_R + P.BLOCK_H  # below this the block can't be pushed toward +axis
HI = P.SIZE - P.BORDER - 2 * P.AGENT_R - P.BLOCK_H
EPS = 0.02  # trajectories are stored rounded to 0.01 px


def dead(block: np.ndarray, goal: np.ndarray) -> np.ndarray:
    """(T, 2) block positions, (2,) goal block -> (T,) bool: goal provably out of reach from this step on."""
    low = (block < LO - EPS) & (goal[None] > block + SUCCESS_PX + EPS)
    high = (block > HI + EPS) & (goal[None] < block - SUCCESS_PX - EPS)
    return (low | high).any(1)


def analyse(path: str, tasks: dict) -> dict:
    d = json.load(open(path))
    traj = np.array(d["traj"])  # (N, T, 4)
    goals = np.array(tasks["goals"])[:, 2:]
    ok = np.array(d["per_task_success"], bool)
    first = []
    for k in range(len(traj)):
        m = dead(traj[k, :, 2:], goals[k])
        first.append(int(m.argmax()) if m.any() else -1)
    first = np.array(first)
    goal_band = ((goals < LO) | (goals > HI)).any(1)
    return {"file": path.split("/")[-1], "n": len(ok), "fail": int((~ok).sum()),
            "fail_dead": int(((~ok) & (first >= 0)).sum()), "ok_but_flagged": int((ok & (first >= 0)).sum()),
            "median_step_dead": float(np.median(first[(~ok) & (first >= 0)])) if ((~ok) & (first >= 0)).any() else None,
            "goal_near_wall": int(goal_band.sum()), "fail_goal_near_wall": int(((~ok) & goal_band).sum())}


def main() -> None:
    tasks = json.load(open("results/push_tasks_test.json"))
    rows = [analyse(p, tasks) for p in sys.argv[1:]]
    print(f"{'file':58s} fail  dead-end  ok-but-flagged  median-step  goal-near-wall(fail)")
    for r in rows:
        print(f"{r['file']:58s} {r['fail']:4d}  {r['fail_dead']:8d}  {r['ok_but_flagged']:14d}  "
              f"{str(r['median_step_dead']):>11s}  {r['goal_near_wall']:3d} ({r['fail_goal_near_wall']})")


if __name__ == "__main__":
    main()
