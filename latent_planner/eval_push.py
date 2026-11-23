"""Scene 2 evaluation: push the block to the goal shown in a goal image.

    python -m latent_planner.eval_push --runs runs/push_fs4_s0 --baselines --taskset test --tag push

Tasks come from env_push.make_tasks: each goal is the end state of a 30-step run of the push behaviour policy
from the start, so it is reachable and its image shows the agent where a pusher would plausibly be. Success =
block centre within SUCCESS_PX of the goal block centre at any step within BUDGET env steps. Results are split
into short (< 20 px) and long block moves and appended to results/eval_push.csv.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch

from . import env_push as P
from .eval import LatentCEM
from .planner import CEMConfig, cem_plan_oracle
from .train import load_model

SUCCESS_PX = 3.0
BUDGET = 60
LONG_PX = 20.0


INNER_LO = P.BORDER + 2 * P.AGENT_R + P.BLOCK_H  # block centre in [12, 52]: pushable from all four sides
INNER_HI = P.SIZE - P.BORDER - 2 * P.AGENT_R - P.BLOCK_H


def make_interior_tasks(n: int, seed: int, steps: int = 12, min_move: float = 6.0) -> dict:
    """Like env_push.make_tasks, but every goal leaves the block in the open (not against a wall), so the agent has
    to stop the block itself. 94 of the 100 standard test goals have the block touching a wall, because the 30-step
    pusher that makes them almost always runs into one. Shorter pushes, filtered: still reachable by construction."""
    rng = np.random.default_rng(seed)
    starts, goals, moved = [], [], []
    while len(starts) < n:
        S, _ = P.rollout(rng, steps, "push")
        d = float(np.linalg.norm(S[-1, 2:] - S[0, 2:]))
        g = S[-1, 2:]
        if d >= min_move and (g >= INNER_LO).all() and (g <= INNER_HI).all():
            starts.append(S[0])
            goals.append(S[-1])
            moved.append(d)
    return {"seed": seed, "starts": np.array(starts).tolist(), "goals": np.array(goals).tolist(), "moved": moved}


def load_tasks(path: Path, n: int, seed: int, interior: bool = False) -> dict:
    if path.exists():
        t = json.loads(path.read_text())
        assert t["seed"] == seed and len(t["starts"]) == n, f"{path} has different tasks"
        return t
    t = make_interior_tasks(n, seed) if interior else P.make_tasks(n, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(t))
    return t


def run_episodes(policy, tasks: dict, budget: int = BUDGET) -> dict:
    s = np.array(tasks["starts"], dtype=np.float64)
    goal = np.array(tasks["goals"], dtype=np.float64)
    N = s.shape[0]
    done = np.zeros(N, bool)
    steps = np.full(N, budget + 1)
    traj, t, plan_times = [s.copy()], 0, []
    while t < budget and not done.all():
        t0 = time.time()
        acts = policy(s, ~done, t)
        plan_times.append(time.time() - t0)
        for j in range(acts.shape[1]):
            if t >= budget:
                break
            s = np.where(done[:, None], s, P.step_state_batch(s, acts[:, j]))
            t += 1
            reached = (~done) & (np.linalg.norm(s[:, 2:] - goal[:, 2:], axis=1) <= SUCCESS_PX)
            steps[reached] = t
            done |= reached
            traj.append(s.copy())
    long_ = np.array(tasks["moved"]) >= LONG_PX
    return {
        "success": float(done.mean()), "success_short": float(done[~long_].mean()),
        "success_long": float(done[long_].mean()),
        "mean_steps_success": float(steps[done].mean()) if done.any() else float("nan"),
        "final_block_dist_px": float(np.linalg.norm(s[:, 2:] - goal[:, 2:], axis=1).mean()),
        "sec_per_replan": float(np.mean(plan_times)) if plan_times else 0.0,
        "per_task_success": done.astype(int).tolist(), "traj": np.stack(traj, 1).round(2).tolist(),
    }


class OracleCEM:
    """Same CEM with the true push simulator; cost = distance between full true states (agent and block)."""

    def __init__(self, tasks: dict, cfg: CEMConfig, seed: int = 0):
        self.goal = np.array(tasks["goals"], dtype=np.float64)
        self.cfg, self.rng, self.mean = cfg, np.random.default_rng(seed), None

    def __call__(self, s, active, t):
        init = None if self.mean is None else np.concatenate([self.mean[:, 1:], np.zeros_like(self.mean[:, :1])], 1)
        self.mean = cem_plan_oracle(s, self.goal, self.cfg, self.rng, init, step_fn=P.step_state_batch)
        return np.repeat(self.mean[:, :1], self.cfg.hold, axis=1)


def scripted_policy(tasks: dict):
    """Hand-written pusher with true positions: get behind the block (opposite the goal direction), then push
    toward the goal. Moves around the block, not through it, when it has to switch sides."""
    goal = np.array(tasks["goals"], dtype=np.float64)

    def f(s, active, t):
        out = np.zeros((len(s), 1, 2))
        for k in range(len(s)):
            a, b, g = s[k, :2], s[k, 2:], goal[k, 2:]
            d = g - b
            if abs(d[0]) >= abs(d[1]):
                side = np.array([np.sign(d[0]) or 1.0, 0.0])
            else:
                side = np.array([0.0, np.sign(d[1]) or 1.0])
            behind = b - side * (P.BLOCK_H + P.AGENT_R + 0.5)
            rel = a - b
            if np.linalg.norm(a - behind) < 1.0:
                act = side * min(1.0, np.abs(d).max() / P.SPEED)
            elif np.dot(rel, side) > -(P.BLOCK_H + P.AGENT_R) + 0.5:  # on the wrong side: go around
                perp = np.array([side[1], side[0]])
                perp = perp if np.dot(rel, perp) >= 0 else -perp
                waypoint = b + perp * (P.BLOCK_H + P.AGENT_R + 3) - side * (P.BLOCK_H + P.AGENT_R + 3)
                act = (waypoint - a) / max(np.linalg.norm(waypoint - a), 1e-6)
            else:
                v = behind - a
                act = v / max(np.linalg.norm(v), 1e-6) * min(1.0, np.linalg.norm(v) / P.SPEED)
            out[k, 0] = act
        return out

    return f


def random_policy(seed: int):
    rng = np.random.default_rng(seed)
    return lambda s, active, t: rng.uniform(-1, 1, (s.shape[0], 1, 2))


FIELDS = ["tag", "taskset", "method", "run", "lam", "seed", "frameskip", "success", "success_short", "success_long",
          "mean_steps_success", "final_block_dist_px", "sec_per_replan", "cem"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=[])
    ap.add_argument("--baselines", action="store_true")
    ap.add_argument("--taskset", choices=["test", "tune", "interior"], default="test")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--tag", default="push")
    ap.add_argument("--knots", type=int, default=5)
    ap.add_argument("--hold", type=int, default=4)
    ap.add_argument("--samples", type=int, default=300)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--running-w", type=float, default=1.0)
    ap.add_argument("--plan-seed", type=int, default=0)
    ap.add_argument("--budget", type=int, default=BUDGET, help="env steps per task (the demo allows 120)")
    ap.add_argument("--results", default="results")
    a = ap.parse_args()
    torch.set_num_threads(4)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    res = Path(a.results)
    seed = {"test": 3030, "tune": 4040, "interior": 5050}[a.taskset]
    tasks = load_tasks(res / f"push_tasks_{a.taskset}.json", a.n, seed, interior=a.taskset == "interior")
    cem = CEMConfig(knots=a.knots, hold=a.hold, samples=a.samples, iters=a.iters, running_w=a.running_w)
    cem_str = f"K{cem.knots}xh{cem.hold} S{cem.samples} I{cem.iters} rw{cem.running_w}"
    out_dir = res / "eval" / a.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    def record(method, run, r, extra):
        row = {"tag": a.tag, "taskset": a.taskset, "method": method, "run": run, "cem": cem_str, **extra, **r}
        path = res / "eval_push.csv"
        new = not path.exists()
        with open(path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
            if new:
                w.writeheader()
            w.writerow(row)
        parts = Path(run).parts if run else ()
        name = "_".join(parts[1:]) if len(parts) > 1 and parts[0] == "runs" else (Path(run).name or "none")
        (out_dir / f"{a.taskset}__{method}__{name}.json").write_text(json.dumps(row))  # runs/a/b -> a_b
        print(f"{method:10s} {run:24s} success {r['success']:.2f} (short {r['success_short']:.2f}, long "
              f"{r['success_long']:.2f}) steps {r['mean_steps_success']:.1f} final block {r['final_block_dist_px']:.1f}px "
              f"{r['sec_per_replan'] * 1000:.0f} ms/replan [{cem_str}]", flush=True)

    if a.baselines:
        record("random", "", run_episodes(random_policy(a.plan_seed), tasks), {})
        record("scripted", "", run_episodes(scripted_policy(tasks), tasks), {})
        record("oracle_cem", "", run_episodes(OracleCEM(tasks, cem, a.plan_seed), tasks), {})
    for run in a.runs:
        model = load_model(Path(run) / "model.pt", dev)
        s = json.loads((Path(run) / "summary.json").read_text())["args"]
        pol = LatentCEM(model, tasks, cem, 1, a.plan_seed, dev, render_batch=P.render_batch)
        record("latent_cem", run, run_episodes(pol, tasks, a.budget),
               {"lam": s["lam"], "seed": s["seed"], "frameskip": s.get("frameskip", 1)})


if __name__ == "__main__":
    main()
