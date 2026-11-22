"""Goal-reaching evaluation: latent CEM vs baselines on fixed (start, goal) tasks.

    python -m latent_planner.eval --runs runs/lam0.09_s0 runs/lam0.09_s1 --baselines --tag main

Success = agent centre within SUCCESS_PX of the goal centre at any step within BUDGET env steps (the episode stops
there). Tasks are fixed by seed and saved to results/; half of them start and end in different rooms. Every
number is appended to results/eval.csv, one row per (model or baseline, task set).
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch

from . import env
from .planner import CEMConfig, cem_plan, cem_plan_oracle, shift_plan
from .train import load_model

SUCCESS_PX = 4.0
BUDGET = 50
MIN_START_GOAL_PX = 12.0


def make_tasks(n: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    starts, goals, cross = [], [], []
    for k in range(n):
        c = k % 2 == 1
        while True:
            s = env.sample_free(rng)
            g = env.sample_free(rng, room=(1 - env.room_of(s[0])) if c else env.room_of(s[0]))
            if np.linalg.norm(s - g) >= MIN_START_GOAL_PX:
                break
        starts.append(s)
        goals.append(g)
        cross.append(c)
    return {"seed": seed, "starts": np.array(starts).tolist(), "goals": np.array(goals).tolist(), "cross": cross}


def load_tasks(path: Path, n: int, seed: int) -> dict:
    if path.exists():
        t = json.loads(path.read_text())
        assert t["seed"] == seed and len(t["starts"]) == n, f"{path} has different tasks"
        return t
    t = make_tasks(n, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(t))
    return t


def run_episodes(policy, tasks: dict, budget: int = BUDGET) -> dict:
    """policy(pos (N,2), active mask (N,), t) -> actions (N, n, 2) to execute open loop before replanning."""
    pos = np.array(tasks["starts"], dtype=np.float64)
    goal = np.array(tasks["goals"], dtype=np.float64)
    N = pos.shape[0]
    done = np.zeros(N, bool)
    steps = np.full(N, budget + 1)
    traj = [pos.copy()]
    t, plan_times = 0, []
    while t < budget and not done.all():
        t_plan = time.time()
        acts = policy(pos, ~done, t)
        plan_times.append(time.time() - t_plan)
        for j in range(acts.shape[1]):
            if t >= budget:
                break
            new = env.step_pos_batch(pos, acts[:, j])
            pos = np.where(done[:, None], pos, new)
            t += 1
            reached = (~done) & (np.linalg.norm(pos - goal, axis=1) <= SUCCESS_PX)
            steps[reached] = t
            done |= reached
            traj.append(pos.copy())
    cross = np.array(tasks["cross"])
    fd = np.linalg.norm(pos - goal, axis=1)
    return {
        "success": float(done.mean()), "success_same": float(done[~cross].mean()),
        "success_cross": float(done[cross].mean()),
        "mean_steps_success": float(steps[done].mean()) if done.any() else float("nan"),
        "final_dist_px": float(fd.mean()), "replans": len(plan_times),
        "sec_per_replan": float(np.mean(plan_times)) if plan_times else 0.0,
        "per_task_success": done.astype(int).tolist(), "traj": np.stack(traj, 1).round(2).tolist(),
    }


class LatentCEM:
    def __init__(self, model, tasks: dict, cfg: CEMConfig, exec_knots: int = 1, seed: int = 0, device="cuda",
                 render_batch=None):
        self.model, self.cfg, self.exec, self.dev = model, cfg, exec_knots, device
        self.render = render_batch or env.render_batch
        goal = np.array(tasks["goals"], dtype=np.float64)
        with torch.no_grad():
            self.zg = model.encode(torch.from_numpy(self.render(goal)).to(device))
        self.mean = None
        self.gen = torch.Generator(device=device)
        self.gen.manual_seed(seed)

    def __call__(self, pos, active, t):
        with torch.no_grad():
            z0 = self.model.encode(torch.from_numpy(self.render(pos)).to(self.dev))
        init = None if self.mean is None else shift_plan(self.mean, self.exec)
        self.mean, _ = cem_plan(self.model, z0, self.zg, self.cfg, init, self.gen)
        knots = self.mean[:, : self.exec].cpu().numpy()
        return np.repeat(knots, self.cfg.hold, axis=1)


class OracleCEM:
    def __init__(self, tasks: dict, cfg: CEMConfig, exec_knots: int = 1, seed: int = 0):
        self.goal = np.array(tasks["goals"], dtype=np.float64)
        self.cfg, self.exec, self.rng, self.mean = cfg, exec_knots, np.random.default_rng(seed), None

    def __call__(self, pos, active, t):
        init = None
        if self.mean is not None:
            init = np.concatenate([self.mean[:, self.exec :], np.zeros_like(self.mean[:, : self.exec])], 1)
        self.mean = cem_plan_oracle(pos, self.goal, self.cfg, self.rng, init)
        return np.repeat(self.mean[:, : self.exec], self.cfg.hold, axis=1)


def random_policy(seed: int):
    rng = np.random.default_rng(seed)
    return lambda pos, active, t: rng.uniform(-1, 1, (pos.shape[0], 1, 2))


def greedy_policy(tasks: dict):
    """Straight line to the goal using the true positions (ignores walls)."""
    goal = np.array(tasks["goals"], dtype=np.float64)

    def f(pos, active, t):
        d = goal - pos
        n = np.linalg.norm(d, axis=1, keepdims=True)
        return (d / np.maximum(n, 1e-6) * np.minimum(1.0, n / env.SPEED))[:, None]

    return f


FIELDS = ["tag", "taskset", "method", "run", "lam", "seed", "frameskip", "success", "success_same", "success_cross",
          "mean_steps_success", "final_dist_px", "replans", "sec_per_replan", "cem"]


def append_csv(path: Path, row: dict) -> None:
    new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow(row)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=[])
    ap.add_argument("--baselines", action="store_true")
    ap.add_argument("--taskset", choices=["test", "tune"], default="test")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--tag", type=str, default="main")
    ap.add_argument("--knots", type=int, default=5)
    ap.add_argument("--hold", type=int, default=4)
    ap.add_argument("--samples", type=int, default=300)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--elites", type=int, default=30)
    ap.add_argument("--running-w", type=float, default=0.0)
    ap.add_argument("--exec-knots", type=int, default=1)
    ap.add_argument("--plan-seed", type=int, default=0)
    ap.add_argument("--results", type=str, default="results")
    args = ap.parse_args()

    torch.set_num_threads(4)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    res = Path(args.results)
    seed = {"test": 20261122, "tune": 777}[args.taskset]
    tasks = load_tasks(res / f"tasks_{args.taskset}.json", args.n, seed)
    cem = CEMConfig(knots=args.knots, hold=args.hold, samples=args.samples, iters=args.iters, elites=args.elites,
                    running_w=args.running_w)
    cem_str = f"K{cem.knots}xh{cem.hold} S{cem.samples} I{cem.iters} E{cem.elites} rw{cem.running_w} x{args.exec_knots}"
    out_dir = res / "eval" / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    def record(method: str, run: str, r: dict, extra: dict) -> None:
        row = {"tag": args.tag, "taskset": args.taskset, "method": method, "run": run, "cem": cem_str, **extra, **r}
        append_csv(res / "eval.csv", row)
        (out_dir / f"{args.taskset}__{method}__{Path(run).name or 'none'}.json").write_text(json.dumps(row))
        print(f"{method:10s} {run:28s} success {r['success']:.2f} (same {r['success_same']:.2f}, cross "
              f"{r['success_cross']:.2f}) steps {r['mean_steps_success']:.1f} final {r['final_dist_px']:.1f}px "
              f"{r['sec_per_replan'] * 1000:.0f} ms/replan", flush=True)

    if args.baselines:
        record("random", "", run_episodes(random_policy(args.plan_seed), tasks), {})
        record("greedy", "", run_episodes(greedy_policy(tasks), tasks), {})
        record("oracle_cem", "", run_episodes(OracleCEM(tasks, cem, args.exec_knots, args.plan_seed), tasks), {})
    for run in args.runs:
        model = load_model(Path(run) / "model.pt", dev)
        a = json.loads((Path(run) / "summary.json").read_text())["args"]
        r = run_episodes(LatentCEM(model, tasks, cem, args.exec_knots, args.plan_seed, dev), tasks)
        record("latent_cem", run, r, {"lam": a["lam"], "seed": a["seed"], "frameskip": a.get("frameskip", 1)})


if __name__ == "__main__":
    main()
