"""Subgoal planning (latent waypoint graph) in the push scene. Frames/embeddings only for planning.

    .venv/bin/python scripts/push_subgoal.py --runs runs/push_fs4_s0 --taskset tune --pick beyond
Rows go to results/eval_push.csv (method latent_cem_subgoal_<pick>).
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from latent_planner import env_push as P  # noqa: E402
from latent_planner.eval_push import FIELDS, load_tasks, run_episodes  # noqa: E402
from latent_planner.planner import CEMConfig  # noqa: E402
from latent_planner.subgoal import SubgoalCEM, WaypointGraph  # noqa: E402
from latent_planner.train import load_model  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--runs", nargs="+", required=True)
ap.add_argument("--taskset", choices=["test", "tune"], default="tune")
ap.add_argument("--pick", choices=["within", "beyond"], default="beyond")
ap.add_argument("--bank-episodes", type=int, default=1000)
ap.add_argument("--knots", type=int, default=5)
ap.add_argument("--thr-pct", type=float, default=25.0)
ap.add_argument("--knn", type=int, default=8)
a = ap.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"
seed = {"test": 3030, "tune": 4040}[a.taskset]
tasks = load_tasks(ROOT / "results" / f"push_tasks_{a.taskset}.json", 100, seed)
cem = CEMConfig(knots=a.knots, hold=4, samples=300, iters=10, running_w=1.0)
tag = "push" if a.taskset == "test" else "push_dev"
for run in a.runs:
    model = load_model(ROOT / run / "model.pt", dev)
    t0 = time.time()
    g = WaypointGraph(model, str(ROOT / "data/push_train"), episodes=a.bank_episodes, stride=4, knn=a.knn,
                      thr_pct=a.thr_pct, device=dev)
    info = g.describe()
    pol = SubgoalCEM(model, tasks, cem, g, 1, 0, dev, pick=a.pick, render_batch=P.render_batch)
    r = run_episodes(pol, tasks)
    s = json.loads((ROOT / run / "summary.json").read_text())["args"]
    method = f"latent_cem_subgoal_{a.pick}"
    row = {"tag": tag, "taskset": a.taskset, "method": method, "run": run, "lam": s["lam"], "seed": s["seed"],
           "frameskip": s.get("frameskip", 1), "cem": f"K{a.knots}xh4 S300 I10 rw1.0 bank{a.bank_episodes} thr{a.thr_pct} knn{a.knn}", **r}
    path = ROOT / "results" / "eval_push.csv"
    with open(path, "a", newline="") as f:
        csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore").writerow(row)
    print(f"{method} {run}: success {r['success']:.2f} (short {r['success_short']:.2f}, long {r['success_long']:.2f}) "
          f"final block {r['final_block_dist_px']:.1f}px | graph {info} | {time.time() - t0:.0f}s", flush=True)
