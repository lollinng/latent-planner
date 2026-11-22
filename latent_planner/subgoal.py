"""Subgoals from a latent waypoint graph, for goals beyond the planner's horizon.

    python -m latent_planner.subgoal --runs runs/fs4_lam0.09_s0 --knots 2 --subgoal --taskset test

The latent cost (distance to the goal embedding) is a sharp basin around the goal, so CEM only gets a strong
signal when the goal is within its imagined horizon. This module gives CEM a nearer target instead:

1. Bank: embed every `stride`-th frame of the first `episodes` training episodes (frames only, no positions).
2. Graph: connect consecutive bank frames of an episode (stride env steps apart = one predictor step for a
   frameskip-`stride` model, so known reachable) with cost 1, plus each node's k nearest latent neighbours whose
   distance is below a threshold calibrated from those transition distances (a low percentile), with cost
   d / median transition distance (a fraction of a step).
3. At every MPC step: find the node nearest the current embedding and the node nearest the goal embedding, take
   the shortest path between them, and hand CEM the node about one horizon (K*hold/frameskip steps of path cost)
   ahead on that path; the goal embedding itself once the rest of the path fits in the horizon.

Shortest paths are read from a tree rooted at each task's goal node (Dijkstra on the undirected graph, computed
once per task); reading the path from the current node at every step gives the same path cost as re-running the
search from it. The planner only ever sees frames and embeddings; positions are used here only to build the
"far" evaluation task set and in `edge_report` (diagnostics).

Results go to <results>/eval.csv (default results/horizon), never to results/eval.csv.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, dijkstra

from . import data as data_mod
from . import env
from .eval import LatentCEM, append_csv, load_tasks, run_episodes
from .planner import CEMConfig, cem_plan, shift_plan
from .train import load_model


class WaypointGraph:
    """Bank of training-frame embeddings connected into a graph. Built from frames only."""

    def __init__(self, model, data_dir: str, episodes: int = 400, stride: int = 4, knn: int = 8,
                 thr_pct: float = 25.0, device: str = "cpu", chunk: int = 20):
        self.stride, self.knn, self.thr_pct = stride, knn, thr_pct
        zs = []
        with torch.no_grad():
            for i in range(0, episodes, chunk):
                f = np.ascontiguousarray(data_mod.read_frames(data_dir, i, min(i + chunk, episodes))[:, ::stride])
                zs.append(model.encode(torch.from_numpy(f).to(device)).cpu())
        Z = torch.cat(zs)  # (E, T, D)
        E, T, D = Z.shape
        self.E, self.T = E, T
        self.z = Z.reshape(E * T, D)  # node i = (episode i // T, bank frame i % T)
        n = E * T
        # temporal edges: consecutive bank frames of an episode, one predictor step apart
        idx = torch.arange(n).reshape(E, T)
        ta, tb = idx[:, :-1].reshape(-1), idx[:, 1:].reshape(-1)
        dtrans = (self.z[ta] - self.z[tb]).norm(dim=1)
        self.d_med = float(dtrans.median())
        self.thr = float(np.percentile(dtrans.numpy(), thr_pct))
        # nearest-neighbour edges below the threshold
        na, nb, nd = [], [], []
        for i0 in range(0, n, 1024):
            d = torch.cdist(self.z[i0 : i0 + 1024], self.z)  # (b, n)
            d[torch.arange(d.shape[0]), torch.arange(i0, i0 + d.shape[0])] = float("inf")
            v, j = d.topk(knn, dim=1, largest=False)
            ok = v < self.thr
            rows = torch.arange(i0, i0 + d.shape[0])[:, None].expand_as(j)
            na.append(rows[ok]), nb.append(j[ok]), nd.append(v[ok])
        na, nb, nd = torch.cat(na), torch.cat(nb), torch.cat(nd)
        a = torch.cat([ta, na]).numpy()
        b = torch.cat([tb, nb]).numpy()
        w = torch.cat([torch.ones_like(dtrans), (nd / self.d_med).clamp_min(1e-3)]).numpy()
        # undirected graph stored as upper triangle; for duplicate pairs keep the cheaper edge
        lo, hi = np.minimum(a, b), np.maximum(a, b)
        order = np.argsort(w, kind="stable")
        _, first = np.unique((lo * n + hi)[order], return_index=True)
        keep = order[first]
        self.G = coo_matrix((w[keep], (lo[keep], hi[keep])), shape=(n, n)).tocsr()
        self.n_temporal, self.n_knn = len(ta), len(na)
        self.n_components = int(connected_components(self.G, directed=False)[0])
        self.edges = (a, b)

    def nearest(self, z: torch.Tensor) -> np.ndarray:
        return torch.cdist(z.cpu().float(), self.z).argmin(1).numpy()

    def describe(self) -> dict:
        return {"nodes": int(self.z.shape[0]), "temporal_edges": self.n_temporal, "knn_edges": self.n_knn,
                "thr": round(self.thr, 3), "d_med": round(self.d_med, 3), "components": self.n_components}

    def edge_report(self, data_dir: str) -> dict:
        """Diagnostics only (uses true positions): how many kNN edges join states the agent can't move between
        directly (opposite sides of the wall, neither in the doorway), and their pixel lengths."""
        pos = np.load(Path(data_dir) / "pos.npy")[: self.E, :: self.stride].reshape(-1, 2)
        a, b = self.edges
        a, b = a[self.n_temporal :], b[self.n_temporal :]
        pa, pb = pos[a], pos[b]

        def in_door(p):
            return (np.abs(p[:, 0] - 32) < 6) & (p[:, 1] > env.DOOR_Y0) & (p[:, 1] < env.DOOR_Y1)

        cross = (np.sign(pa[:, 0] - 32) != np.sign(pb[:, 0] - 32)) & ~in_door(pa) & ~in_door(pb)
        px = np.linalg.norm(pa - pb, axis=1)
        return {"knn_wall_crossing": int(cross.sum()), "knn_px_p50": float(np.median(px)),
                "knn_px_p99": float(np.percentile(px, 99)), "knn_px_max": float(px.max())}


class SubgoalCEM(LatentCEM):
    """LatentCEM whose CEM target at each MPC step is a waypoint on the graph path to the goal."""

    def __init__(self, model, tasks: dict, cfg: CEMConfig, graph: WaypointGraph, exec_knots: int = 1,
                 seed: int = 0, device: str = "cuda", pick: str = "within", render_batch=None):
        super().__init__(model, tasks, cfg, exec_knots, seed, device, render_batch=render_batch)
        self.g, self.pick = graph, pick
        self.reach = cfg.knots * cfg.hold / model.cfg.frameskip  # horizon in predictor steps
        self.goal_node = graph.nearest(self.zg)
        uniq, inv = np.unique(self.goal_node, return_inverse=True)
        dist, pred = dijkstra(graph.G, directed=False, indices=uniq, return_predecessors=True)
        self.dist, self.pred, self.inv = dist, pred, inv  # tree rooted at each goal node
        self.log = []  # per MPC step: fraction of tasks aimed at the goal itself, mean path cost left

    def subgoals(self, z0: torch.Tensor) -> torch.Tensor:
        s = self.g.nearest(z0)
        zt = self.zg.clone()
        left = np.zeros(len(s))
        for i, (si, k) in enumerate(zip(s, self.inv)):
            dist, pred = self.dist[k], self.pred[k]
            left[i] = dist[si]
            if not np.isfinite(dist[si]) or dist[si] <= self.reach:
                continue  # goal within reach (or unreachable on the graph): aim at the goal itself
            node = si
            if self.pick == "within":  # farthest path node whose path cost is within the horizon
                while pred[node] >= 0 and dist[si] - dist[pred[node]] <= self.reach:
                    node = pred[node]
            else:  # "beyond": first path node at or past the horizon (always makes progress)
                while pred[node] >= 0 and dist[si] - dist[node] < self.reach:
                    node = pred[node]
            zt[i] = self.g.z[node].to(zt.device)
        self.log.append((float((left <= self.reach).mean()), float(np.nanmean(np.where(np.isfinite(left), left, np.nan)))))
        return zt

    def __call__(self, pos, active, t):
        with torch.no_grad():
            z0 = self.model.encode(torch.from_numpy(self.render(pos)).to(self.dev))
        zt = self.subgoals(z0)
        init = None if self.mean is None else shift_plan(self.mean, self.exec)
        self.mean, _ = cem_plan(self.model, z0, zt, self.cfg, init, self.gen)
        knots = self.mean[:, : self.exec].cpu().numpy()
        return np.repeat(knots, self.cfg.hold, axis=1)


def make_far_tasks(n: int, seed: int, min_geo: float = 70.0) -> dict:
    """Cross-room tasks whose shortest path (through the door) is at least min_geo px. Positions are used only to
    choose the tasks; the planner sees the start and goal frames."""
    rng = np.random.default_rng(seed)
    starts, goals = [], []
    while len(starts) < n:
        s = env.sample_free(rng)
        g = env.sample_free(rng, room=1 - env.room_of(s[0]))
        if env.geodesic_dist(s, g) >= min_geo:
            starts.append(s)
            goals.append(g)
    return {"seed": seed, "min_geo": min_geo, "starts": np.array(starts).tolist(), "goals": np.array(goals).tolist(),
            "cross": [True] * n}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--taskset", choices=["test", "tune", "far"], default="test")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--tag", type=str, required=True)
    ap.add_argument("--subgoal", action="store_true", help="plan toward graph waypoints (default: plain latent CEM)")
    ap.add_argument("--knots", type=int, default=5)
    ap.add_argument("--hold", type=int, default=4)
    ap.add_argument("--samples", type=int, default=300)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--elites", type=int, default=30)
    ap.add_argument("--running-w", type=float, default=1.0)
    ap.add_argument("--plan-seed", type=int, default=0)
    ap.add_argument("--bank-episodes", type=int, default=400)
    ap.add_argument("--bank-stride", type=int, default=4)
    ap.add_argument("--knn", type=int, default=8)
    ap.add_argument("--thr-pct", type=float, default=25.0)
    ap.add_argument("--pick", choices=["within", "beyond"], default="within",
                    help="subgoal = farthest path node within one horizon, or first node at/after one horizon")
    ap.add_argument("--train", type=str, default="data/train")
    ap.add_argument("--results", type=str, default="results/horizon")
    args = ap.parse_args()

    torch.set_num_threads(4)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    res = Path(args.results)
    res.mkdir(parents=True, exist_ok=True)
    if args.taskset in ("test", "tune"):
        tasks = load_tasks(res / f"tasks_{args.taskset}.json", args.n, {"test": 20261122, "tune": 777}[args.taskset])
    else:
        p = res / "tasks_far.json"
        if not p.exists():
            p.write_text(json.dumps(make_far_tasks(args.n, 20261123)))
        tasks = json.loads(p.read_text())
    cem = CEMConfig(knots=args.knots, hold=args.hold, samples=args.samples, iters=args.iters, elites=args.elites,
                    running_w=args.running_w)
    cem_str = f"K{cem.knots}xh{cem.hold} S{cem.samples} I{cem.iters} E{cem.elites} rw{cem.running_w} x1"
    method = ("latent_cem_subgoal" + ("" if args.pick == "within" else "_beyond")) if args.subgoal else "latent_cem"
    out_dir = res / "eval" / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    for run in args.runs:
        if dev == "cuda":
            torch.cuda.reset_peak_memory_stats()
        model = load_model(Path(run) / "model.pt", dev)
        a = json.loads((Path(run) / "summary.json").read_text())["args"]
        extra = {}
        if args.subgoal:
            t0 = time.time()
            graph = WaypointGraph(model, args.train, args.bank_episodes, args.bank_stride, args.knn, args.thr_pct, dev)
            extra = {"graph": graph.describe(), "graph_diag": graph.edge_report(args.train),
                     "graph_sec": round(time.time() - t0, 1)}
            cem_str_run = cem_str + f" bank{args.bank_episodes}/{args.bank_stride} knn{args.knn} thr_p{args.thr_pct:g} pick={args.pick}"
            policy = SubgoalCEM(model, tasks, cem, graph, 1, args.plan_seed, dev, args.pick)
        else:
            cem_str_run = cem_str
            policy = LatentCEM(model, tasks, cem, 1, args.plan_seed, dev)
        r = run_episodes(policy, tasks)
        if args.subgoal:
            extra["aim_goal_frac_per_step"] = [round(x[0], 3) for x in policy.log]
        gpu = torch.cuda.max_memory_allocated() / 1e9 if dev == "cuda" else 0.0
        row = {"tag": args.tag, "taskset": args.taskset, "method": method, "run": run, "cem": cem_str_run,
               "lam": a["lam"], "seed": a["seed"], "frameskip": a.get("frameskip", 1), **r}
        append_csv(res / "eval.csv", row)
        (out_dir / f"{args.taskset}__{method}__{Path(run).name}.json").write_text(
            json.dumps({**row, **extra, "gpu_peak_gb": gpu, "host_peak_rss_gb": data_mod.peak_rss_gb()}))
        print(f"{method:18s} {args.taskset:4s} K{args.knots} {run:22s} success {r['success']:.2f} (same "
              f"{r['success_same']:.2f}, cross {r['success_cross']:.2f}) steps {r['mean_steps_success']:.1f} "
              f"final {r['final_dist_px']:.1f}px {r['sec_per_replan'] * 1000:.0f} ms/replan, gpu {gpu:.2f} GB"
              + (f" graph {extra['graph']} {extra['graph_diag']}" if args.subgoal else ""), flush=True)


if __name__ == "__main__":
    main()
