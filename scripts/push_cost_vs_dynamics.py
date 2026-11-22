"""Where does the push planner lose: the learned dynamics or the learned cost?

Runs the same CEM on the push tasks three ways:
  oracle  : true simulator, true-state distance cost        (latent_planner/eval_push.py OracleCEM)
  hybrid  : true simulator, latent cost: every imagined state is rendered and encoded, cost = same latent cost
            as the latent planner (||z_H - z_g||^2 + running_w * mean_t ||z_t - z_g||^2, one z per frameskip chunk)
  latent  : learned predictor, latent cost                   (the planner in the demo)
If hybrid is close to oracle, the embedding is a good cost and the dynamics are the weak part; if hybrid is close
to latent, the embedding itself doesn't measure "block at target" well enough.

    python scripts/push_cost_vs_dynamics.py --run runs/push_fs4_s0 --n 100 [--knots 5 --hold 4 --samples 300 --iters 10]

Rendering every candidate needs ~10^5 frames per CEM iteration, so states are rendered with a batched torch
float64 port of env_push.render (checked pixel-exact against it at start-up).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from latent_planner import env_push as P  # noqa: E402
from latent_planner import eval_push as EPU  # noqa: E402
from latent_planner.env import _SUB, AGENT, AGENT_R, SIZE, SS  # noqa: E402
from latent_planner.planner import CEMConfig  # noqa: E402
from latent_planner.train import load_model  # noqa: E402


class TorchRender:
    """Batched, pixel-exact port of env_push.render (same float64 operations in the same order)."""

    def __init__(self, dev):
        self.dev = dev
        self.static = torch.from_numpy(P._STATIC).to(dev)  # (64, 64, 3) float64
        self.idx = torch.arange(SIZE, dtype=torch.float64, device=dev)
        self.sub = torch.from_numpy(_SUB).to(dev)
        self.block = torch.tensor(P.BLOCK, dtype=torch.float64, device=dev)
        self.agent = torch.tensor(AGENT, dtype=torch.float64, device=dev)

    def __call__(self, s: torch.Tensor) -> torch.Tensor:
        """s (M, 4) float64 -> (M, 64, 64, 3) uint8."""
        M, h, j = s.shape[0], P.BLOCK_H, self.idx
        ax, ay, bx, by = (s[:, k : k + 1] for k in range(4))
        cx = torch.minimum(j + 1.0, bx + h) - torch.maximum(j, bx - h)  # (M, 64) per column
        cy = torch.minimum(j + 1.0, by + h) - torch.maximum(j, by - h)  # per row
        cx, cy = cx.clamp_min(0), cy.clamp_min(0)
        c = (cx[:, None, :] * cy[:, :, None])[..., None]  # (M, row, col, 1); cx * cy as in render()
        # render() multiplies cx * cy in that order; IEEE multiplication is commutative, so cy * cx is the same.
        img = self.static.expand(M, SIZE, SIZE, 3)
        img = torch.where(c > 0, img * (1.0 - c) + self.block * c, img)
        # agent: count 4x4 sub-samples inside the disc, py = (i + sy) - ay, px = (j + sx) - ax, over the same
        # pixel window render() loops over (at most W x W pixels from floor(a - R) - 1), then scatter into the frame
        W = int(2 * AGENT_R) + 4
        w = torch.arange(W, device=self.dev)
        r0 = torch.floor(ay - AGENT_R).long() - 1 + w  # (M, W) window rows
        c0 = torch.floor(ax - AGENT_R).long() - 1 + w  # (M, W) window cols
        py = (r0.to(torch.float64)[..., None] + self.sub) - ay[:, :, None]  # (M, W, SS)
        px = (c0.to(torch.float64)[..., None] + self.sub) - ax[:, :, None]
        inside = (px[:, None, :, None, :] * px[:, None, :, None, :] + py[:, :, None, :, None] * py[:, :, None, :, None]
                  <= AGENT_R * AGENT_R)
        cw = inside.sum((-1, -2)).to(torch.float64)  # (M, W rows, W cols)
        ok = ((r0 >= 0) & (r0 < SIZE))[:, :, None] & ((c0 >= 0) & (c0 < SIZE))[:, None, :]
        flat = (r0.clamp(0, SIZE - 1)[:, :, None] * SIZE + c0.clamp(0, SIZE - 1)[:, None, :]).reshape(M, -1)
        cnt = torch.zeros(M, SIZE * SIZE, dtype=torch.float64, device=self.dev)
        cnt.scatter_add_(1, flat, (cw * ok).reshape(M, -1))  # valid window pixels are distinct
        cnt = cnt.reshape(M, SIZE, SIZE, 1)
        ca = cnt / (SS * SS)
        img = torch.where(cnt > 0, img * (1.0 - ca) + self.agent * ca, img)
        return torch.floor(img + 0.5).to(torch.uint8)


def check_render(tr: TorchRender, dev) -> None:
    rng = np.random.default_rng(1)
    s = np.stack([P.sample_state(rng) for _ in range(500)])
    s[:50, :2] = s[:50, 2:] + rng.uniform(-8, 8, (50, 2))  # some overlapping / touching states
    ref = P.render_batch(s)
    got = tr(torch.from_numpy(s).to(dev)).cpu().numpy()
    bad = (ref != got).any(axis=(1, 2, 3)).sum()
    assert bad == 0, f"torch renderer differs from env_push.render on {bad} of {len(s)} states"


class HybridCEM:
    """CEM with the true push simulator and the latent cost."""

    def __init__(self, model, tasks, cfg: CEMConfig, seed: int, dev, chunk: int = 2048):
        self.model, self.cfg, self.dev, self.chunk = model, cfg, dev, chunk
        self.fs = model.cfg.frameskip
        self.render = TorchRender(dev)
        goal = torch.tensor(tasks["goals"], dtype=torch.float64, device=dev)
        with torch.no_grad():
            self.zg = model.encode(self.render(goal))  # (N, D)
        self.rng, self.mean = np.random.default_rng(seed), None

    @torch.no_grad()
    def encode_states(self, s: np.ndarray) -> torch.Tensor:
        st = torch.from_numpy(s).to(self.dev)
        return torch.cat([self.model.encode(self.render(st[i : i + self.chunk])) for i in range(0, len(st), self.chunk)])

    def plan(self, pos: np.ndarray, init_mean):
        cfg, N, K, S = self.cfg, pos.shape[0], self.cfg.knots, self.cfg.samples
        mean = np.zeros((N, K, 2)) if init_mean is None else init_mean.copy()
        std = np.full((N, K, 2), cfg.init_std)
        zg = self.zg[:, None, None]  # (N, 1, 1, D)
        for _ in range(cfg.iters):
            cand = np.clip(mean[:, None] + std[:, None] * self.rng.standard_normal((N, S, K, 2)), -1, 1)
            cand[:, 0] = mean
            p = np.repeat(pos[:, None], S, axis=1).reshape(-1, 4)
            acts = np.repeat(cand, cfg.hold, axis=2).reshape(N * S, K * cfg.hold, 2)
            zs = []
            for t in range(K * cfg.hold):
                p = P.step_state_batch(p, acts[:, t])
                if (t + 1) % self.fs == 0:
                    zs.append(self.encode_states(p))
            zs = torch.stack(zs, 1).reshape(N, S, len(zs), -1)
            d2 = (zs - zg).pow(2).sum(-1)
            cost = (d2[..., -1] + cfg.running_w * d2.mean(-1)).cpu().numpy()
            idx = np.argsort(cost, axis=1)[:, : cfg.elites]
            el = np.take_along_axis(cand, idx[..., None, None], axis=1)
            mean = cfg.momentum * mean + (1 - cfg.momentum) * el.mean(1)
            std = np.maximum(cfg.momentum * std + (1 - cfg.momentum) * el.std(1, ddof=1), cfg.min_std)
        return mean

    def __call__(self, s, active, t):
        init = None
        if self.mean is not None:
            init = np.concatenate([self.mean[:, 1:], np.zeros_like(self.mean[:, :1])], 1)
        self.mean = self.plan(s, init)
        return np.repeat(self.mean[:, :1], self.cfg.hold, axis=1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/push_fs4_s0")
    ap.add_argument("--taskset", choices=["test", "tune"], default="tune")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--knots", type=int, default=5)
    ap.add_argument("--hold", type=int, default=4)
    ap.add_argument("--samples", type=int, default=300)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--out", default="results/push_cost_vs_dynamics.jsonl")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    seed = {"test": 3030, "tune": 4040}[a.taskset]
    tasks = EPU.load_tasks(ROOT / "results" / f"push_tasks_{a.taskset}.json", 100, seed)
    tasks = {k: (v[: a.n] if isinstance(v, list) else v) for k, v in tasks.items()}
    model = load_model(ROOT / a.run / "model.pt", dev)
    cfg = CEMConfig(knots=a.knots, hold=a.hold, samples=a.samples, iters=a.iters, running_w=1.0)
    pol = HybridCEM(model, tasks, cfg, 0, dev)
    check_render(pol.render, dev)
    t0 = time.time()
    r = EPU.run_episodes(pol, tasks)
    row = {"method": "hybrid (true sim, latent cost)", "run": a.run, "taskset": a.taskset, "n": a.n,
           "cem": f"K{a.knots}xh{a.hold} S{a.samples} I{a.iters} rw1.0", "success": r["success"],
           "short": r["success_short"], "long": r["success_long"], "steps": r["mean_steps_success"],
           "final_block_px": r["final_block_dist_px"], "minutes": round((time.time() - t0) / 60, 1)}
    with open(ROOT / a.out, "a") as f:
        f.write(json.dumps(row) + "\n")
    print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
