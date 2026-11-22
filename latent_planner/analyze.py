"""Per-run probes that back the demo's claims with numbers. Appends rows to results/analysis.csv.

    python -m latent_planner.analyze --runs runs/lam0.09_s0 ...

surprise_auroc   Can the prediction error tell a real transition from a teleport? For validation windows, the
                 error ||z_hat_{t+k} - z_{t+k}||^2 of the real next frame is compared with the error against a
                 frame from a random other episode/time. AUROC 1.0 = perfectly separable.
spearman_geo     Spearman correlation between latent distance and geodesic (through-the-door) distance over
spearman_euc     random pairs of free positions, and the same against straight-line distance.
basin_px         Median over 50 goals of the radius (px, 2 px bins) where the mean latent distance to the goal first
                 reaches 90% of its far-field (> 30 px) median: how far the planner's cost "sees".
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from . import data as data_mod
from . import env
from .train import chunk_actions, load_model


@torch.no_grad()
def encode_np(model, frames: np.ndarray, bs: int = 512) -> np.ndarray:
    dev = next(model.parameters()).device
    return np.concatenate([model.encode(torch.from_numpy(frames[i : i + bs]).to(dev)).cpu().numpy()
                           for i in range(0, len(frames), bs)])


@torch.no_grad()
def surprise_auroc(model, val_dir: str, n_ep: int = 200, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    fs = model.cfg.frameskip
    dev = next(model.parameters()).device
    frames = data_mod.read_frames(val_dir, 0, n_ep)  # (E, L, 64, 64, 3), 23 MB per 100 episodes
    acts = np.load(Path(val_dir) / "actions.npy")[:n_ep]
    E, L = frames.shape[:2]
    t = rng.integers(0, L - fs, E)
    z_t = encode_np(model, frames[np.arange(E), t])
    z_next = encode_np(model, frames[np.arange(E), t + fs])
    a = np.stack([acts[e, t[e] : t[e] + fs] for e in range(E)])  # (E, fs, 2)
    z_hat = model.predictor(torch.from_numpy(z_t).to(dev),
                            chunk_actions(torch.from_numpy(a).to(dev), fs)[:, 0]).cpu().numpy()
    # teleport: the "next frame" comes from a random other episode and time
    e2 = (np.arange(E) + rng.integers(1, E, E)) % E
    z_tel = encode_np(model, frames[e2, rng.integers(0, L, E)])
    err_real = ((z_hat - z_next) ** 2).mean(1)
    err_tel = ((z_hat - z_tel) ** 2).mean(1)
    y = np.r_[np.zeros(E), np.ones(E)]
    return {"surprise_auroc": float(roc_auc_score(y, np.r_[err_real, err_tel])),
            "surprise_real_median": float(np.median(err_real)), "surprise_teleport_median": float(np.median(err_tel))}


@torch.no_grad()
def latent_geometry(model, n_pairs: int = 4000, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    lo, hi = env.BORDER + env.AGENT_R, env.SIZE - env.BORDER - env.AGENT_R
    xs = np.arange(lo, hi + 1e-9, 1.0)
    grid = np.array([[x, y] for y in xs for x in xs if env.is_free(x, y)])
    z = np.concatenate([encode_np(model, env.render_batch(grid[i : i + 512])) for i in range(0, len(grid), 512)])
    a, b = rng.integers(0, len(grid), n_pairs), rng.integers(0, len(grid), n_pairs)
    dl = np.linalg.norm(z[a] - z[b], axis=1)
    de = np.linalg.norm(grid[a] - grid[b], axis=1)
    dg = np.array([env.geodesic_dist(grid[i], grid[j]) for i, j in zip(a, b)])
    # basin radius around 50 random goals
    radii = []
    for g in rng.integers(0, len(grid), 50):
        d = np.linalg.norm(z - z[g], axis=1)
        r = np.linalg.norm(grid - grid[g], axis=1)
        far = np.median(d[r > 30])
        edges = np.arange(0, 30, 2.0)
        means = [d[(r >= e) & (r < e + 2)].mean() for e in edges if ((r >= e) & (r < e + 2)).any()]
        hit = [e for e, m in zip(edges, means) if m >= 0.9 * far]
        radii.append(float(hit[0]) if hit else 30.0)
    return {"spearman_geo": float(spearmanr(dl, dg)[0]), "spearman_euc": float(spearmanr(dl, de)[0]),
            "basin_px": float(np.median(radii))}


FIELDS = ["run", "lam", "seed", "frameskip", "surprise_auroc", "surprise_real_median", "surprise_teleport_median",
          "spearman_geo", "spearman_euc", "basin_px", "val_eff_rank", "val_nmse_m1", "val_nmse_e16",
          "probe_rmse_px"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--val", default="data/val")
    ap.add_argument("--out", default="results/analysis.csv")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    new = not out.exists()
    with open(out, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        if new:
            w.writeheader()
        for run in args.runs:
            model = load_model(Path(run) / "model.pt", dev)
            s = json.loads((Path(run) / "summary.json").read_text())
            fs = s["args"].get("frameskip", 1)
            n = s["val_rollout_nmse"]
            row = {"run": Path(run).name, "lam": s["args"]["lam"], "seed": s["args"]["seed"], "frameskip": fs,
                   "val_eff_rank": s["val_eff_rank"], "val_nmse_m1": n[0], "val_nmse_e16": n[16 // fs - 1],
                   "probe_rmse_px": s["probe_rmse_px"], **surprise_auroc(model, args.val), **latent_geometry(model)}
            w.writerow(row)
            f.flush()
            print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)


if __name__ == "__main__":
    main()
