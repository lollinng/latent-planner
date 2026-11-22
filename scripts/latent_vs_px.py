"""Diagnostic for results/horizon/SUBGOAL.md: how latent distance grows with true distance (uses positions, so
analysis only). Random pairs of bank states (first 400 training episodes, every 4th frame), CPU only.

    .venv/bin/python scripts/latent_vs_px.py > results/horizon/latent_vs_px.txt
"""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from latent_planner import data as D, env  # noqa: E402
from latent_planner.train import load_model  # noqa: E402

torch.set_num_threads(4)
NE, STRIDE = 400, 4
BINS = [(0, 4), (4, 8), (8, 12), (12, 16), (16, 20), (20, 30), (30, 45), (45, 60), (60, 80), (80, 110)]
pos = np.load("data/train/pos.npy")[:NE, ::STRIDE].reshape(-1, 2).astype(np.float64)
rng = np.random.default_rng(0)
i, j = rng.integers(0, len(pos), (2, 300000))
geo = np.array([env.geodesic_dist(pos[a], pos[b]) for a, b in zip(i, j)])
cross = np.array([env.room_of(pos[a, 0]) != env.room_of(pos[b, 0]) for a, b in zip(i, j)])
for s in (0, 1, 2):
    m = load_model(f"runs/fs4_lam0.09_s{s}/model.pt", "cpu")
    with torch.no_grad():
        z = torch.cat([m.encode(torch.from_numpy(np.ascontiguousarray(D.read_frames("data/train", e, e + 20)[:, ::STRIDE])))
                       for e in range(0, NE, 20)]).reshape(-1, m.cfg.latent_dim).numpy()
    dl = np.linalg.norm(z[i] - z[j], axis=1)
    print(f"fs4_lam0.09_s{s}: median latent distance by shortest-path distance (px), same room | other room")
    for lo, hi in BINS:
        b = (geo >= lo) & (geo < hi)
        same = np.median(dl[b & ~cross]) if (b & ~cross).sum() > 50 else float("nan")
        oth = np.median(dl[b & cross]) if (b & cross).sum() > 50 else float("nan")
        print(f"  [{lo:3d},{hi:3d}) px: {same:6.2f} | {oth:6.2f}   (n {int((b & ~cross).sum())} | {int((b & cross).sum())})")
