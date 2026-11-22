"""CEM planning in latent space (model-predictive control), batched over many tasks.

The plan is K knots of 2-D actions, each held for `hold` env steps (H = K * hold env steps of lookahead). For a
model with frameskip k, `hold` must be a multiple of k; every predictor step then sees a constant action chunk.

Cost of a sampled plan = ||z_H - z_goal||^2 + running_w * mean_t ||z_t - z_goal||^2, where z_t are the predictor's
imagined latents. Only the encoder output of the goal frame is used: no reward, no positions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .model import WorldModel


@dataclass
class CEMConfig:
    knots: int = 5
    hold: int = 4
    samples: int = 300
    elites: int = 30
    iters: int = 10
    init_std: float = 1.0
    min_std: float = 0.05
    momentum: float = 0.1  # weight of the old mean/std in the update
    running_w: float = 0.0


def expand_plan(knots: torch.Tensor, hold: int, fs: int) -> torch.Tensor:
    """(..., K, 2) knots -> (..., K*hold/fs, 2*fs) predictor-step action chunks."""
    a = knots.repeat_interleave(hold, dim=-2)  # (..., K*hold, 2)
    *lead, n, _ = a.shape
    return a.reshape(*lead, n // fs, 2 * fs)


@torch.no_grad()
def imagine(model: WorldModel, z0: torch.Tensor, knots: torch.Tensor, hold: int) -> torch.Tensor:
    """z0 (M, D), knots (M, K, 2) -> imagined latents (M, K*hold/fs, D)."""
    return model.rollout(z0, expand_plan(knots, hold, model.cfg.frameskip))


@torch.no_grad()
def cem_plan(model: WorldModel, z0: torch.Tensor, zg: torch.Tensor, cfg: CEMConfig,
             init_mean: torch.Tensor | None = None, gen: torch.Generator | None = None):
    """Plan for N tasks at once. z0, zg (N, D). Returns (best knots (N, K, 2), cost of the final mean (N,))."""
    N, D = z0.shape
    K, S, dev = cfg.knots, cfg.samples, z0.device
    assert cfg.hold % model.cfg.frameskip == 0
    mean = torch.zeros(N, K, 2, device=dev) if init_mean is None else init_mean.clone()
    std = torch.full((N, K, 2), cfg.init_std, device=dev)
    z0s = z0[:, None].expand(N, S, D).reshape(N * S, D)
    zgs = zg[:, None, None]  # (N, 1, 1, D)
    for _ in range(cfg.iters):
        eps = torch.randn(N, S, K, 2, device=dev, generator=gen)
        cand = (mean[:, None] + std[:, None] * eps).clamp(-1, 1)
        cand[:, 0] = mean  # keep the current mean as one candidate
        zs = imagine(model, z0s, cand.reshape(N * S, K, 2), cfg.hold).reshape(N, S, -1, D)
        d2 = (zs - zgs).pow(2).sum(-1)  # (N, S, steps)
        cost = d2[..., -1] + cfg.running_w * d2.mean(-1)
        idx = cost.topk(cfg.elites, dim=1, largest=False).indices  # (N, E)
        el = torch.gather(cand, 1, idx[..., None, None].expand(N, cfg.elites, K, 2))
        mean = cfg.momentum * mean + (1 - cfg.momentum) * el.mean(1)
        std = (cfg.momentum * std + (1 - cfg.momentum) * el.std(1)).clamp_min(cfg.min_std)
    zf = imagine(model, z0, mean, cfg.hold)
    d2 = (zf - zg[:, None]).pow(2).sum(-1)
    return mean, d2[:, -1] + cfg.running_w * d2.mean(-1)


def shift_plan(knots: torch.Tensor, n: int) -> torch.Tensor:
    """Warm start for the next MPC step: drop the first n executed knots, pad with zeros."""
    if n <= 0:
        return knots
    return torch.cat([knots[:, n:], torch.zeros_like(knots[:, :n])], dim=1)


# -------------------------------------------------------------------------------------------------------------
# Oracle baseline: the same CEM with the true simulator and true (x, y) distance as the cost.
# -------------------------------------------------------------------------------------------------------------

def cem_plan_oracle(pos: np.ndarray, goal: np.ndarray, cfg: CEMConfig, rng: np.random.Generator,
                    init_mean: np.ndarray | None = None) -> np.ndarray:
    from . import env

    N, K, S = pos.shape[0], cfg.knots, cfg.samples
    mean = np.zeros((N, K, 2)) if init_mean is None else init_mean.copy()
    std = np.full((N, K, 2), cfg.init_std)
    for _ in range(cfg.iters):
        cand = np.clip(mean[:, None] + std[:, None] * rng.standard_normal((N, S, K, 2)), -1, 1)
        cand[:, 0] = mean
        p = np.repeat(pos[:, None], S, axis=1).reshape(-1, 2)
        acts = np.repeat(cand, cfg.hold, axis=2).reshape(N * S, K * cfg.hold, 2)
        run = np.zeros(N * S)
        g = np.repeat(goal[:, None], S, axis=1).reshape(-1, 2)
        for t in range(K * cfg.hold):
            p = env.step_pos_batch(p, acts[:, t])
            run += ((p - g) ** 2).sum(1)
        cost = (((p - g) ** 2).sum(1) + cfg.running_w * run / (K * cfg.hold)).reshape(N, S)
        idx = np.argsort(cost, axis=1)[:, : cfg.elites]
        el = np.take_along_axis(cand, idx[..., None, None], axis=1)
        mean = cfg.momentum * mean + (1 - cfg.momentum) * el.mean(1)
        std = np.maximum(cfg.momentum * std + (1 - cfg.momentum) * el.std(1, ddof=1), cfg.min_std)
    return mean
