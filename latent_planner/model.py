"""LeWM-style JEPA world model, sized for the browser.

    frame o_t --encoder--> h_t --projector(BN)--> z_t                      (z is what SIGReg regularises)
    (z_t, a_t) --predictor--> z_hat_{t+1}                                   (MLP with a residual connection)
    loss = mean ||z_hat_{t+1} - z_{t+1}||^2  +  lambda * SIGReg(z)

Like LeWM there is no EMA target encoder and no stop-gradient: gradients from the prediction loss flow into the
target embeddings too, and SIGReg alone keeps the embedding from collapsing.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class ModelConfig:
    latent_dim: int = 32
    enc_channels: tuple[int, ...] = (32, 64, 128, 128)
    enc_feat: int = 256
    proj_hidden: int = 512
    act_emb: int = 32
    pred_hidden: int = 256
    pred_layers: int = 2

    def to_dict(self) -> dict:
        d = asdict(self)
        d["enc_channels"] = list(self.enc_channels)
        return d


class Encoder(nn.Module):
    """64x64x3 uint8 frame -> latent z. Four stride-2 convs, no normalisation inside the conv stack."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        layers, c_in = [], 3
        for c in cfg.enc_channels:
            layers += [nn.Conv2d(c_in, c, 4, 2, 1), nn.GELU()]
            c_in = c
        self.convs = nn.Sequential(*layers)
        side = 64 // (2 ** len(cfg.enc_channels))
        self.fc = nn.Linear(c_in * side * side, cfg.enc_feat)
        # Projector as in LeWM: MLP with BatchNorm. SIGReg is applied to its output.
        self.proj = nn.Sequential(
            nn.Linear(cfg.enc_feat, cfg.proj_hidden),
            nn.BatchNorm1d(cfg.proj_hidden),
            nn.GELU(),
            nn.Linear(cfg.proj_hidden, cfg.latent_dim),
        )

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """frames: (N, 64, 64, 3) uint8 (or float in [0,255]) -> (N, D)."""
        x = frames.permute(0, 3, 1, 2).float() * (2.0 / 255.0) - 1.0
        h = self.convs(x).flatten(1)
        return self.proj(self.fc(h))


class Predictor(nn.Module):
    """z_{t+1} = z_t + MLP([z_t, emb(a_t)])."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.act = nn.Sequential(nn.Linear(2, cfg.act_emb), nn.SiLU(), nn.Linear(cfg.act_emb, cfg.act_emb))
        layers, d_in = [], cfg.latent_dim + cfg.act_emb
        for _ in range(cfg.pred_layers):
            layers += [nn.Linear(d_in, cfg.pred_hidden), nn.LayerNorm(cfg.pred_hidden), nn.GELU()]
            d_in = cfg.pred_hidden
        layers.append(nn.Linear(d_in, cfg.latent_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return z + self.net(torch.cat([z, self.act(a)], dim=-1))


class SIGReg(nn.Module):
    """Sketched Isotropic Gaussian Regulariser (LeJEPA, Balestriero & LeCun 2025, arXiv 2511.08544).

    Project the batch of embeddings onto M random unit directions. For each 1-D projection, compare its empirical
    characteristic function with that of N(0, 1) using the Epps-Pulley statistic
        EP = N * integral w(t) |phi_emp(t) - exp(-t^2/2)|^2 dt,   w(t) = exp(-t^2/2),
    integrated on t in [-t_max, t_max] with the trapezoid rule (the integrand is even, so we integrate [0, t_max]
    and double). Average over directions. Minimising it pushes every 1-D marginal, hence the joint, towards an
    isotropic Gaussian, which rules out collapse. Our own implementation; the quadrature setup (17 knots on [0, 3])
    follows LeWM's single-GPU version (MIT, github.com/lucas-maes/le-wm).
    """

    def __init__(self, num_proj: int = 512, knots: int = 17, t_max: float = 3.0):
        super().__init__()
        self.num_proj = num_proj
        t = torch.linspace(0.0, t_max, knots)
        dt = t_max / (knots - 1)
        trap = torch.full((knots,), dt)
        trap[0] = trap[-1] = dt / 2
        window = torch.exp(-0.5 * t**2)
        self.register_buffer("t", t)
        self.register_buffer("target", window)  # characteristic function of N(0,1)
        self.register_buffer("weights", 2.0 * trap * window)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: (..., N, D) -> scalar. Leading dims (e.g. time) are averaged."""
        d = z.shape[-1]
        A = torch.randn(d, self.num_proj, device=z.device, dtype=z.dtype)
        A = A / A.norm(dim=0, keepdim=True)
        x = (z @ A).unsqueeze(-1) * self.t  # (..., N, M, K)
        re = torch.cos(x).mean(dim=-3)
        im = torch.sin(x).mean(dim=-3)
        err = (re - self.target) ** 2 + im**2  # (..., M, K)
        stat = (err @ self.weights) * z.shape[-2]
        return stat.mean()


class WorldModel(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = Encoder(cfg)
        self.predictor = Predictor(cfg)

    def encode(self, frames: torch.Tensor) -> torch.Tensor:
        shp = frames.shape[:-3]
        z = self.encoder(frames.reshape(-1, *frames.shape[-3:]))
        return z.reshape(*shp, -1)

    @torch.no_grad()
    def rollout(self, z0: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """z0 (B, D), actions (B, H, 2) -> predicted latents (B, H, D) for steps 1..H."""
        zs, z = [], z0
        for t in range(actions.shape[1]):
            z = self.predictor(z, actions[:, t])
            zs.append(z)
        return torch.stack(zs, dim=1)


def jepa_loss(model: WorldModel, sigreg: SIGReg, frames: torch.Tensor, actions: torch.Tensor, lam: float,
              rollout_steps: int = 1) -> dict[str, torch.Tensor]:
    """frames (B, T+1, 64, 64, 3) uint8, actions (B, T, 2).

    Prediction loss: teacher-forced one-step predictions for every t, plus (if rollout_steps > 1) open-loop
    rollouts of up to rollout_steps steps from every start, which keeps long planning rollouts on track.
    """
    z = model.encode(frames)  # (B, T+1, D)
    B, T1, D = z.shape
    T = T1 - 1
    pred_losses = []
    zt = z[:, :-1]
    for k in range(rollout_steps):
        n = T - k
        if n <= 0:
            break
        zt = model.predictor(zt[:, :n], actions[:, k : k + n])
        pred_losses.append((zt - z[:, k + 1 : k + 1 + n]).pow(2).mean())
    pred = torch.stack(pred_losses).mean()
    sig = sigreg(z.transpose(0, 1))  # SIGReg over the batch, per time step, averaged over time
    return {"loss": pred + lam * sig, "pred": pred.detach(), "sigreg": sig.detach()}


@torch.no_grad()
def effective_rank(z: torch.Tensor) -> float:
    """Roy & Vetterli (2007) effective rank: exp(entropy of normalised singular values)."""
    z = z.float() - z.float().mean(0, keepdim=True)
    s = torch.linalg.svdvals(z)
    p = s / s.sum().clamp_min(1e-12)
    p = p[p > 0]
    return float(torch.exp(-(p * p.log()).sum()))


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def predictor_macs(cfg: ModelConfig) -> int:
    d_in = cfg.latent_dim + cfg.act_emb
    macs = 2 * cfg.act_emb + cfg.act_emb * cfg.act_emb
    for _ in range(cfg.pred_layers):
        macs += d_in * cfg.pred_hidden
        d_in = cfg.pred_hidden
    return macs + d_in * cfg.latent_dim

