"""Train the JEPA world model on the offline dataset.

    python -m latent_planner.train --lam 0.09 --seed 0 --out runs/lam0.09_s0

The whole dataset is uploaded to the GPU once as uint8 (1.2 GB for 2000 episodes), in chunks so host RAM stays
small, and batches of short windows are gathered on the GPU. Everything logged goes to <out>/log.csv and
<out>/summary.json; the checkpoint is <out>/model.pt.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from . import data as data_mod
from .model import ModelConfig, SIGReg, WorldModel, count_params, effective_rank, jepa_loss


def to_gpu_uint8(data_dir: str, device: str, chunk: int = 100) -> torch.Tensor:
    """Upload frames.npy to the GPU chunk by chunk with plain file reads, so host RAM peaks at one chunk."""
    n = data_mod.num_episodes(data_dir)
    first = data_mod.read_frames(data_dir, 0, 1)
    out = torch.empty((n, *first.shape[1:]), dtype=torch.uint8, device=device)
    for i in range(0, n, chunk):
        out[i : i + chunk] = torch.from_numpy(data_mod.read_frames(data_dir, i, i + chunk)).to(device)
    return out


def chunk_actions(actions: torch.Tensor, fs: int) -> torch.Tensor:
    """(..., T*fs, 2) per-env-step actions -> (..., T, 2*fs) action chunks."""
    *lead, n, _ = actions.shape
    return actions.reshape(*lead, n // fs, 2 * fs)


class WindowSampler:
    """Random windows of T predictor steps (T*fs env steps) from GPU-resident episodes."""

    def __init__(self, frames: torch.Tensor, actions: torch.Tensor, pos: torch.Tensor, T: int, fs: int = 1):
        self.frames, self.actions, self.pos, self.T, self.fs = frames, actions, pos, T, fs
        self.E, self.L = frames.shape[0], frames.shape[1]  # L = steps + 1
        assert T * fs < self.L, "window longer than an episode"

    def sample(self, B: int, gen: torch.Generator):
        dev = self.frames.device
        e = torch.randint(0, self.E, (B,), device=dev, generator=gen)
        t0 = torch.randint(0, self.L - self.T * self.fs, (B,), device=dev, generator=gen)
        tt = t0[:, None] + torch.arange(self.T + 1, device=dev)[None] * self.fs
        ta = t0[:, None] + torch.arange(self.T * self.fs, device=dev)[None]
        f = self.frames[e[:, None], tt]  # (B, T+1, 64, 64, 3)
        a = chunk_actions(self.actions[e[:, None], ta], self.fs)  # (B, T, 2*fs)
        p = self.pos[e[:, None], tt]
        return f, a, p


@torch.no_grad()
def evaluate(model: WorldModel, frames: torch.Tensor, actions: torch.Tensor, env_horizon: int = 16,
             n_ep: int = 200) -> dict:
    """Open-loop rollout error on validation episodes: encode frame 0, roll the predictor through the logged
    actions, compare with the encoding of each true frame. Reported normalised by the latent variance, so
    1.0 means "no better than predicting the mean embedding". Entry h of the list is predictor step h+1,
    i.e. env step (h+1)*frameskip."""
    model.eval()
    fs = model.cfg.frameskip
    H = env_horizon // fs
    z = torch.cat([model.encode(frames[i : min(i + 25, n_ep), : H * fs + 1 : fs]) for i in range(0, n_ep, 25)])
    zhat = model.rollout(z[:, 0], chunk_actions(actions[:n_ep, : H * fs], fs))  # z: (N, H+1, D)
    var = z.reshape(-1, z.shape[-1]).var(0).sum()
    err = (zhat - z[:, 1:]).pow(2).sum(-1).mean(0) / var  # (H,)
    zall = z.reshape(-1, z.shape[-1])
    out = {"val_rollout_nmse": err.cpu().tolist(), "val_eff_rank": effective_rank(zall),
           "val_z_std": float(zall.std(0).mean())}
    model.train()
    return out


@torch.no_grad()
def position_probe(model: WorldModel, train_dir: str, val_dir: str, device: str) -> dict:
    """Linear probe z -> (x, y), fitted on train frames, R^2 and RMSE (px) on val. Diagnostic only."""
    from sklearn.linear_model import Ridge

    model.eval()

    def embed(d, n):
        zs = [model.encode(torch.from_numpy(data_mod.read_frames(d, i, min(i + 20, n))).to(device)).cpu()
              for i in range(0, n, 20)]
        return torch.cat(zs).reshape(-1, model.cfg.latent_dim).numpy()

    ntr, nva = min(500, data_mod.num_episodes(train_dir)), data_mod.num_episodes(val_dir)
    ztr, zva = embed(train_dir, ntr), embed(val_dir, nva)
    ptr = np.load(Path(train_dir) / "pos.npy")[:ntr]
    ptr = ptr.reshape(-1, ptr.shape[-1])  # (x, y) for two-room, (agent x, y, block x, y) for push
    pva = np.load(Path(val_dir) / "pos.npy")[:nva]
    pva = pva.reshape(-1, pva.shape[-1])
    reg = Ridge(alpha=1e-3).fit(ztr, ptr)
    pred = reg.predict(zva)
    r2 = 1 - ((pred - pva) ** 2).sum(0) / ((pva - pva.mean(0)) ** 2).sum(0)
    model.train()
    out = {"probe_r2_x": float(r2[0]), "probe_r2_y": float(r2[1]),
           "probe_rmse_px": float(np.sqrt(((pred[:, :2] - pva[:, :2]) ** 2).sum(1).mean()))}
    if pva.shape[1] == 4:  # push scene: also how well the block position can be read out
        out["probe_rmse_block_px"] = float(np.sqrt(((pred[:, 2:] - pva[:, 2:]) ** 2).sum(1).mean()))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lam", type=float, default=0.09, help="SIGReg weight (LeWM default 0.09)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--T", type=int, default=8, help="window length in predictor steps")
    ap.add_argument("--rollout", type=int, default=8, help="open-loop rollout steps in the prediction loss")
    ap.add_argument("--frameskip", type=int, default=1, help="env steps per predictor step (LeWM uses 5)")
    ap.add_argument("--pred-hidden", type=int, default=256)
    ap.add_argument("--pred-layers", type=int, default=2, help="hidden layers in the predictor MLP")
    ap.add_argument("--pred-dropout", type=float, default=0.0, help="dropout in the predictor MLP (training only)")
    ap.add_argument("--proj-norm", choices=["bn", "ln"], default="bn", help="projector normalisation")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--latent-dim", type=int, default=32)
    ap.add_argument("--num-proj", type=int, default=512)
    ap.add_argument("--train", type=str, default="data/train")
    ap.add_argument("--val", type=str, default="data/val")
    ap.add_argument("--log-every", type=int, default=200)
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(4)  # other jobs share this laptop
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    t_load = time.time()
    frames = to_gpu_uint8(args.train, dev)
    vframes = to_gpu_uint8(args.val, dev)
    actions = torch.from_numpy(np.load(Path(args.train) / "actions.npy")).to(dev)
    vactions = torch.from_numpy(np.load(Path(args.val) / "actions.npy")).to(dev)
    pos = torch.from_numpy(np.load(Path(args.train) / "pos.npy")).to(dev)
    sampler = WindowSampler(frames, actions, pos, args.T, args.frameskip)
    print(f"data on {dev}: {tuple(frames.shape)} in {time.time() - t_load:.1f}s, "
          f"host peak RSS {data_mod.peak_rss_gb():.2f} GB", flush=True)

    cfg = ModelConfig(latent_dim=args.latent_dim, frameskip=args.frameskip, pred_hidden=args.pred_hidden,
                      pred_layers=args.pred_layers, proj_norm=args.proj_norm,
                      pred_dropout=args.pred_dropout)
    model = WorldModel(cfg).to(dev)
    sigreg = SIGReg(num_proj=args.num_proj).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    warm = 200
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(s, args.steps) / args.steps)))
    gen = torch.Generator(device=dev)
    gen.manual_seed(args.seed)
    print(f"params: encoder {count_params(model.encoder):,}, predictor {count_params(model.predictor):,}", flush=True)

    fields = ["step", "loss", "pred", "sigreg", "lr", "train_eff_rank", "val_eff_rank", "val_z_std",
              "val_nmse_m1", "val_nmse_e4", "val_nmse_e8", "val_nmse_e16", "sec"]
    logf = open(out / "log.csv", "w", newline="")
    log = csv.DictWriter(logf, fieldnames=fields)
    log.writeheader()
    t0 = time.time()
    acc = {"loss": 0.0, "pred": 0.0, "sigreg": 0.0, "n": 0}
    for step in range(1, args.steps + 1):
        f, a, _ = sampler.sample(args.batch, gen)
        out_d = jepa_loss(model, sigreg, f, a, args.lam, rollout_steps=args.rollout)
        opt.zero_grad(set_to_none=True)
        out_d["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        for k in ("loss", "pred", "sigreg"):
            acc[k] += float(out_d[k])
        acc["n"] += 1
        if step % args.log_every == 0 or step == args.steps:
            with torch.no_grad():
                model.eval()
                ztr = model.encode(f[:, 0])
                model.train()
            ev = evaluate(model, vframes, vactions, env_horizon=16)
            nm = ev["val_rollout_nmse"]
            fs = args.frameskip
            row = {"step": step, "loss": acc["loss"] / acc["n"], "pred": acc["pred"] / acc["n"],
                   "sigreg": acc["sigreg"] / acc["n"], "lr": sched.get_last_lr()[0],
                   "train_eff_rank": effective_rank(ztr), "val_eff_rank": ev["val_eff_rank"],
                   "val_z_std": ev["val_z_std"], "val_nmse_m1": nm[0], "val_nmse_e4": nm[4 // fs - 1],
                   "val_nmse_e8": nm[8 // fs - 1], "val_nmse_e16": nm[16 // fs - 1], "sec": time.time() - t0}
            log.writerow(row)
            logf.flush()
            print(" ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()), flush=True)
            acc = {"loss": 0.0, "pred": 0.0, "sigreg": 0.0, "n": 0}
    logf.close()

    torch.save({"cfg": cfg.to_dict(), "state_dict": model.state_dict(), "args": vars(args)}, out / "model.pt")
    ev = evaluate(model, vframes, vactions, env_horizon=16)
    del frames, vframes
    torch.cuda.empty_cache()
    summary = {"args": vars(args), "cfg": cfg.to_dict(), **ev, **position_probe(model, args.train, args.val, dev),
               "train_seconds": time.time() - t0, "host_peak_rss_gb": data_mod.peak_rss_gb(),
               "gpu_peak_mem_gb": torch.cuda.max_memory_allocated() / 1e9 if dev == "cuda" else 0.0,
               "device": torch.cuda.get_device_name(0) if dev == "cuda" else "cpu", "torch": torch.__version__}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k not in ("args", "cfg")}, indent=1))


def load_model(path: str | Path, device: str = "cpu") -> WorldModel:
    ck = torch.load(path, map_location=device, weights_only=False)
    c = ck["cfg"]
    c["enc_channels"] = tuple(c["enc_channels"])
    m = WorldModel(ModelConfig(**c)).to(device)
    m.load_state_dict(ck["state_dict"])
    return m.eval()


if __name__ == "__main__":
    main()
