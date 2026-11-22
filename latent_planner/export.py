"""Export a trained model for the browser and check the ONNX graphs against PyTorch.

    python -m latent_planner.export --run runs/fs4_lam0.09_s0 --out web/model

Writes:
  encoder.onnx   frames uint8 (N, 64, 64, 3) -> z float32 (N, D)          (BatchNorm in eval mode)
  rollout.onnx   z0 (S, D), knots (S, K, 2) -> zs (S, K*hold/fs, D)        (whole CEM horizon unrolled, one call
                                                                           scores every sample of one iteration)
  meta.json      shapes, CEM defaults, and visual-only helpers: a linear probe z -> (x, y) and a 2-D PCA basis.
                 The planner never uses the probe or PCA; they only draw the imagined path and the latent map.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from . import data as data_mod
from . import env
from .planner import CEMConfig, expand_plan
from .train import load_model


class EncoderExport(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.enc = model.encoder

    def forward(self, frames):
        return self.enc(frames)


class RolloutExport(nn.Module):
    def __init__(self, model, hold: int):
        super().__init__()
        self.pred, self.hold, self.fs = model.predictor, hold, model.cfg.frameskip

    def forward(self, z0, knots):
        acts = expand_plan(knots.clamp(-1, 1), self.hold, self.fs)
        zs, z = [], z0
        for t in range(acts.shape[1]):
            z = self.pred(z, acts[:, t])
            zs.append(z)
        return torch.stack(zs, dim=1)


def export(run: Path, out: Path, cem: CEMConfig, opset: int = 17) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    model = load_model(run / "model.pt", "cpu")
    D = model.cfg.latent_dim
    enc, ro = EncoderExport(model).eval(), RolloutExport(model, cem.hold).eval()
    frames = torch.from_numpy(env.render_batch(np.array([[10.0, 10.0], [50.0, 30.0]])))
    z0 = torch.randn(7, D)
    knots = torch.rand(7, cem.knots, 2) * 2 - 1
    kw = dict(opset_version=opset, dynamo=False, do_constant_folding=True)
    torch.onnx.export(enc, (frames,), out / "encoder.onnx", input_names=["frames"], output_names=["z"],
                      dynamic_axes={"frames": {0: "N"}, "z": {0: "N"}}, **kw)
    torch.onnx.export(ro, (z0, knots), out / "rollout.onnx", input_names=["z0", "knots"], output_names=["zs"],
                      dynamic_axes={"z0": {0: "S"}, "knots": {0: "S"}, "zs": {0: "S"}}, **kw)

    # Parity: onnxruntime vs torch on fresh inputs (different batch sizes than the export trace).
    import onnxruntime as ort

    rng = np.random.default_rng(0)
    pos = np.stack([env.sample_free(rng) for _ in range(64)])
    f = env.render_batch(pos)
    s_enc = ort.InferenceSession(str(out / "encoder.onnx"), providers=["CPUExecutionProvider"])
    s_ro = ort.InferenceSession(str(out / "rollout.onnx"), providers=["CPUExecutionProvider"])
    with torch.no_grad():
        zt = enc(torch.from_numpy(f)).numpy()
        z0 = torch.from_numpy(zt[:50])
        kn = torch.from_numpy(rng.uniform(-1.2, 1.2, (50, cem.knots, 2)).astype(np.float32))
        rt = ro(z0, kn).numpy()
    zo = s_enc.run(None, {"frames": f})[0]
    ro_ = s_ro.run(None, {"z0": z0.numpy(), "knots": kn.numpy()})[0]
    parity = {"encoder_max_abs": float(np.abs(zo - zt).max()), "rollout_max_abs": float(np.abs(ro_ - rt).max())}
    assert parity["encoder_max_abs"] < 1e-3 and parity["rollout_max_abs"] < 1e-3, parity

    # Visual-only helpers, fitted on a dense grid of free positions.
    from sklearn.linear_model import Ridge

    xs = np.arange(env.BORDER + env.AGENT_R, env.SIZE - env.BORDER - env.AGENT_R + 1e-9, 1.0)
    grid = np.array([[x, y] for y in xs for x in xs if env.is_free(x, y)])
    with torch.no_grad():
        zg = np.concatenate([enc(torch.from_numpy(env.render_batch(grid[i : i + 512]))).numpy()
                             for i in range(0, len(grid), 512)])
    probe = Ridge(alpha=1e-3).fit(zg, grid)
    probe_rmse = float(np.sqrt(((probe.predict(zg) - grid) ** 2).sum(1).mean()))
    mu = zg.mean(0)
    _, sv, vt = np.linalg.svd(zg - mu, full_matrices=False)
    pcs = vt[:2]
    pc_xy = (zg - mu) @ pcs.T
    sub = grid[:: max(1, len(grid) // 900)]
    sub_idx = np.array([np.where((grid == p).all(1))[0][0] for p in sub])

    summary = json.loads((run / "summary.json").read_text())
    meta = {
        "run": run.name, "latent_dim": D, "frameskip": model.cfg.frameskip,
        "cem": {"knots": cem.knots, "hold": cem.hold, "samples": cem.samples, "elites": cem.elites,
                "iters": cem.iters, "init_std": cem.init_std, "min_std": cem.min_std, "momentum": cem.momentum,
                "running_w": cem.running_w},
        "env": {"SIZE": env.SIZE, "SPEED": env.SPEED, "SUBSTEPS": env.SUBSTEPS, "AGENT_R": env.AGENT_R},
        "probe": {"W": probe.coef_.round(6).tolist(), "b": probe.intercept_.round(6).tolist(),
                  "grid_rmse_px": probe_rmse},
        "pca": {"mean": mu.round(6).tolist(), "components": pcs.round(6).tolist(),
                "explained": (sv[:2] ** 2 / (sv**2).sum()).round(4).tolist(),
                "grid_xy": grid[sub_idx].round(1).tolist(), "grid_pc": pc_xy[sub_idx].round(4).tolist()},
        "train": {k: summary[k] for k in ("val_eff_rank", "probe_rmse_px", "train_seconds")},
        "val_rollout_nmse": [round(x, 5) for x in summary["val_rollout_nmse"]],
        "parity": parity,
        "bytes": {p.name: p.stat().st_size for p in (out / "encoder.onnx", out / "rollout.onnx")},
    }
    (out / "meta.json").write_text(json.dumps(meta))
    return meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=str, required=True)
    ap.add_argument("--out", type=str, default="web/model")
    ap.add_argument("--samples", type=int, default=300)
    ap.add_argument("--iters", type=int, default=10)
    args = ap.parse_args()
    cem = CEMConfig(samples=args.samples, iters=args.iters, running_w=1.0)
    m = export(Path(args.run), Path(args.out), cem)
    print(json.dumps({k: m[k] for k in ("run", "parity", "bytes", "frameskip")}, indent=1),
          f"probe grid rmse {m['probe']['grid_rmse_px']:.2f}px, PCA explained {m['pca']['explained']}")


if __name__ == "__main__":
    main()
