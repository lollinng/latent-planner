"""Export a trained model for the browser and check the ONNX graphs against PyTorch.

    python -m latent_planner.export --run runs/fs4_lam0.09_s0 --out web/model

Writes:
  encoder.onnx   frames uint8 (N, 64, 64, 3) -> z float32 (N, D)          (BatchNorm in eval mode)
  rollout.onnx   z0 (S, D), knots (S, K, 2) -> zs (S, K*hold/fs, D)        (whole CEM horizon unrolled, one call
                                                                           scores every sample of one iteration)
  meta.json      shapes, CEM defaults, and a visual-only linear probe z -> (x, y) that draws the imagined path.
  grid_z.bin     latents of a 1 px grid of positions, used only to draw the cost landscape.
                 The planner never uses the probe or the grid.
"""

from __future__ import annotations

import argparse
import json
import shutil
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


FP16_TOL = 2e-2  # max abs latent error allowed from fp16 weight storage (latents have std ~1)


def fp16_storage(path: Path, min_elems: int = 256) -> None:
    """Store large float32 weights as float16 followed by a Cast back to float32. Halves the download; ONNX
    Runtime folds the casts when the session is created, so the arithmetic stays float32."""
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    m = onnx.load(str(path))
    g = m.graph
    keep, half, casts = [], [], []
    for init in g.initializer:
        if init.data_type == TensorProto.FLOAT and int(np.prod(init.dims)) >= min_elems:
            h = numpy_helper.from_array(numpy_helper.to_array(init).astype(np.float16), init.name + "__fp16")
            half.append(h)
            casts.append(helper.make_node("Cast", [h.name], [init.name], to=TensorProto.FLOAT))
        else:
            keep.append(init)
    del g.initializer[:]
    g.initializer.extend(keep + half)
    nodes = list(g.node)
    del g.node[:]
    g.node.extend(casts + nodes)
    onnx.checker.check_model(m)
    onnx.save(m, str(path))


def export(run: Path, final_out: Path, cem: CEMConfig, opset: int = 17, fp16: bool = True) -> dict:
    # Write into a temp dir and only replace final_out once every check has passed, so a failed export never
    # leaves the demo with mismatched files.
    out = final_out.parent / (final_out.name + ".tmp")
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
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

    # Parity: onnxruntime vs the float32 torch model on fresh inputs (different batch sizes than the export
    # trace). fp16 weight storage is kept per graph only if it stays within FP16_TOL: in some models the
    # projector's BatchNorm sees features with tiny variance (running var ~1e-7) and amplifies fp16 rounding
    # of the layers before it, so those graphs ship as float32.
    import onnxruntime as ort

    rng = np.random.default_rng(0)
    pos = np.stack([env.sample_free(rng) for _ in range(64)])
    f = env.render_batch(pos)
    with torch.no_grad():
        zt = enc(torch.from_numpy(f)).numpy()
        z0 = torch.from_numpy(zt[:50])
        kn = torch.from_numpy(rng.uniform(-1.2, 1.2, (50, cem.knots, 2)).astype(np.float32))
        rt = ro(z0, kn).numpy()

    def ort_err(path: Path) -> float:
        sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        if path.name.startswith("encoder"):
            return float(np.abs(sess.run(None, {"frames": f})[0] - zt).max())
        return float(np.abs(sess.run(None, {"z0": z0.numpy(), "knots": kn.numpy()})[0] - rt).max())

    parity = {}
    for name in ("encoder", "rollout"):
        path = out / f"{name}.onnx"
        e32 = ort_err(path)
        assert e32 < 1e-3, f"{name}: float32 ONNX differs from torch by {e32}"
        parity[f"{name}_fp32_max_abs"] = e32
        parity[f"{name}_fp16_storage"] = False
        if fp16:
            tmp = out / f"{name}.fp16.onnx"
            shutil.copy(path, tmp)
            fp16_storage(tmp)
            e16 = ort_err(tmp)
            parity[f"{name}_fp16_max_abs"] = e16
            if e16 < FP16_TOL:
                tmp.replace(path)
                parity[f"{name}_fp16_storage"] = True
            else:
                tmp.unlink()

    from sklearn.linear_model import Ridge

    # Visual-only helpers on a dense 1 px grid of agent positions. grid_z.bin holds the latent of every grid
    # position (float32, row-major over y then x, NaN where the agent can't be), so the page can draw the
    # planner's cost landscape ||z(p) - z_goal|| for any goal without running the encoder 3000 times.
    lo, hi = env.BORDER + env.AGENT_R, env.SIZE - env.BORDER - env.AGENT_R
    xs = np.arange(lo, hi + 1e-9, 1.0)
    allp = np.array([[x, y] for y in xs for x in xs])
    free = np.array([env.is_free(x, y) for x, y in allp])
    with torch.no_grad():
        zall = np.concatenate([enc(torch.from_numpy(env.render_batch(allp[i : i + 512]))).numpy()
                               for i in range(0, len(allp), 512)])
    zall[~free] = np.nan
    zall.astype(np.float32).tofile(out / "grid_z.bin")
    grid, zg = allp[free], zall[free]
    probe = Ridge(alpha=1e-3).fit(zg, grid)
    probe_rmse = float(np.sqrt(((probe.predict(zg) - grid) ** 2).sum(1).mean()))
    sv = np.linalg.svd(zg - zg.mean(0), compute_uv=False)

    summary = json.loads((run / "summary.json").read_text())
    meta = {
        "run": run.name, "latent_dim": D, "frameskip": model.cfg.frameskip,
        "cem": {"knots": cem.knots, "hold": cem.hold, "samples": cem.samples, "elites": cem.elites,
                "iters": cem.iters, "init_std": cem.init_std, "min_std": cem.min_std, "momentum": cem.momentum,
                "running_w": cem.running_w},
        "env": {"SIZE": env.SIZE, "SPEED": env.SPEED, "SUBSTEPS": env.SUBSTEPS, "AGENT_R": env.AGENT_R},
        "probe": {"W": probe.coef_.round(6).tolist(), "b": probe.intercept_.round(6).tolist(),
                  "grid_rmse_px": probe_rmse},
        "grid": {"x0": float(lo), "step": 1.0, "n": int(len(xs)), "file": "grid_z.bin"},
        "pca_explained_top2": (sv[:2] ** 2 / (sv**2).sum()).round(4).tolist(),
        "train": {k: summary[k] for k in ("val_eff_rank", "probe_rmse_px", "train_seconds")},
        "val_rollout_nmse": [round(x, 5) for x in summary["val_rollout_nmse"]],
        "parity": parity,
        "bytes": {p.name: p.stat().st_size for p in (out / "encoder.onnx", out / "rollout.onnx", out / "grid_z.bin")},
    }
    (out / "meta.json").write_text(json.dumps(meta))
    if final_out.exists():
        shutil.rmtree(final_out)
    out.rename(final_out)
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
          f"probe grid rmse {m['probe']['grid_rmse_px']:.2f}px, PCA top-2 explained {m['pca_explained_top2']}")


if __name__ == "__main__":
    main()
