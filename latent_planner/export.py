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


def export(run: Path, final_out: Path, cem: CEMConfig, opset: int = 17, fp16: bool = True,
           scene: str = "tworoom", val_dir: str = "data/push_val") -> dict:
    # Write into a temp dir and only replace final_out once every check has passed, so a failed export never
    # leaves the demo with mismatched files.
    out = final_out.parent / (final_out.name + ".tmp")
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    model = load_model(run / "model.pt", "cpu")
    D = model.cfg.latent_dim
    enc, ro = EncoderExport(model).eval(), RolloutExport(model, cem.hold).eval()
    if scene == "push":
        from . import env_push

        def sample_frames(rng, n):
            return env_push.render_batch(np.stack([env_push.rollout(rng, 10, "push")[0][-1] for _ in range(n)]))
    else:
        def sample_frames(rng, n):
            return env.render_batch(np.stack([env.sample_free(rng) for _ in range(n)]))

    frames = torch.from_numpy(sample_frames(np.random.default_rng(1), 2))
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
    f = sample_frames(rng, 64)
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

    if scene == "push":
        # Visual-only read-out z -> (agent x, y, block x, y), fitted on validation frames; no cost-map grid.
        from . import data as data_mod

        n_ep = min(100, data_mod.num_episodes(val_dir))
        st = np.load(Path(val_dir) / "pos.npy")[:n_ep].reshape(-1, 4)
        with torch.no_grad():
            zg = np.concatenate([enc(torch.from_numpy(data_mod.read_frames(val_dir, i, min(i + 20, n_ep)).reshape(
                -1, 64, 64, 3))).numpy() for i in range(0, n_ep, 20)])
        probe = Ridge(alpha=1e-3).fit(zg, st)
        err = probe.predict(zg) - st
        probe_rmse = float(np.sqrt((err[:, :2] ** 2).sum(1).mean()))
        probe_rmse_block = float(np.sqrt((err[:, 2:] ** 2).sum(1).mean()))
        grid_meta, files = None, ("encoder.onnx", "rollout.onnx")
        env_meta = {"SIZE": env.SIZE, "SPEED": env_push.SPEED, "SUBSTEPS": env.SUBSTEPS, "AGENT_R": env.AGENT_R,
                    "BLOCK_H": env_push.BLOCK_H}
    else:
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
        probe_rmse_block = None
        grid_meta = {"x0": float(lo), "step": 1.0, "n": int(len(xs)), "file": "grid_z.bin"}
        files = ("encoder.onnx", "rollout.onnx", "grid_z.bin")
        env_meta = {"SIZE": env.SIZE, "SPEED": env.SPEED, "SUBSTEPS": env.SUBSTEPS, "AGENT_R": env.AGENT_R}
    sv = np.linalg.svd(zg - zg.mean(0), compute_uv=False)

    summary = json.loads((run / "summary.json").read_text())
    meta = {
        "run": run.name, "scene": scene, "latent_dim": D, "frameskip": model.cfg.frameskip,
        "cem": {"knots": cem.knots, "hold": cem.hold, "samples": cem.samples, "elites": cem.elites,
                "iters": cem.iters, "init_std": cem.init_std, "min_std": cem.min_std, "momentum": cem.momentum,
                "running_w": cem.running_w},
        "env": env_meta,
        "probe": {"W": probe.coef_.round(6).tolist(), "b": probe.intercept_.round(6).tolist(),
                  "grid_rmse_px": probe_rmse, "block_rmse_px": probe_rmse_block},
        "grid": grid_meta,
        "pca_explained_top2": (sv[:2] ** 2 / (sv**2).sum()).round(4).tolist(),
        # predictor-retraining rounds (scripts/push_dagger.py) store only the rollout error, as val_nmse
        "train": {k: summary.get(k) for k in ("val_eff_rank", "probe_rmse_px", "train_seconds")},
        "val_rollout_nmse": [round(x, 5) for x in summary.get("val_rollout_nmse", summary.get("val_nmse", []))],
        "parity": parity,
        "bytes": {f: (out / f).stat().st_size for f in files},
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
    ap.add_argument("--knots", type=int, default=5)
    ap.add_argument("--hold", type=int, default=4)
    ap.add_argument("--scene", choices=["tworoom", "push"], default="tworoom")
    ap.add_argument("--val", type=str, default="data/push_val", help="validation data for the push read-out")
    ap.add_argument("--running-w", type=float, default=1.0,
                    help="running-cost weight stored for the browser planner (push demo: 0, final state only)")
    args = ap.parse_args()
    cem = CEMConfig(knots=args.knots, hold=args.hold, samples=args.samples, iters=args.iters, running_w=args.running_w)
    m = export(Path(args.run), Path(args.out), cem, scene=args.scene, val_dir=args.val)
    print(json.dumps({k: m[k] for k in ("run", "parity", "bytes", "frameskip")}, indent=1),
          f"probe grid rmse {m['probe']['grid_rmse_px']:.2f}px, PCA top-2 explained {m['pca_explained_top2']}")


if __name__ == "__main__":
    main()
