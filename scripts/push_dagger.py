"""Push: keep the encoder (its latent distance is a very good cost: 86% with true dynamics) and retrain only the
predictor, adding data from the planner's own attempts (DAgger-style data aggregation).

Why: with the learned predictor the planner reaches 32% on the tuning tasks, and more search doesn't help: CEM
finds plans that look good only inside the model, in states and actions the training data barely covers. Running
the planner and training on what really happened next targets exactly those errors. No rewards or positions are
used for training: the new data is frames and actions, encoded by the frozen encoder.

Each round: collect `--collect` episodes (48 steps) with the current model's planner on fresh random tasks
(not the tuning/test seeds), with Gaussian action noise; then train the predictor on all latents so far. Round 0
trains on the original data only (the control for fine-tuning without new data).

    python scripts/push_dagger.py --run runs/push_fs4_s0 --out runs/push_dagger_s0 --rounds 3

Data is collected with the final-state cost (`--running-w 0`, the demo's planner) by default; every round is
evaluated on the tuning tasks with both costs.

Writes <out>/round<k>/model.pt (+ summary.json) loadable by eval_push / export, and <out>/log.jsonl.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from latent_planner import env_push as P  # noqa: E402
from latent_planner import eval_push as EPU  # noqa: E402
from latent_planner.data import num_episodes, read_frames  # noqa: E402
from latent_planner.eval import LatentCEM  # noqa: E402
from latent_planner.planner import CEMConfig  # noqa: E402
from latent_planner.train import chunk_actions, load_model  # noqa: E402
from push_cost_vs_dynamics import TorchRender  # noqa: E402


@torch.no_grad()
def encode_dataset(model, data_dir: Path, dev, chunk: int = 50):
    """Frozen-encoder latents (E, L, D) and actions (E, L-1, 2) of a stored dataset, read in chunks."""
    E = num_episodes(data_dir)
    zs = []
    for e0 in range(0, E, chunk):
        f = torch.from_numpy(read_frames(data_dir, e0, min(e0 + chunk, E))).to(dev)
        zs.append(model.encode(f))
    acts = torch.from_numpy(np.load(data_dir / "actions.npy")).to(dev)
    return torch.cat(zs), acts


@torch.no_grad()
def collect(model, n: int, seed: int, noise: float, cfg: CEMConfig, dev, render: TorchRender, steps: int = 48,
            batch: int = 200):
    """Run the latent planner on n fresh tasks for `steps` steps with action noise. Returns latents of the visited
    frames (n, steps+1, D), executed actions (n, steps, 2) and the fraction of tasks reached at some step."""
    tasks = P.make_tasks(n, seed)
    rng = np.random.default_rng(seed)
    Z, A, reached = [], [], []
    for b0 in range(0, n, batch):
        tb = {k: (v[b0 : b0 + batch] if isinstance(v, list) else v) for k, v in tasks.items()}
        pol = LatentCEM(model, tb, cfg, 1, seed + b0, dev, render_batch=lambda s: render(torch.from_numpy(s).to(dev)).cpu().numpy())
        s = np.array(tb["starts"], dtype=np.float64)
        goal = np.array(tb["goals"], dtype=np.float64)
        zs = [model.encode(render(torch.from_numpy(s).to(dev)))]
        acts, done = [], np.zeros(len(s), bool)
        t = 0
        while t < steps:
            plan = pol(s, ~done, t)  # (N, hold, 2)
            for j in range(plan.shape[1]):
                if t >= steps:
                    break
                a = np.clip(plan[:, j] + noise * rng.standard_normal(plan[:, j].shape), -1, 1)
                s = P.step_state_batch(s, a)
                acts.append(a)
                zs.append(model.encode(render(torch.from_numpy(s).to(dev))))
                done |= np.linalg.norm(s[:, 2:] - goal[:, 2:], axis=1) <= EPU.SUCCESS_PX
                t += 1
        Z.append(torch.stack(zs, 1))
        A.append(torch.from_numpy(np.stack(acts, 1)).float().to(dev))
        reached.append(done)
    return torch.cat(Z), torch.cat(A), float(np.concatenate(reached).mean())


def train_predictor(pred, datasets, fs: int, steps: int, T: int, rollout: int, lr: float, seed: int, dev,
                    batch: int = 256):
    """Prediction loss of latent_planner.model.jepa_loss (teacher-forced + open-loop rollouts) on fixed latents."""
    gen = torch.Generator(device=dev)
    gen.manual_seed(seed)
    opt = torch.optim.AdamW(pred.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    sizes = torch.tensor([z.shape[0] for z, _ in datasets], dtype=torch.float, device=dev)
    pred.train()
    for it in range(steps):
        # episodes sampled uniformly over all datasets
        which = torch.multinomial(sizes, batch, replacement=True, generator=gen)
        losses = []
        for d, (Z, A) in enumerate(datasets):
            m = int((which == d).sum())
            if m == 0:
                continue
            L = Z.shape[1]
            e = torch.randint(0, Z.shape[0], (m,), device=dev, generator=gen)
            t0 = torch.randint(0, L - T * fs, (m,), device=dev, generator=gen)
            z = Z[e[:, None], t0[:, None] + torch.arange(T + 1, device=dev)[None] * fs]  # (m, T+1, D)
            a = chunk_actions(A[e[:, None], t0[:, None] + torch.arange(T * fs, device=dev)[None]], fs)
            zt, pl = z[:, :-1], []
            for k in range(rollout):
                n = T - k
                if n <= 0:
                    break
                zt = pred(zt[:, :n], a[:, k : k + n])
                pl.append((zt - z[:, k + 1 : k + 1 + n]).pow(2).mean())
            losses.append(torch.stack(pl).mean() * m)
        loss = torch.stack(losses).sum() / batch
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
    pred.eval()
    return loss.item()


@torch.no_grad()
def val_rollout_nmse(pred, Zv, Av, fs: int, H: int = 4):
    z0, a = Zv[:, 0], chunk_actions(Av[:, : H * fs], fs)
    zs, z = [], z0
    for t in range(H):
        z = pred(z, a[:, t])
        zs.append(z)
    zt = Zv[:, fs : H * fs + 1 : fs]
    var = Zv.reshape(-1, Zv.shape[-1]).var(0).sum()
    return ((torch.stack(zs, 1) - zt).pow(2).sum(-1).mean(0) / var).cpu().tolist()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/push_fs4_s0")
    ap.add_argument("--out", default="runs/push_dagger_s0")
    ap.add_argument("--train", default="data/push_train")
    ap.add_argument("--val", default="data/push_val")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--collect", type=int, default=2000)
    ap.add_argument("--noise", type=float, default=0.3)
    ap.add_argument("--train-steps", type=int, default=6000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--rollout", type=int, default=8, help="open-loop steps in the prediction loss (8 = training default)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-n", type=int, default=100)
    ap.add_argument("--running-w", type=float, default=0.0,
                    help="planner cost used to collect data (0 = final-state cost, as in the demo); "
                         "every round is evaluated with both 0 and 1")
    a = ap.parse_args()
    dev = "cuda"
    torch.manual_seed(a.seed)
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    model = load_model(ROOT / a.run / "model.pt", dev)
    for p in model.encoder.parameters():
        p.requires_grad_(False)
    fs = model.cfg.frameskip
    cfg = CEMConfig(knots=5, hold=4, samples=300, iters=10, running_w=a.running_w)
    cfgs = {rw: CEMConfig(knots=5, hold=4, samples=300, iters=10, running_w=rw) for rw in (0.0, 1.0)}
    render = TorchRender(dev)
    datasets = [encode_dataset(model, ROOT / a.train, dev)]
    Zv, Av = encode_dataset(model, ROOT / a.val, dev)
    tune = EPU.load_tasks(ROOT / "results" / "push_tasks_tune.json", 100, 4040)
    tune = {k: (v[: a.eval_n] if isinstance(v, list) else v) for k, v in tune.items()}
    base = json.loads((ROOT / a.run / "summary.json").read_text())["args"]

    def evaluate():
        """Tuning-task success with the final-state cost (rw0) and the running cost (rw1)."""
        res = {}
        for rw, c in cfgs.items():
            r = EPU.run_episodes(LatentCEM(model, tune, c, 1, 0, dev, render_batch=P.render_batch), tune)
            k = f"rw{rw:g}"
            res.update({f"tune_{k}": r["success"], f"short_{k}": r["success_short"], f"long_{k}": r["success_long"]})
        return res

    log = open(out / "log.jsonl", "a")
    row = {"round": "start", **evaluate(), "val_nmse": val_rollout_nmse(model.predictor, Zv, Av, fs)}
    print(json.dumps(row), flush=True)
    log.write(json.dumps(row) + "\n")
    for k in range(a.rounds + 1):
        t0 = time.time()
        info = {}
        if k > 0:  # collect with the current predictor, then add
            Z, A, hit = collect(model, a.collect, 10_000 + 100 * a.seed + k, a.noise, cfg, dev, render)
            datasets.append((Z, A))
            info = {"collected": a.collect, "collect_reached": hit}
        loss = train_predictor(model.predictor, datasets, fs, a.train_steps, 8, a.rollout, a.lr, a.seed * 100 + k, dev)
        row = {"round": k, **info, "episodes": sum(z.shape[0] for z, _ in datasets), "train_loss": loss,
               "val_nmse": val_rollout_nmse(model.predictor, Zv, Av, fs), **evaluate(),
               "minutes": round((time.time() - t0) / 60, 1)}
        print(json.dumps(row), flush=True)
        log.write(json.dumps(row) + "\n")
        log.flush()
        rd = out / f"round{k}"
        rd.mkdir(exist_ok=True)
        torch.save({"cfg": model.cfg.to_dict(), "state_dict": model.state_dict(),
                    "args": {**base, "dagger": vars(a), "round": k}}, rd / "model.pt")
        (rd / "summary.json").write_text(json.dumps({"args": {**base, "dagger_round": k}, **row}, indent=2))


if __name__ == "__main__":
    main()
