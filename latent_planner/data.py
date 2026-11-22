"""Offline dataset generation: exploratory trajectories, rendered to 64x64 frames.

No reward, no goals given to the model. Two behaviour policies are mixed so that the data covers both rooms and
crosses the door often enough:
  - "wander": a random walk that keeps its action for a while (resample with prob 0.2 per step).
  - "waypoint": a noisy go-to-random-point controller that routes through the door when the target is in the
    other room. It is only a data collector; the world model never sees its targets.

Usage: python -m latent_planner.data --episodes 2000 --steps 48 --seed 0 --out data/train

Frames are streamed to disk in chunks (the first version built all frames in RAM as float64 and was OOM-killed).
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import resource
import time
from pathlib import Path

import numpy as np

from . import env


def wander_actions(rng: np.random.Generator, steps: int) -> np.ndarray:
    acts = np.empty((steps, 2))
    a = rng.uniform(-1, 1, 2)
    for t in range(steps):
        if rng.random() < 0.2:
            a = rng.uniform(-1, 1, 2)
        acts[t] = a
    return acts


def rollout(rng: np.random.Generator, steps: int, policy: str) -> tuple[np.ndarray, np.ndarray]:
    pos = np.empty((steps + 1, 2))
    acts = np.empty((steps, 2))
    p = env.sample_free(rng)
    pos[0] = p
    if policy == "wander":
        acts[:] = wander_actions(rng, steps)
        for t in range(steps):
            p = env.step_pos(p, acts[t])
            pos[t + 1] = p
        return pos, acts

    target = None
    for t in range(steps):
        if target is None or np.linalg.norm(target - p) < 3.0:
            other = rng.random() < 0.6
            room = 1 - env.room_of(p[0]) if other else env.room_of(p[0])
            target = env.sample_free(rng, room)
        goal = target
        if env.room_of(p[0]) != env.room_of(target[0]):
            in_door_band = env.DOOR_Y0 + env.AGENT_R + 0.5 <= p[1] <= env.DOOR_Y1 - env.AGENT_R - 0.5
            goal = env.DOOR_CENTER + np.array([np.sign(target[0] - p[0]) * 6.0, 0.0]) if in_door_band else env.DOOR_CENTER
        d = goal - p
        n = np.linalg.norm(d)
        a = d / max(n, 1e-6) * min(1.0, n / env.SPEED) * rng.uniform(0.6, 1.0)
        a = np.clip(a + rng.normal(0, 0.35, 2), -1, 1)
        acts[t] = a
        p = env.step_pos(p, a)
        pos[t + 1] = p
    return pos, acts


def rss_gb() -> float:
    """Current resident set size of this process in GB (Linux)."""
    with open("/proc/self/statm") as f:
        return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 1e9


def peak_rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6  # ru_maxrss is in kB on Linux


def generate(out_dir: str | Path, episodes: int, steps: int, seed: int, waypoint_frac: float = 0.4,
             chunk: int = 64, max_rss_gb: float = 3.0) -> dict:
    """Generate a dataset straight to disk, memory-bounded.

    Writes out_dir/frames.npy (uint8, (E, T+1, 64, 64, 3)) by appending chunk episodes at a time after the header,
    plus pos.npy (float32, (E, T+1, 2)), actions.npy (float32, (E, T, 2)) and meta.json. RAM use is about
    chunk * (T+1) * 12 kB for frames plus small per-chunk temporaries, independent of the episode count.
    Aborts if RSS exceeds max_rss_gb, so a bug can't take the machine down again.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    frames_path = out_dir / "frames.npy"
    shape = (episodes, steps + 1, env.SIZE, env.SIZE, 3)
    # Plain sequential writes after a .npy header (no memmap), so written pages never count toward our RSS.
    fh = open(frames_path, "wb")
    np.lib.format.write_array_header_1_0(fh, {"descr": "|u1", "fortran_order": False, "shape": shape})
    P = np.empty((episodes, steps + 1, 2))
    A = np.empty((episodes, steps, 2))
    policies = []
    t0 = time.time()
    for e0 in range(0, episodes, chunk):
        e1 = min(e0 + chunk, episodes)
        for e in range(e0, e1):
            policy = "waypoint" if rng.random() < waypoint_frac else "wander"
            policies.append(policy)
            P[e], A[e] = rollout(rng, steps, policy)
        imgs = env.render_batch(P[e0:e1].reshape(-1, 2))
        fh.write(np.ascontiguousarray(imgs).tobytes())
        del imgs
        fh.flush()
        gc.collect()
        r = rss_gb()
        if r > max_rss_gb:
            raise MemoryError(f"RSS {r:.2f} GB > limit {max_rss_gb} GB after {e1} episodes; lower --chunk")
        if (e0 // chunk) % 8 == 0 or e1 == episodes:
            print(f"  {e1}/{episodes} episodes, rss {r:.2f} GB, {time.time() - t0:.0f}s", flush=True)
    fh.close()
    assert np.load(frames_path, mmap_mode="r").shape == shape
    np.save(out_dir / "pos.npy", P.astype(np.float32))
    np.save(out_dir / "actions.npy", A.astype(np.float32))
    cross = float((np.sign(P[:, :, 0] - 32).std(axis=1) > 0).mean())
    meta = {"episodes": episodes, "steps": steps, "seed": seed, "waypoint_frac": waypoint_frac,
            "waypoint_episodes": policies.count("waypoint"), "frac_cross_door": cross,
            "env": {"SIZE": env.SIZE, "SPEED": env.SPEED, "SUBSTEPS": env.SUBSTEPS, "AGENT_R": env.AGENT_R,
                    "DOOR": [env.DOOR_Y0, env.DOOR_Y1], "WALL_X": [env.WALL_X0, env.WALL_X1]},
            "seconds": round(time.time() - t0, 1), "peak_rss_gb": round(peak_rss_gb(), 3)}
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def load(data_dir: str | Path) -> dict[str, np.ndarray]:
    """Open a generated dataset. frames is a read-only memmap (no RAM cost until touched)."""
    d = Path(data_dir)
    return {"frames": np.load(d / "frames.npy", mmap_mode="r"), "pos": np.load(d / "pos.npy"),
            "actions": np.load(d / "actions.npy"), "meta": json.loads((d / "meta.json").read_text())}


def read_frames(data_dir: str | Path, e0: int, e1: int) -> np.ndarray:
    """Read episodes [e0, e1) of frames.npy with a plain file read (no memmap, so nothing stays mapped and
    counted in RSS once the caller drops the array)."""
    path = Path(data_dir) / "frames.npy"
    with open(path, "rb") as fh:
        version = np.lib.format.read_magic(fh)
        read_header = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
        shape, _, _ = read_header(fh)
        per_ep = int(np.prod(shape[1:]))
        e1 = min(e1, shape[0])
        fh.seek(fh.tell() + e0 * per_ep)
        arr = np.fromfile(fh, dtype=np.uint8, count=(e1 - e0) * per_ep)
    return arr.reshape((e1 - e0, *shape[1:]))


def num_episodes(data_dir: str | Path) -> int:
    return int(np.load(Path(data_dir) / "frames.npy", mmap_mode="r").shape[0])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=2000)
    ap.add_argument("--steps", type=int, default=48)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=64, help="episodes rendered per batch")
    ap.add_argument("--max-rss-gb", type=float, default=3.0)
    ap.add_argument("--out", type=str, default="data/train")
    args = ap.parse_args()
    meta = generate(args.out, args.episodes, args.steps, args.seed, chunk=args.chunk, max_rss_gb=args.max_rss_gb)
    gb = args.episodes * (args.steps + 1) * env.SIZE * env.SIZE * 3 / 1e9
    print(f"wrote {args.out}/: {args.episodes} x {args.steps + 1} frames ({gb:.2f} GB on disk), "
          f"{meta['frac_cross_door']:.0%} of episodes cross the door, {meta['seconds']:.0f}s, "
          f"peak RSS {meta['peak_rss_gb']:.2f} GB")


if __name__ == "__main__":
    main()
