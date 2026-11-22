"""Offline dataset generation: exploratory trajectories, rendered to 64x64 frames.

No reward, no goals given to the model. Two behaviour policies are mixed so that the data covers both rooms and
crosses the door often enough:
  - "wander": a random walk that keeps its action for a while (resample with prob 0.2 per step).
  - "waypoint": a noisy go-to-random-point controller that routes through the door when the target is in the
    other room. It is only a data collector; the world model never sees its targets.

Usage: python -m latent_planner.data --episodes 2000 --steps 48 --seed 0 --out data/train.npz
"""

from __future__ import annotations

import argparse
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


def generate(episodes: int, steps: int, seed: int, waypoint_frac: float = 0.4) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    P = np.empty((episodes, steps + 1, 2))
    A = np.empty((episodes, steps, 2))
    for e in range(episodes):
        policy = "waypoint" if rng.random() < waypoint_frac else "wander"
        P[e], A[e] = rollout(rng, steps, policy)
    frames = env.render_batch(P.reshape(-1, 2)).reshape(episodes, steps + 1, env.SIZE, env.SIZE, 3)
    return {"frames": frames, "pos": P.astype(np.float32), "actions": A.astype(np.float32)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=2000)
    ap.add_argument("--steps", type=int, default=48)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="data/train.npz")
    args = ap.parse_args()
    t0 = time.time()
    d = generate(args.episodes, args.steps, args.seed)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, **d)
    cross = (np.sign(d["pos"][:, :, 0] - 32).std(axis=1) > 0).mean()
    print(
        f"wrote {args.out}: frames {d['frames'].shape} ({d['frames'].nbytes / 1e9:.2f} GB), "
        f"{cross:.0%} of episodes cross the door, {time.time() - t0:.0f}s"
    )


if __name__ == "__main__":
    main()
