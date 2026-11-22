"""Scene 2, "Push": the agent pushes a square block around a walled room. NumPy reference implementation.

Line-by-line JS port in web/src/env_push.js (parity checked by tests/test_js_parity.py). State is
(agent_x, agent_y, block_x, block_y) in pixels; the model only sees the rendered 64x64 frame.

Dynamics are kinematic and axis-separated, so they are exact and easy to port:
  - The agent moves by action * SPEED per step in SUBSTEPS substeps, x then y, and collides as a square of
    half-size AGENT_R with the border walls.
  - If a move would make the agent overlap the block, the block is pushed along that axis so the two just touch;
    a block pressed against a wall stops, and so does the agent. No momentum, no rotation, no friction model.
  - Agent and block never overlap (touching edges are allowed, so the agent can slide along a block's side).
"""

from __future__ import annotations

import numpy as np

from .env import _SUB, AGENT, AGENT_R, BG, BORDER, SIZE, SS, SUBSTEPS, WALL

SPEED = 4.0
BLOCK_H = 4.0  # block half-size: an 8x8 px square
BLOCK = np.array([72.0, 196.0, 176.0])

WALLS = np.array(
    [
        [0, 0, SIZE, BORDER],
        [0, SIZE - BORDER, SIZE, SIZE],
        [0, 0, BORDER, SIZE],
        [SIZE - BORDER, 0, SIZE, SIZE],
    ],
    dtype=np.float64,
)


def _wall_hit(x: float, y: float, h: float) -> int:
    for i in range(WALLS.shape[0]):
        x0, y0, x1, y1 = WALLS[i]
        if x + h > x0 and x - h < x1 and y + h > y0 and y - h < y1:
            return i
    return -1


def _touch(ax: float, ay: float, bx: float, by: float) -> bool:
    """Agent box (half AGENT_R) overlaps block box (half BLOCK_H)? Touching edges don't count."""
    s = AGENT_R + BLOCK_H
    return ax + s > bx and ax - s < bx and ay + s > by and ay - s < by


def step_state(state: np.ndarray, action: np.ndarray) -> np.ndarray:
    ax, ay, bx, by = (float(v) for v in state)
    a0 = min(max(float(action[0]), -1.0), 1.0)
    a1 = min(max(float(action[1]), -1.0), 1.0)
    dx = a0 * SPEED / SUBSTEPS
    dy = a1 * SPEED / SUBSTEPS
    r, h = AGENT_R, BLOCK_H
    for _ in range(SUBSTEPS):
        # x axis
        nx = ax + dx
        i = _wall_hit(nx, ay, r)
        if i >= 0:
            nx = WALLS[i][0] - r if dx > 0 else WALLS[i][2] + r
        if _touch(nx, ay, bx, by):
            nbx = nx + r + h if dx > 0 else nx - r - h
            j = _wall_hit(nbx, by, h)
            if j >= 0:
                nbx = WALLS[j][0] - h if dx > 0 else WALLS[j][2] + h
            nx = nbx - h - r if dx > 0 else nbx + h + r
            bx = nbx
        ax = nx
        # y axis
        ny = ay + dy
        i = _wall_hit(ax, ny, r)
        if i >= 0:
            ny = WALLS[i][1] - r if dy > 0 else WALLS[i][3] + r
        if _touch(ax, ny, bx, by):
            nby = ny + r + h if dy > 0 else ny - r - h
            j = _wall_hit(bx, nby, h)
            if j >= 0:
                nby = WALLS[j][1] - h if dy > 0 else WALLS[j][3] + h
            ny = nby - h - r if dy > 0 else nby + h + r
            by = nby
        ay = ny
    return np.array([ax, ay, bx, by], dtype=np.float64)


# ---------------------------------------------------------------------------------------------------------------
# Rendering: border walls, then the block (exact area coverage), then the agent disc (4x4 supersampled) on top.
# ---------------------------------------------------------------------------------------------------------------

def _static_layer() -> np.ndarray:
    img = np.empty((SIZE, SIZE, 3), dtype=np.float64)
    img[:] = BG
    for x0, y0, x1, y1 in WALLS:
        img[int(y0) : int(y1), int(x0) : int(x1)] = WALL
    return img


_STATIC = _static_layer()


def render(state: np.ndarray) -> np.ndarray:
    img = _STATIC.copy()
    ax, ay, bx, by = (float(v) for v in state)
    h = BLOCK_H
    for i in range(max(int(np.floor(by - h)), 0), min(int(np.floor(by + h)) + 1, SIZE)):
        cy = min(i + 1.0, by + h) - max(float(i), by - h)
        if cy <= 0:
            continue
        for j in range(max(int(np.floor(bx - h)), 0), min(int(np.floor(bx + h)) + 1, SIZE)):
            cx = min(j + 1.0, bx + h) - max(float(j), bx - h)
            if cx <= 0:
                continue
            c = cx * cy
            img[i, j] = img[i, j] * (1.0 - c) + BLOCK * c
    r2 = AGENT_R * AGENT_R
    for i in range(max(int(np.floor(ay - AGENT_R)) - 1, 0), min(int(np.floor(ay + AGENT_R)) + 2, SIZE)):
        for j in range(max(int(np.floor(ax - AGENT_R)) - 1, 0), min(int(np.floor(ax + AGENT_R)) + 2, SIZE)):
            cnt = 0
            for sy in _SUB:
                py = i + sy - ay
                for sx in _SUB:
                    px = j + sx - ax
                    if px * px + py * py <= r2:
                        cnt += 1
            if cnt:
                c = cnt / (SS * SS)
                img[i, j] = img[i, j] * (1.0 - c) + AGENT * c
    return np.floor(img + 0.5).astype(np.uint8)


def render_batch(states: np.ndarray) -> np.ndarray:
    """Same pixels as render() for many states; loops over states but stays light on memory. (N, 4) -> uint8."""
    return np.stack([render(s) for s in states]) if len(states) else np.zeros((0, SIZE, SIZE, 3), np.uint8)


# ---------------------------------------------------------------------------------------------------------------
# Vectorised dynamics for the oracle baseline (same arithmetic as step_state)
# ---------------------------------------------------------------------------------------------------------------

def _wall_hit_batch(x, y, h):
    hit = ((x[:, None] + h > WALLS[None, :, 0]) & (x[:, None] - h < WALLS[None, :, 2])
           & (y[:, None] + h > WALLS[None, :, 1]) & (y[:, None] - h < WALLS[None, :, 3]))
    return np.where(hit.any(1), hit.argmax(1), -1)


def step_state_batch(states: np.ndarray, actions: np.ndarray) -> np.ndarray:
    ax, ay, bx, by = (states[:, k].astype(np.float64).copy() for k in range(4))
    a = np.clip(actions.astype(np.float64), -1.0, 1.0)
    dx = a[:, 0] * SPEED / SUBSTEPS
    dy = a[:, 1] * SPEED / SUBSTEPS
    r, h, s = AGENT_R, BLOCK_H, AGENT_R + BLOCK_H
    for _ in range(SUBSTEPS):
        for axis in (0, 1):
            d = dx if axis == 0 else dy
            p, q = (ax, ay) if axis == 0 else (ay, ax)  # p moves, q is the other coordinate
            bp, bq = (bx, by) if axis == 0 else (by, bx)
            npos = p + d
            i = _wall_hit_batch(npos, q, r) if axis == 0 else _wall_hit_batch(q, npos, r)
            m = i >= 0
            if m.any():
                w = WALLS[i[m]]
                lo, hi = (w[:, 0], w[:, 2]) if axis == 0 else (w[:, 1], w[:, 3])
                npos[m] = np.where(d[m] > 0, lo - r, hi + r)
            t = (npos + s > bp) & (npos - s < bp) & (q + s > bq) & (q - s < bq)
            if t.any():
                nb = np.where(d > 0, npos + r + h, npos - r - h)
                j = _wall_hit_batch(nb, bq, h) if axis == 0 else _wall_hit_batch(bq, nb, h)
                mj = t & (j >= 0)
                if mj.any():
                    w = WALLS[j[mj]]
                    lo, hi = (w[:, 0], w[:, 2]) if axis == 0 else (w[:, 1], w[:, 3])
                    nb[mj] = np.where(d[mj] > 0, lo - h, hi + h)
                npos = np.where(t, np.where(d > 0, nb - h - r, nb + h + r), npos)
                bp = np.where(t, nb, bp)
            if axis == 0:
                ax, bx = npos, bp
            else:
                ay, by = npos, bp
    return np.stack([ax, ay, bx, by], axis=1)


# ---------------------------------------------------------------------------------------------------------------
# Sampling, behaviour policy for data, tasks
# ---------------------------------------------------------------------------------------------------------------

LO_A, HI_A = BORDER + AGENT_R, SIZE - BORDER - AGENT_R
LO_B, HI_B = BORDER + BLOCK_H, SIZE - BORDER - BLOCK_H


def sample_state(rng: np.random.Generator, block_margin: float = 6.0) -> np.ndarray:
    bx, by = rng.uniform(LO_B + block_margin, HI_B - block_margin, 2)
    while True:
        ax, ay = rng.uniform(LO_A, HI_A, 2)
        if max(abs(ax - bx), abs(ay - by)) > AGENT_R + BLOCK_H + 1.0:
            return np.array([ax, ay, bx, by], dtype=np.float64)


def rollout(rng: np.random.Generator, steps: int, policy: str) -> tuple[np.ndarray, np.ndarray]:
    """Behaviour policies: 'wander' (sticky random actions) or 'push' (go behind the block on a random side, then
    push for a while, with noise). Returns states (steps+1, 4) and actions (steps, 2)."""
    S = np.empty((steps + 1, 4))
    A = np.empty((steps, 2))
    s = sample_state(rng)
    S[0] = s
    a = rng.uniform(-1, 1, 2)
    phase, side, left = "approach", None, 0
    for t in range(steps):
        if policy == "wander":
            if rng.random() < 0.2:
                a = rng.uniform(-1, 1, 2)
            act = a
        else:
            if side is None:
                side = [np.array(v, float) for v in ([1, 0], [-1, 0], [0, 1], [0, -1])][rng.integers(4)]
                phase, left = "approach", int(rng.integers(4, 14))
            behind = s[2:] - side * (BLOCK_H + AGENT_R + 1.5)
            if phase == "approach":
                d = behind - s[:2]
                n = np.linalg.norm(d)
                if n < 1.5:
                    phase = "push"
                act = d / max(n, 1e-6) * min(1.0, n / SPEED)
            if phase == "push":
                act = side.copy()
                left -= 1
                if left <= 0:
                    side = None
            act = np.clip(act + rng.normal(0, 0.3, 2), -1, 1)
        A[t] = act
        s = step_state(s, act)
        S[t + 1] = s
    return S, A


def make_tasks(n: int, seed: int, steps: int = 30, min_move: float = 6.0) -> dict:
    """Start states plus goal states reached by running the push policy from the start, so every goal is
    reachable and the goal image shows a plausible agent position next to the moved block."""
    rng = np.random.default_rng(seed)
    starts, goals, moved = [], [], []
    while len(starts) < n:
        S, _ = rollout(rng, steps, "push")
        d = float(np.linalg.norm(S[-1, 2:] - S[0, 2:]))
        if d >= min_move:
            starts.append(S[0])
            goals.append(S[-1])
            moved.append(d)
    return {"seed": seed, "starts": np.array(starts).tolist(), "goals": np.array(goals).tolist(), "moved": moved}
