"""Two-room point-mass environment, NumPy reference implementation.

The browser demo has a line-by-line JS port in web/src/env.js. Both versions use only float64 arithmetic in the
same order and the same rasteriser, so a state rendered in Python and in JS gives the same pixels (checked by
tests/test_parity.py). The model only ever sees the 64x64 RGB frame.

Geometry (pixel units, origin top-left, x right, y down):
  - 64x64 arena with a 2 px border wall.
  - Vertical wall at x in [31, 33] with a door at y in [DOOR_Y0, DOOR_Y1].
  - Agent: drawn as a circle of radius AGENT_R, collides as a square of half-size AGENT_R (axis-separated moves),
    which keeps collision handling exact and trivial to port.
  - Action a in [-1, 1]^2 moves the agent by a * SPEED px per step, integrated in SUBSTEPS substeps.
"""

from __future__ import annotations

import numpy as np

SIZE = 64
BORDER = 2
WALL_X0, WALL_X1 = 31.0, 33.0
DOOR_Y0, DOOR_Y1 = 42.0, 56.0
AGENT_R = 3.0
SPEED = 5.0
SUBSTEPS = 5
SS = 4  # supersampling per axis for anti-aliased circle coverage

BG = np.array([18.0, 22.0, 32.0])
WALL = np.array([110.0, 122.0, 148.0])
AGENT = np.array([240.0, 96.0, 72.0])

# Axis-aligned wall rectangles (x0, y0, x1, y1).
WALLS = np.array(
    [
        [0, 0, SIZE, BORDER],  # top
        [0, SIZE - BORDER, SIZE, SIZE],  # bottom
        [0, 0, BORDER, SIZE],  # left
        [SIZE - BORDER, 0, SIZE, SIZE],  # right
        [WALL_X0, 0, WALL_X1, DOOR_Y0],  # middle wall, above the door
        [WALL_X0, DOOR_Y1, WALL_X1, SIZE],  # middle wall, below the door
    ],
    dtype=np.float64,
)


def _overlaps(x: float, y: float) -> int:
    """Index of the first wall the agent box overlaps, or -1."""
    r = AGENT_R
    for i in range(WALLS.shape[0]):
        x0, y0, x1, y1 = WALLS[i]
        if x + r > x0 and x - r < x1 and y + r > y0 and y - r < y1:
            return i
    return -1


def is_free(x: float, y: float) -> bool:
    return _overlaps(x, y) < 0


def step_pos(pos: np.ndarray, action: np.ndarray) -> np.ndarray:
    """Exact deterministic dynamics. pos (2,), action (2,) -> new pos (2,)."""
    x, y = float(pos[0]), float(pos[1])
    ax = min(max(float(action[0]), -1.0), 1.0)
    ay = min(max(float(action[1]), -1.0), 1.0)
    dx = ax * SPEED / SUBSTEPS
    dy = ay * SPEED / SUBSTEPS
    r = AGENT_R
    for _ in range(SUBSTEPS):
        nx = x + dx
        i = _overlaps(nx, y)
        if i >= 0:
            x0, _, x1, _ = WALLS[i]
            nx = x0 - r if dx > 0 else x1 + r
        x = nx
        ny = y + dy
        i = _overlaps(x, ny)
        if i >= 0:
            _, y0, _, y1 = WALLS[i]
            ny = y0 - r if dy > 0 else y1 + r
        y = ny
    return np.array([x, y], dtype=np.float64)


def room_of(x: float) -> int:
    return 0 if x < WALL_X0 else 1


# ---------------------------------------------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------------------------------------------

def _static_layer() -> np.ndarray:
    img = np.empty((SIZE, SIZE, 3), dtype=np.float64)
    img[:] = BG
    for x0, y0, x1, y1 in WALLS:
        img[int(y0) : int(y1), int(x0) : int(x1)] = WALL
    return img


_STATIC = _static_layer()
_STATIC_U8 = np.floor(_STATIC + 0.5).astype(np.uint8)
assert np.array_equal(_STATIC_U8, _STATIC)  # static colours are integers
_SUB = (np.arange(SS, dtype=np.float64) + 0.5) / SS


def render(pos: np.ndarray) -> np.ndarray:
    """Render a state to a (64, 64, 3) uint8 frame."""
    img = _STATIC.copy()
    cx, cy = float(pos[0]), float(pos[1])
    r2 = AGENT_R * AGENT_R
    i0 = max(int(np.floor(cy - AGENT_R)) - 1, 0)
    i1 = min(int(np.floor(cy + AGENT_R)) + 2, SIZE)
    j0 = max(int(np.floor(cx - AGENT_R)) - 1, 0)
    j1 = min(int(np.floor(cx + AGENT_R)) + 2, SIZE)
    for i in range(i0, i1):
        for j in range(j0, j1):
            cnt = 0
            for sy in _SUB:
                py = i + sy - cy
                for sx in _SUB:
                    px = j + sx - cx
                    if px * px + py * py <= r2:
                        cnt += 1
            if cnt:
                c = cnt / (SS * SS)
                img[i, j] = img[i, j] * (1.0 - c) + AGENT * c
    return np.floor(img + 0.5).astype(np.uint8)


def render_batch(pos: np.ndarray) -> np.ndarray:
    """Vectorised render for many states, identical output to render(). pos (N, 2) -> (N, 64, 64, 3) uint8.

    Memory: the output starts as a uint8 copy of the static layer (all static colours are integers, so rounding
    leaves them unchanged) and only the 10x10 window around each agent is blended in float64. Peak extra memory is
    about N * 13 kB, so callers should still render in chunks (see data.py).
    """
    n = pos.shape[0]
    out = np.broadcast_to(_STATIC_U8, (n, SIZE, SIZE, 3)).copy()
    # 10x10 pixel window around each agent covers radius 3 plus margins.
    W = 10
    j0 = np.clip(np.floor(pos[:, 0] - AGENT_R).astype(int) - 1, 0, SIZE - W)
    i0 = np.clip(np.floor(pos[:, 1] - AGENT_R).astype(int) - 1, 0, SIZE - W)
    ii = i0[:, None] + np.arange(W)[None, :]  # (n, W)
    jj = j0[:, None] + np.arange(W)[None, :]
    py = ii[:, :, None] + _SUB[None, None, :] - pos[:, 1, None, None]  # (n, W, SS)
    px = jj[:, :, None] + _SUB[None, None, :] - pos[:, 0, None, None]
    d2 = px[:, None, :, None, :] ** 2 + py[:, :, None, :, None] ** 2  # (n, Wi, Wj, SSy, SSx)
    cnt = (d2 <= AGENT_R * AGENT_R).sum(axis=(3, 4))
    c = (cnt / (SS * SS))[..., None]  # (n, W, W, 1)
    rows = ii[:, :, None]
    cols = jj[:, None, :]
    nidx = np.arange(n)[:, None, None]
    patch = _STATIC[rows, cols]  # (n, W, W, 3) float64
    out[nidx, rows, cols] = np.floor(patch * (1.0 - c) + AGENT * c + 0.5).astype(np.uint8)
    return out


# ---------------------------------------------------------------------------------------------------------------
# Sampling helpers
# ---------------------------------------------------------------------------------------------------------------

def sample_free(rng: np.random.Generator, room: int | None = None) -> np.ndarray:
    while True:
        x = rng.uniform(BORDER + AGENT_R, SIZE - BORDER - AGENT_R)
        y = rng.uniform(BORDER + AGENT_R, SIZE - BORDER - AGENT_R)
        if room is not None and room_of(x) != room:
            continue
        if is_free(x, y):
            return np.array([x, y], dtype=np.float64)


DOOR_CENTER = np.array([(WALL_X0 + WALL_X1) / 2, (DOOR_Y0 + DOOR_Y1) / 2])


def geodesic_dist(a: np.ndarray, b: np.ndarray) -> float:
    """Shortest path length (through the door if rooms differ), used only for task stats."""
    if room_of(a[0]) == room_of(b[0]) or (DOOR_Y0 + AGENT_R <= a[1] <= DOOR_Y1 - AGENT_R and abs(a[0] - 32) < 5):
        return float(np.linalg.norm(a - b))
    return float(np.linalg.norm(a - DOOR_CENTER) + np.linalg.norm(DOOR_CENTER - b))


# ---------------------------------------------------------------------------------------------------------------
# Vectorised dynamics (used by the oracle-CEM baseline, which plans with the true simulator)
# ---------------------------------------------------------------------------------------------------------------

def _first_overlap_batch(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    r = AGENT_R
    hit = ((x[:, None] + r > WALLS[None, :, 0]) & (x[:, None] - r < WALLS[None, :, 2])
           & (y[:, None] + r > WALLS[None, :, 1]) & (y[:, None] - r < WALLS[None, :, 3]))
    return np.where(hit.any(1), hit.argmax(1), -1)


def step_pos_batch(pos: np.ndarray, action: np.ndarray) -> np.ndarray:
    """Same dynamics as step_pos for many states at once. pos (M, 2), action (M, 2) -> (M, 2)."""
    x = pos[:, 0].astype(np.float64).copy()
    y = pos[:, 1].astype(np.float64).copy()
    a = np.clip(action.astype(np.float64), -1.0, 1.0)
    dx = a[:, 0] * SPEED / SUBSTEPS
    dy = a[:, 1] * SPEED / SUBSTEPS
    r = AGENT_R
    for _ in range(SUBSTEPS):
        nx = x + dx
        i = _first_overlap_batch(nx, y)
        m = i >= 0
        if m.any():
            w = WALLS[i[m]]
            nx[m] = np.where(dx[m] > 0, w[:, 0] - r, w[:, 2] + r)
        x = nx
        ny = y + dy
        i = _first_overlap_batch(x, ny)
        m = i >= 0
        if m.any():
            w = WALLS[i[m]]
            ny[m] = np.where(dy[m] > 0, w[:, 1] - r, w[:, 3] + r)
        y = ny
    return np.stack([x, y], axis=1)
