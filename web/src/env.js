// Two-room point-mass environment: line-by-line port of latent_planner/env.py.
// Same float64 arithmetic in the same order, so positions and rendered pixels match the Python version exactly
// (checked by tests/test_js_parity.py). The model only ever sees the 64x64 RGB frame this file renders.

export const SIZE = 64;
export const BORDER = 2;
export const WALL_X0 = 31.0, WALL_X1 = 33.0;
export const DOOR_Y0 = 42.0, DOOR_Y1 = 56.0;
export const AGENT_R = 3.0;
export const SPEED = 5.0;
export const SUBSTEPS = 5;
const SS = 4;

const BG = [18.0, 22.0, 32.0];
const WALL = [110.0, 122.0, 148.0];
const AGENT = [240.0, 96.0, 72.0];

export const WALLS = [
  [0, 0, SIZE, BORDER],
  [0, SIZE - BORDER, SIZE, SIZE],
  [0, 0, BORDER, SIZE],
  [SIZE - BORDER, 0, SIZE, SIZE],
  [WALL_X0, 0, WALL_X1, DOOR_Y0],
  [WALL_X0, DOOR_Y1, WALL_X1, SIZE],
];

function overlaps(x, y) {
  const r = AGENT_R;
  for (let i = 0; i < WALLS.length; i++) {
    const [x0, y0, x1, y1] = WALLS[i];
    if (x + r > x0 && x - r < x1 && y + r > y0 && y - r < y1) return i;
  }
  return -1;
}

export function isFree(x, y) {
  return overlaps(x, y) < 0;
}

export function stepPos(pos, action) {
  let x = pos[0], y = pos[1];
  const ax = Math.min(Math.max(action[0], -1.0), 1.0);
  const ay = Math.min(Math.max(action[1], -1.0), 1.0);
  const dx = (ax * SPEED) / SUBSTEPS;
  const dy = (ay * SPEED) / SUBSTEPS;
  const r = AGENT_R;
  for (let s = 0; s < SUBSTEPS; s++) {
    let nx = x + dx;
    let i = overlaps(nx, y);
    if (i >= 0) nx = dx > 0 ? WALLS[i][0] - r : WALLS[i][2] + r;
    x = nx;
    let ny = y + dy;
    i = overlaps(x, ny);
    if (i >= 0) ny = dy > 0 ? WALLS[i][1] - r : WALLS[i][3] + r;
    y = ny;
  }
  return [x, y];
}

export function roomOf(x) {
  return x < WALL_X0 ? 0 : 1;
}

const STATIC = (() => {
  const img = new Float64Array(SIZE * SIZE * 3);
  for (let k = 0; k < SIZE * SIZE; k++) img.set(BG, k * 3);
  for (const [x0, y0, x1, y1] of WALLS)
    for (let i = Math.trunc(y0); i < Math.trunc(y1); i++)
      for (let j = Math.trunc(x0); j < Math.trunc(x1); j++) img.set(WALL, (i * SIZE + j) * 3);
  return img;
})();
const SUB = Array.from({ length: SS }, (_, k) => (k + 0.5) / SS);

// Render a state to a 64x64x3 (HWC) Uint8Array, identical to env.render().
export function render(pos, out = new Uint8Array(SIZE * SIZE * 3)) {
  const img = Float64Array.from(STATIC);
  const cx = pos[0], cy = pos[1];
  const r2 = AGENT_R * AGENT_R;
  const i0 = Math.max(Math.floor(cy - AGENT_R) - 1, 0);
  const i1 = Math.min(Math.floor(cy + AGENT_R) + 2, SIZE);
  const j0 = Math.max(Math.floor(cx - AGENT_R) - 1, 0);
  const j1 = Math.min(Math.floor(cx + AGENT_R) + 2, SIZE);
  for (let i = i0; i < i1; i++) {
    for (let j = j0; j < j1; j++) {
      let cnt = 0;
      for (const sy of SUB) {
        const py = i + sy - cy;
        for (const sx of SUB) {
          const px = j + sx - cx;
          if (px * px + py * py <= r2) cnt++;
        }
      }
      if (cnt) {
        const c = cnt / (SS * SS);
        const k = (i * SIZE + j) * 3;
        for (let ch = 0; ch < 3; ch++) img[k + ch] = img[k + ch] * (1.0 - c) + AGENT[ch] * c;
      }
    }
  }
  for (let k = 0; k < img.length; k++) out[k] = Math.floor(img[k] + 0.5);
  return out;
}

// Small seeded PRNG (mulberry32) for sampling start/goal positions in the browser.
export function rng(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function sampleFree(rand, room = null) {
  for (;;) {
    const x = BORDER + AGENT_R + rand() * (SIZE - 2 * (BORDER + AGENT_R));
    const y = BORDER + AGENT_R + rand() * (SIZE - 2 * (BORDER + AGENT_R));
    if (room !== null && roomOf(x) !== room) continue;
    if (isFree(x, y)) return [x, y];
  }
}
