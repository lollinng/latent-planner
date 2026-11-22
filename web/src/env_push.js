// Scene 2 "Push": line-by-line port of latent_planner/env_push.py (parity: tests/test_js_parity.py).
// State is [agentX, agentY, blockX, blockY]. Kinematic, axis-separated pushing; no momentum or rotation.
import { SIZE, BORDER, AGENT_R, SUBSTEPS } from "./env.js";

export const SPEED = 4.0;
export const BLOCK_H = 4.0;
const SS = 4;
const BG = [18.0, 22.0, 32.0];
const WALL = [110.0, 122.0, 148.0];
const AGENT = [240.0, 96.0, 72.0];
const BLOCK = [72.0, 196.0, 176.0];

export const WALLS = [
  [0, 0, SIZE, BORDER],
  [0, SIZE - BORDER, SIZE, SIZE],
  [0, 0, BORDER, SIZE],
  [SIZE - BORDER, 0, SIZE, SIZE],
];
export const LO_A = BORDER + AGENT_R, HI_A = SIZE - BORDER - AGENT_R;
export const LO_B = BORDER + BLOCK_H, HI_B = SIZE - BORDER - BLOCK_H;

function wallHit(x, y, h) {
  for (let i = 0; i < WALLS.length; i++) {
    const [x0, y0, x1, y1] = WALLS[i];
    if (x + h > x0 && x - h < x1 && y + h > y0 && y - h < y1) return i;
  }
  return -1;
}

function touch(ax, ay, bx, by) {
  const s = AGENT_R + BLOCK_H;
  return ax + s > bx && ax - s < bx && ay + s > by && ay - s < by;
}

export function stepState(state, action) {
  let [ax, ay, bx, by] = state;
  const a0 = Math.min(Math.max(action[0], -1.0), 1.0);
  const a1 = Math.min(Math.max(action[1], -1.0), 1.0);
  const dx = (a0 * SPEED) / SUBSTEPS;
  const dy = (a1 * SPEED) / SUBSTEPS;
  const r = AGENT_R, h = BLOCK_H;
  for (let s = 0; s < SUBSTEPS; s++) {
    let nx = ax + dx;
    let i = wallHit(nx, ay, r);
    if (i >= 0) nx = dx > 0 ? WALLS[i][0] - r : WALLS[i][2] + r;
    if (touch(nx, ay, bx, by)) {
      let nbx = dx > 0 ? nx + r + h : nx - r - h;
      const j = wallHit(nbx, by, h);
      if (j >= 0) nbx = dx > 0 ? WALLS[j][0] - h : WALLS[j][2] + h;
      nx = dx > 0 ? nbx - h - r : nbx + h + r;
      bx = nbx;
    }
    ax = nx;
    let ny = ay + dy;
    i = wallHit(ax, ny, r);
    if (i >= 0) ny = dy > 0 ? WALLS[i][1] - r : WALLS[i][3] + r;
    if (touch(ax, ny, bx, by)) {
      let nby = dy > 0 ? ny + r + h : ny - r - h;
      const j = wallHit(bx, nby, h);
      if (j >= 0) nby = dy > 0 ? WALLS[j][1] - h : WALLS[j][3] + h;
      ny = dy > 0 ? nby - h - r : nby + h + r;
      by = nby;
    }
    ay = ny;
  }
  return [ax, ay, bx, by];
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

export function render(state, out = new Uint8Array(SIZE * SIZE * 3)) {
  const img = Float64Array.from(STATIC);
  const [ax, ay, bx, by] = state;
  const h = BLOCK_H;
  for (let i = Math.max(Math.floor(by - h), 0); i < Math.min(Math.floor(by + h) + 1, SIZE); i++) {
    const cy = Math.min(i + 1.0, by + h) - Math.max(i, by - h);
    if (cy <= 0) continue;
    for (let j = Math.max(Math.floor(bx - h), 0); j < Math.min(Math.floor(bx + h) + 1, SIZE); j++) {
      const cx = Math.min(j + 1.0, bx + h) - Math.max(j, bx - h);
      if (cx <= 0) continue;
      const c = cx * cy;
      const k = (i * SIZE + j) * 3;
      for (let ch = 0; ch < 3; ch++) img[k + ch] = img[k + ch] * (1.0 - c) + BLOCK[ch] * c;
    }
  }
  const r2 = AGENT_R * AGENT_R;
  for (let i = Math.max(Math.floor(ay - AGENT_R) - 1, 0); i < Math.min(Math.floor(ay + AGENT_R) + 2, SIZE); i++) {
    for (let j = Math.max(Math.floor(ax - AGENT_R) - 1, 0); j < Math.min(Math.floor(ax + AGENT_R) + 2, SIZE); j++) {
      let cnt = 0;
      for (const sy of SUB) {
        const py = i + sy - ay;
        for (const sx of SUB) {
          const px = j + sx - ax;
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

// Random start: block away from the walls, agent anywhere not overlapping it.
export function sampleState(rand, blockMargin = 6.0) {
  const u = (lo, hi) => lo + rand() * (hi - lo);
  const bx = u(LO_B + blockMargin, HI_B - blockMargin), by = u(LO_B + blockMargin, HI_B - blockMargin);
  for (;;) {
    const ax = u(LO_A, HI_A), ay = u(LO_A, HI_A);
    if (Math.max(Math.abs(ax - bx), Math.abs(ay - by)) > AGENT_R + BLOCK_H + 1.0) return [ax, ay, bx, by];
  }
}

// A task like the test set (latent_planner/env_push.py make_tasks): a fresh start, then the scripted push policy
// for `steps` steps (walk behind the block on a random side, push for 4-13 steps, repeat, with action noise). The
// goal is where that ends, kept only if the block moved at least `minMove` px. Same distribution as the test
// tasks, though not the same random numbers as Python.
export function makeTask(rand, steps = 30, minMove = 6.0) {
  const normal = () => Math.sqrt(-2 * Math.log(1 - rand())) * Math.cos(2 * Math.PI * rand());
  const clip = (v) => Math.min(1, Math.max(-1, v));
  const SIDES = [[1, 0], [-1, 0], [0, 1], [0, -1]];
  for (;;) {
    const start = sampleState(rand);
    let s = start, side = null, phase = "approach", left = 0;
    for (let t = 0; t < steps; t++) {
      let act;
      if (side === null) {
        side = SIDES[Math.floor(rand() * 4)];
        phase = "approach";
        left = 4 + Math.floor(rand() * 10);
      }
      const off = BLOCK_H + AGENT_R + 1.5;
      if (phase === "approach") {
        const dx = s[2] - side[0] * off - s[0], dy = s[3] - side[1] * off - s[1], n = Math.hypot(dx, dy);
        if (n < 1.5) phase = "push";
        const k = Math.min(1, n / SPEED) / Math.max(n, 1e-6);
        act = [dx * k, dy * k];
      }
      if (phase === "push") {
        act = side.slice();
        if (--left <= 0) side = null;
      }
      s = stepState(s, [clip(act[0] + 0.3 * normal()), clip(act[1] + 0.3 * normal())]);
    }
    if (Math.hypot(s[2] - start[2], s[3] - start[3]) >= minMove) return { start, goal: s };
  }
}
