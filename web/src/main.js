// LatentPlanner demo: env (exact port of the training env) + ONNX world model + CEM, all in the browser.
// Two scenes share the loop: "Two rooms" (navigate through a door) and "Push" (push a block to a target).
import * as E from "./env.js";
import * as EP from "./env_push.js";
import { Planner } from "./cem.js";

const ORT_VERSION = "1.31.1";
const ort = window.ort;
const $ = (id) => document.getElementById(id);
const STEP_MS = 70; // animation time per env step

const ui = {
  arena: $("arena"), overlay: $("overlay"), seeNow: $("seeNow"), seeGoal: $("seeGoal"), spark: $("spark"),
  status: $("status"), steps: $("steps"), planMs: $("planMs"), search: $("search"), backend: $("backend"),
  surpriseVal: $("surpriseVal"), showPlan: $("showPlan"), showCost: $("showCost"), costToggle: $("costToggle"),
  play: $("play"), newTask: $("newTask"), nudge: $("nudge"), hint: $("hint"), sceneNote: $("sceneNote"),
  loading: $("loading"), loadBar: $("loadBar"), loadText: $("loadText"),
};

const rand = E.rng((Date.now() ^ 0x5eed) >>> 0);
const dist = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1]);
const clamp = (v, lo, hi) => Math.min(Math.max(v, lo), hi);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// ---------------------------------------------------------------------------------------------------- scenes
const SCENES = {
  tworoom: {
    dir: "model/",
    hint: "Tap to set a goal. Drag the agent to move it.",
    note: "",
    render: E.render,
    step: E.stepPos,
    initState: () => E.sampleFree(rand, 0),
    initGoal: () => E.sampleFree(rand, 1),
    randomGoal(s) {
      for (;;) {
        const g = E.sampleFree(rand, 1 - E.roomOf(s[0]));
        if (dist(g, s) > 15) return g;
      }
    },
    goalFromTap: (p) => (E.isFree(p[0], p[1]) ? p : null),
    reached: (s, g) => dist(s, g) <= 4,
    grab: (s, p) => (dist(p, s) < E.AGENT_R + 3.5 ? "agent" : null),
    drag: (s, what, p) => (E.isFree(p[0], p[1]) ? p : null),
    teleport: () => E.sampleFree(rand),
    drawGoal(ctx, g, X, sc) {
      ctx.strokeStyle = "#ffd166";
      ctx.lineWidth = Math.max(2, sc * 0.4);
      ctx.beginPath();
      ctx.arc(X(g[0]), X(g[1]), X(E.AGENT_R + 1.2), 0, 2 * Math.PI);
      ctx.stroke();
      ctx.globalAlpha = 0.25;
      ctx.beginPath();
      ctx.arc(X(g[0]), X(g[1]), X(4), 0, 2 * Math.PI);
      ctx.stroke();
      ctx.globalAlpha = 1;
    },
  },
  push: {
    dir: "model_push/",
    hint: "Tap where the block should go. Drag the agent or the block to move them.",
    note: "",
    render: EP.render,
    step: EP.stepState,
    initState: () => EP.sampleState(rand),
    initGoal: null, // set from the start state below
    // Goal image = block at the target and the agent just behind it along the main push direction, which is
    // what the end of a push looks like in the training data and in the test tasks.
    goalAt(s, b) {
      const bx = clamp(b[0], EP.LO_B, EP.HI_B), by = clamp(b[1], EP.LO_B, EP.HI_B);
      const dx = bx - s[2], dy = by - s[3];
      if (Math.hypot(dx, dy) < 4) return null;
      const side = Math.abs(dx) >= Math.abs(dy) ? [Math.sign(dx), 0] : [0, Math.sign(dy)];
      const off = EP.BLOCK_H + E.AGENT_R + 1.5;
      return [clamp(bx - side[0] * off, EP.LO_A, EP.HI_A), clamp(by - side[1] * off, EP.LO_A, EP.HI_A), bx, by];
    },
    randomGoal(s) {
      for (let k = 0; k < 100; k++) {
        const ax = rand() < 0.5, sign = rand() < 0.5 ? -1 : 1, d = 10 + rand() * 16;
        const b = [s[2] + (ax ? sign * d : 0), s[3] + (ax ? 0 : sign * d)];
        if (b[0] < EP.LO_B + 2 || b[0] > EP.HI_B - 2 || b[1] < EP.LO_B + 2 || b[1] > EP.HI_B - 2) continue;
        const g = this.goalAt(s, b);
        if (g) return g;
      }
      return this.goalAt(s, [32, 32]) || [s[0], s[1], 32, 40];
    },
    goalFromTap(p, s) { return this.goalAt(s, p); },
    reached: (s, g) => dist([s[2], s[3]], [g[2], g[3]]) <= 3,
    grab(s, p) {
      if (dist(p, s) < E.AGENT_R + 3) return "agent";
      if (Math.abs(p[0] - s[2]) < EP.BLOCK_H + 1.5 && Math.abs(p[1] - s[3]) < EP.BLOCK_H + 1.5) return "block";
      return null;
    },
    drag(s, what, p) {
      const n = what === "agent"
        ? [clamp(p[0], EP.LO_A, EP.HI_A), clamp(p[1], EP.LO_A, EP.HI_A), s[2], s[3]]
        : [s[0], s[1], clamp(p[0], EP.LO_B, EP.HI_B), clamp(p[1], EP.LO_B, EP.HI_B)];
      const sep = Math.max(Math.abs(n[0] - n[2]), Math.abs(n[1] - n[3])) >= E.AGENT_R + EP.BLOCK_H;
      return sep ? n : null;
    },
    teleport(s) {
      for (;;) {
        const n = [EP.LO_A + rand() * (EP.HI_A - EP.LO_A), EP.LO_A + rand() * (EP.HI_A - EP.LO_A), s[2], s[3]];
        if (Math.max(Math.abs(n[0] - n[2]), Math.abs(n[1] - n[3])) > E.AGENT_R + EP.BLOCK_H + 1) return n;
      }
    },
    drawGoal(ctx, g, X, sc) {
      const h = EP.BLOCK_H;
      ctx.strokeStyle = "#ffd166";
      ctx.lineWidth = Math.max(2, sc * 0.4);
      ctx.setLineDash([sc * 1.2, sc * 0.8]);
      ctx.strokeRect(X(g[2] - h), X(g[3] - h), X(2 * h), X(2 * h));
      ctx.setLineDash([]);
      ctx.globalAlpha = 0.35;
      ctx.beginPath();
      ctx.arc(X(g[0]), X(g[1]), X(E.AGENT_R), 0, 2 * Math.PI);
      ctx.stroke();
      ctx.globalAlpha = 1;
    },
  },
};

// ---------------------------------------------------------------------------------------------------- state
let sceneName = location.hash === "#push" ? "push" : "tworoom";
let scene = SCENES[sceneName];
const cache = {}; // per scene: {meta, planner, gridZ}
let planner = null, meta = null, gridZ = null;

const S = {
  pos: null, // full state: [x, y] for two rooms, [agentX, agentY, blockX, blockY] for push
  goal: null,
  running: true,
  trail: [],
  blockTrail: [],
  plan: null, // {mean, zs, cost, ms}
  warm: null,
  zGoal: null,
  zPredNext: null,
  steps: 0,
  reached: false,
  surprise: [],
  planTimes: [],
  dragging: null,
  costImg: null,
  ready: false,
  goalVersion: 0,
  sceneVersion: 0,
};

function resetScene() {
  S.pos = scene.initState();
  S.goal = scene.initGoal ? scene.initGoal() : scene.randomGoal(S.pos);
  S.surprise = [];
  S.planTimes = [];
  S.zPredNext = null;
  ui.surpriseVal.textContent = "–";
  ui.planMs.textContent = "–";
}

// ---------------------------------------------------------------------------------------------------- drawing
function putFrame(canvas, frame) {
  const ctx = canvas.getContext("2d");
  const img = ctx.createImageData(64, 64);
  for (let k = 0, j = 0; k < 64 * 64 * 3; k += 3, j += 4) {
    img.data[j] = frame[k];
    img.data[j + 1] = frame[k + 1];
    img.data[j + 2] = frame[k + 2];
    img.data[j + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
}

function sizeOverlay() {
  const r = ui.overlay.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  ui.overlay.width = Math.round(r.width * dpr);
  ui.overlay.height = Math.round(r.height * dpr);
  const rr = ui.spark.getBoundingClientRect();
  ui.spark.width = Math.round(rr.width * dpr);
  ui.spark.height = Math.round(rr.height * dpr);
}

// perceptual ramp, dark violet (far) -> pink -> warm yellow (near the goal)
const RAMP = [[24, 16, 48], [70, 30, 100], [200, 80, 122], [255, 209, 102]];
function ramp(t) {
  const x = Math.min(Math.max(t, 0), 1) * (RAMP.length - 1);
  const i = Math.min(Math.floor(x), RAMP.length - 2), f = x - i;
  return RAMP[i].map((c, k) => Math.round(c + (RAMP[i + 1][k] - c) * f));
}

function buildCostImage() {
  S.costImg = null;
  if (!gridZ || !S.zGoal || !meta.grid) return;
  const { n } = meta.grid, D = meta.latent_dim;
  const d = new Float32Array(n * n);
  let lo = Infinity, hi = -Infinity;
  for (let p = 0; p < n * n; p++) {
    let s = 0;
    for (let k = 0; k < D; k++) {
      const e = gridZ[p * D + k] - S.zGoal[k];
      s += e * e;
    }
    d[p] = Math.sqrt(s);
    if (!Number.isNaN(d[p])) { lo = Math.min(lo, d[p]); hi = Math.max(hi, d[p]); }
  }
  const c = document.createElement("canvas");
  c.width = c.height = n;
  const ctx = c.getContext("2d");
  const img = ctx.createImageData(n, n);
  for (let p = 0; p < n * n; p++) {
    if (Number.isNaN(d[p])) continue;
    const [r, g, b] = ramp(1 - (d[p] - lo) / (hi - lo));
    img.data.set([r, g, b, 190], p * 4);
  }
  ctx.putImageData(img, 0, 0);
  S.costImg = c;
}

function polyline(ctx, pts, X) {
  ctx.beginPath();
  pts.forEach((p, i) => (i ? ctx.lineTo(X(p[0]), X(p[1])) : ctx.moveTo(X(p[0]), X(p[1]))));
  ctx.stroke();
}

function draw() {
  if (!S.pos) return;
  const frame = scene.render(S.pos);
  putFrame(ui.arena, frame);
  putFrame(ui.seeNow, frame);
  const ctx = ui.overlay.getContext("2d");
  const W = ui.overlay.width, sc = W / 64;
  ctx.clearRect(0, 0, W, W);
  const X = (v) => v * sc;

  if (sceneName === "tworoom" && ui.showCost.checked && S.costImg) {
    const { x0, n } = meta.grid;
    ctx.imageSmoothingEnabled = true;
    ctx.drawImage(S.costImg, X(x0 - 0.5), X(x0 - 0.5), X(n), X(n));
    ctx.fillStyle = "#f06048"; // redraw the agent on top of the map so it stays visible
    ctx.beginPath();
    ctx.arc(X(S.pos[0]), X(S.pos[1]), X(E.AGENT_R), 0, 2 * Math.PI);
    ctx.fill();
  }

  ctx.lineJoin = "round";
  ctx.lineWidth = Math.max(1.5, sc * 0.35);
  if (S.blockTrail.length > 1) {
    ctx.strokeStyle = "rgba(72, 196, 176, 0.45)";
    polyline(ctx, S.blockTrail, X);
  }
  if (S.trail.length > 1) {
    ctx.strokeStyle = "rgba(240, 96, 72, 0.45)";
    polyline(ctx, S.trail, X);
  }
  if (S.goal) scene.drawGoal(ctx, S.goal, X, sc);

  // imagined path of the current best plan (decoded with the visual-only linear probe)
  if (ui.showPlan.checked && S.plan && planner) {
    const D = meta.latent_dim, steps = S.plan.zs.length / D;
    const from = S.planFrom || S.pos;
    const dec = [];
    for (let t = 0; t < steps; t++) dec.push(planner.probe(S.plan.zs.subarray(t * D, (t + 1) * D)));
    const paths = [[[from[0], from[1]], ...dec.map((q) => [q[0], q[1]]), "#7bdff2"]];
    if (sceneName === "push") paths.push([[from[2], from[3]], ...dec.map((q) => [q[2], q[3]]), "#a6f0dd"]);
    for (const path of paths) {
      const color = path.pop();
      ctx.strokeStyle = color;
      ctx.globalAlpha = 0.85;
      ctx.lineWidth = Math.max(1.5, sc * 0.3);
      ctx.setLineDash([sc * 0.8, sc * 0.8]);
      polyline(ctx, path, X);
      ctx.setLineDash([]);
      ctx.fillStyle = color;
      for (let i = 1; i < path.length; i++) {
        ctx.beginPath();
        ctx.arc(X(path[i][0]), X(path[i][1]), Math.max(1.5, sc * 0.45), 0, 2 * Math.PI);
        ctx.fill();
      }
      ctx.globalAlpha = 1;
    }
  }
  drawSpark();
}

function drawSpark() {
  const c = ui.spark, ctx = c.getContext("2d");
  const w = c.width, h = c.height;
  ctx.clearRect(0, 0, w, h);
  const v = S.surprise.slice(-60);
  if (!v.length) return;
  // square-root scale so small everyday errors and big surprises both stay readable
  const top = Math.sqrt(Math.max(0.5, ...v));
  const y = (s) => h - 3 - (Math.sqrt(s) / top) * (h - 8);
  ctx.strokeStyle = "#232a3a";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(0, h - 2.5);
  ctx.lineTo(w, h - 2.5);
  ctx.stroke();
  ctx.strokeStyle = "#f06048";
  ctx.lineWidth = Math.max(1.5, w / 300);
  ctx.beginPath();
  const dx = w / 59;
  v.forEach((s, i) => (i ? ctx.lineTo(i * dx, y(s)) : ctx.moveTo(i * dx, y(s))));
  ctx.stroke();
}

function setStatus(t) {
  ui.status.textContent = t;
}

// ---------------------------------------------------------------------------------------------------- control
function pushTrail() {
  S.trail.push([S.pos[0], S.pos[1]]);
  if (sceneName === "push") S.blockTrail.push([S.pos[2], S.pos[3]]);
}

async function setGoal(g) {
  S.goal = g;
  S.goalVersion++;
  S.reached = false;
  S.warm = null;
  S.plan = null;
  S.trail = [];
  S.blockTrail = [];
  pushTrail();
  S.steps = 0;
  ui.steps.textContent = "0";
  const gf = scene.render(g);
  putFrame(ui.seeGoal, gf);
  if (planner) {
    S.zGoal = await planner.encode(gf);
    buildCostImage();
  }
  draw();
}

async function loop() {
  for (;;) {
    if (!S.ready || !S.running || S.reached || S.dragging) {
      await sleep(60);
      continue;
    }
    const version = S.goalVersion, sv = S.sceneVersion;
    const z = await planner.encode(scene.render(S.pos));
    if (sv !== S.sceneVersion) continue;
    if (S.zPredNext) {
      let e = 0;
      for (let k = 0; k < z.length; k++) e += (z[k] - S.zPredNext[k]) ** 2;
      S.surprise.push(e / z.length);
      ui.surpriseVal.textContent = (e / z.length).toFixed(3);
    }
    setStatus("Planning");
    const res = await planner.plan(z, S.zGoal, S.warm);
    if (version !== S.goalVersion || sv !== S.sceneVersion) { S.zPredNext = null; continue; } // goal/scene changed
    S.plan = res;
    S.planFrom = S.pos.slice();
    S.planTimes.push(res.ms);
    const avg = S.planTimes.slice(-20).reduce((a, b) => a + b, 0) / Math.min(20, S.planTimes.length);
    ui.planMs.textContent = `${res.ms.toFixed(0)} ms (avg ${avg.toFixed(0)})`;
    // the imagined latent after executing the first knot = the model's prediction for the next replan
    const D = meta.latent_dim, k1 = meta.cem.hold / meta.frameskip;
    S.zPredNext = res.zs.slice((k1 - 1) * D, k1 * D);
    const a = [res.mean[0], res.mean[1]];
    setStatus("Moving");
    let interrupted = false;
    for (let k = 0; k < meta.cem.hold; k++) {
      while (S.dragging) await sleep(30);
      if (version !== S.goalVersion || sv !== S.sceneVersion || !S.running) { interrupted = true; break; }
      S.pos = scene.step(S.pos, a);
      pushTrail();
      S.steps++;
      ui.steps.textContent = String(S.steps);
      draw();
      if (scene.reached(S.pos, S.goal)) {
        S.reached = true;
        S.zPredNext = null;
        S.plan = null;
        setStatus(`Reached in ${S.steps} steps`);
        draw();
        const v = S.goalVersion;
        setTimeout(() => {
          if (S.reached && v === S.goalVersion && S.running) setGoal(scene.randomGoal(S.pos));
        }, 2200);
        break;
      }
      await sleep(STEP_MS);
    }
    if (interrupted) {
      S.zPredNext = null; // the first knot was not fully executed, so its prediction doesn't apply
      continue;
    }
    S.warm = Planner.shift(res.mean);
  }
}

// ---------------------------------------------------------------------------------------------------- input
function toArena(ev) {
  const r = ui.overlay.getBoundingClientRect();
  return [((ev.clientX - r.left) / r.width) * 64, ((ev.clientY - r.top) / r.height) * 64];
}

ui.overlay.addEventListener("pointerdown", (ev) => {
  if (!S.pos) return;
  const p = toArena(ev);
  const what = scene.grab(S.pos, p);
  if (what) {
    S.dragging = what;
    ui.overlay.classList.add("dragging");
    ui.overlay.setPointerCapture(ev.pointerId);
  } else {
    const g = scene.goalFromTap(p, S.pos);
    if (g) setGoal(g);
  }
});
ui.overlay.addEventListener("pointermove", (ev) => {
  if (!S.dragging) return;
  const n = scene.drag(S.pos, S.dragging, toArena(ev));
  if (n) {
    S.pos = n;
    pushTrail();
    draw();
  }
});
const endDrag = () => {
  if (!S.dragging) return;
  S.dragging = null;
  S.warm = null;
  ui.overlay.classList.remove("dragging");
};
ui.overlay.addEventListener("pointerup", endDrag);
ui.overlay.addEventListener("pointercancel", endDrag);

ui.play.addEventListener("click", () => {
  S.running = !S.running;
  ui.play.textContent = S.running ? "Pause" : "Play";
  if (S.running && S.reached) setGoal(scene.randomGoal(S.pos));
});
ui.newTask.addEventListener("click", () => setGoal(scene.randomGoal(S.pos)));
ui.nudge.addEventListener("click", () => {
  S.pos = scene.teleport(S.pos);
  pushTrail();
  S.warm = null;
  draw();
});
ui.showPlan.addEventListener("change", draw);
ui.showCost.addEventListener("change", draw);
window.addEventListener("resize", () => { sizeOverlay(); draw(); });

for (const b of document.querySelectorAll("[data-scene]")) {
  b.addEventListener("click", () => switchScene(b.dataset.scene));
}

// ---------------------------------------------------------------------------------------------------- loading
async function fetchWithProgress(url, onBytes) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`);
  const reader = res.body.getReader();
  const chunks = [];
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    onBytes(value.length);
  }
  const out = new Uint8Array(chunks.reduce((n, c) => n + c.length, 0));
  let o = 0;
  for (const c of chunks) { out.set(c, o); o += c.length; }
  return out;
}

async function loadModel(name) {
  if (cache[name]) return cache[name];
  const dir = SCENES[name].dir;
  const m = await (await fetch(`${dir}meta.json`)).json();
  const files = Object.keys(m.bytes);
  const total = files.reduce((n, f) => n + m.bytes[f], 0);
  let got = 0;
  const tick = (n) => {
    got += n;
    ui.loadBar.style.width = `${Math.min(100, (100 * got) / total).toFixed(1)}%`;
    ui.loadText.textContent = `Loading model ${(got / 1e6).toFixed(1)} / ${(total / 1e6).toFixed(1)} MB`;
  };
  const bytes = await Promise.all(files.map((f) => fetchWithProgress(dir + f, tick)));
  const get = (f) => bytes[files.indexOf(f)];
  ui.loadText.textContent = "Starting the runtime…";
  ort.env.wasm.numThreads = 1; // static hosting has no cross-origin isolation, so no wasm threads
  ort.env.wasm.wasmPaths = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/`;
  const opts = { executionProviders: ["wasm"], graphOptimizationLevel: "all" };
  const [enc, ro] = await Promise.all([
    ort.InferenceSession.create(get("encoder.onnx"), opts),
    ort.InferenceSession.create(get("rollout.onnx"), opts),
  ]);
  const g = get("grid_z.bin");
  cache[name] = {
    meta: m,
    planner: new Planner(ort, enc, ro, m, rand),
    gridZ: g ? new Float32Array(g.buffer, g.byteOffset, g.byteLength / 4) : null,
  };
  return cache[name];
}

async function switchScene(name, first = false) {
  if (!first && name === sceneName) return;
  S.ready = false;
  S.sceneVersion++;
  S.plan = null;
  sceneName = name;
  scene = SCENES[name];
  history.replaceState(null, "", name === "push" ? "#push" : location.pathname + location.search);
  for (const b of document.querySelectorAll("[data-scene]")) b.setAttribute("aria-pressed", String(b.dataset.scene === name));
  ui.costToggle.hidden = name !== "tworoom";
  ui.hint.textContent = scene.hint;
  ui.sceneNote.textContent = scene.note;
  ui.sceneNote.hidden = !scene.note;
  resetScene();
  putFrame(ui.seeGoal, scene.render(S.goal));
  draw();
  ui.loading.classList.remove("done");
  setStatus("Loading");
  try {
    ({ meta, planner, gridZ } = await loadModel(name));
  } catch (e) {
    ui.loadText.textContent = `Could not load the model: ${e.message}`;
    setStatus("Error");
    console.error(e);
    return;
  }
  const c = meta.cem, steps = (c.knots * c.hold) / meta.frameskip;
  ui.search.textContent = `${c.samples} plans × ${c.iters} rounds × ${steps} steps`;
  ui.backend.textContent = `ONNX Runtime Web ${ORT_VERSION}, wasm`;
  // for automated tests (tests/web_smoke.py, tests/web_eval.py)
  window.__lp = { S, meta, planner, E, EP, Planner, scene: name };
  await setGoal(S.goal);
  ui.loading.classList.add("done");
  S.ready = true;
  setStatus("Planning");
}

// ---------------------------------------------------------------------------------------------------- results
async function loadResults() {
  try {
    const r = await (await fetch("results.json")).json();
    renderResults(r);
  } catch {
    /* results table is optional */
  }
}

function renderResults(r) {
  const pct = (m, s) => (s === undefined || s === null ? `${(100 * m).toFixed(0)}%` : `${(100 * m).toFixed(0)} ± ${(100 * s).toFixed(0)}%`);
  const table = (rows, cols) => `<div class="table-wrap"><table><thead><tr><th>Method</th>${cols.map((c) => `<th>${c[0]}</th>`).join("")}<th>Steps</th></tr></thead><tbody>${
    rows.map((x) => `<tr class="${x.ours ? "ours" : ""}"><td>${x.label}</td>${cols.map((c) => `<td>${pct(x[c[1]], x[`${c[1]}_std`])}</td>`).join("")}<td>${Number.isFinite(x.steps) ? x.steps.toFixed(1) : "–"}</td></tr>`).join("")
  }</tbody></table></div>`;
  let html = `<h3>Measured results: two rooms</h3>${table(r.rows, [["Success", "success"], ["Same room", "success_same"], ["Other room", "success_cross"]])}<p class="note">${r.note}</p>`;
  if (r.push_rows && r.push_rows.length) {
    html += `<h3>Measured results: push</h3>${table(r.push_rows, [["Success", "success"], ["Short push", "success_short"], ["Long push", "success_long"]])}<p class="note">${r.push_note}</p>`;
  }
  $("results").innerHTML = html;
}

function init() {
  sizeOverlay();
  loadResults();
  switchScene(sceneName, true);
  loop();
}

init();
