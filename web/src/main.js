// LatentPlanner demo: env (exact port of the training env) + ONNX world model + CEM, all in the browser.
import * as E from "./env.js";
import { Planner } from "./cem.js";

const ORT_VERSION = "1.31.1";
const ort = window.ort;
const $ = (id) => document.getElementById(id);
const SUCCESS_PX = 4;
const STEP_MS = 70; // animation time per env step

const ui = {
  arena: $("arena"), overlay: $("overlay"), seeNow: $("seeNow"), seeGoal: $("seeGoal"), spark: $("spark"),
  status: $("status"), steps: $("steps"), planMs: $("planMs"), search: $("search"), backend: $("backend"),
  surpriseVal: $("surpriseVal"), showPlan: $("showPlan"), showCost: $("showCost"),
  play: $("play"), newTask: $("newTask"), nudge: $("nudge"),
};

const rand = E.rng((Date.now() ^ 0x5eed) >>> 0);
const S = {
  pos: E.sampleFree(rand, 0),
  goal: null,
  running: true,
  trail: [],
  plan: null, // {mean, zs, cost, ms}
  warm: null,
  zGoal: null,
  zPredNext: null,
  steps: 0,
  reached: false,
  surprise: [],
  planTimes: [],
  dragging: false,
  costImg: null,
  ready: false,
  goalVersion: 0,
};
S.goal = E.sampleFree(rand, 1);

let planner = null, meta = null, gridZ = null;

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
  for (const c of [ui.spark]) {
    const rr = c.getBoundingClientRect();
    c.width = Math.round(rr.width * dpr);
    c.height = Math.round(rr.height * dpr);
  }
}

// perceptual ramp, dark violet (far) -> pink -> warm yellow (near the goal)
const RAMP = [[24, 16, 48], [70, 30, 100], [200, 80, 122], [255, 209, 102]];
function ramp(t) {
  const x = Math.min(Math.max(t, 0), 1) * (RAMP.length - 1);
  const i = Math.min(Math.floor(x), RAMP.length - 2), f = x - i;
  return RAMP[i].map((c, k) => Math.round(c + (RAMP[i + 1][k] - c) * f));
}

function buildCostImage() {
  if (!gridZ || !S.zGoal) return;
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

function draw() {
  const frame = E.render(S.pos);
  putFrame(ui.arena, frame);
  putFrame(ui.seeNow, frame);
  const ctx = ui.overlay.getContext("2d");
  const W = ui.overlay.width, sc = W / 64;
  ctx.clearRect(0, 0, W, W);
  const X = (v) => v * sc;

  if (ui.showCost.checked && S.costImg) {
    const { x0, n } = meta.grid;
    ctx.imageSmoothingEnabled = true;
    ctx.drawImage(S.costImg, X(x0 - 0.5), X(x0 - 0.5), X(n), X(n));
    // redraw the agent on top of the map so it stays visible
    ctx.fillStyle = "#f06048";
    ctx.beginPath();
    ctx.arc(X(S.pos[0]), X(S.pos[1]), X(E.AGENT_R), 0, 2 * Math.PI);
    ctx.fill();
  }

  // executed trail
  if (S.trail.length > 1) {
    ctx.strokeStyle = "rgba(240, 96, 72, 0.45)";
    ctx.lineWidth = Math.max(1.5, sc * 0.35);
    ctx.lineJoin = "round";
    ctx.beginPath();
    S.trail.forEach((p, i) => (i ? ctx.lineTo(X(p[0]), X(p[1])) : ctx.moveTo(X(p[0]), X(p[1]))));
    ctx.stroke();
  }

  // goal
  if (S.goal) {
    ctx.strokeStyle = "#ffd166";
    ctx.lineWidth = Math.max(2, sc * 0.4);
    ctx.setLineDash([]);
    ctx.beginPath();
    ctx.arc(X(S.goal[0]), X(S.goal[1]), X(E.AGENT_R + 1.2), 0, 2 * Math.PI);
    ctx.stroke();
    ctx.beginPath();
    ctx.arc(X(S.goal[0]), X(S.goal[1]), X(SUCCESS_PX), 0, 2 * Math.PI);
    ctx.globalAlpha = 0.25;
    ctx.stroke();
    ctx.globalAlpha = 1;
  }

  // imagined path of the current best plan (decoded with the visual-only linear probe)
  if (ui.showPlan.checked && S.plan && planner) {
    const D = meta.latent_dim, steps = S.plan.zs.length / D;
    const pts = [S.planFrom || S.pos];
    for (let t = 0; t < steps; t++) pts.push(planner.probe(S.plan.zs.subarray(t * D, (t + 1) * D)));
    ctx.strokeStyle = "rgba(123, 223, 242, 0.8)";
    ctx.lineWidth = Math.max(1.5, sc * 0.3);
    ctx.setLineDash([sc * 0.8, sc * 0.8]);
    ctx.beginPath();
    pts.forEach((p, i) => (i ? ctx.lineTo(X(p[0]), X(p[1])) : ctx.moveTo(X(p[0]), X(p[1]))));
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = "#7bdff2";
    for (let i = 1; i < pts.length; i++) {
      ctx.beginPath();
      ctx.arc(X(pts[i][0]), X(pts[i][1]), Math.max(1.5, sc * 0.45), 0, 2 * Math.PI);
      ctx.fill();
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
const dist = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1]);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function setGoal(g) {
  S.goal = g;
  S.goalVersion++;
  S.reached = false;
  S.warm = null;
  S.plan = null;
  S.trail = [S.pos.slice()];
  S.steps = 0;
  ui.steps.textContent = "0";
  const gf = E.render(g);
  putFrame(ui.seeGoal, gf);
  if (planner) {
    S.zGoal = await planner.encode(gf);
    buildCostImage();
  }
  draw();
}

function randomGoal() {
  for (;;) {
    const g = E.sampleFree(rand, 1 - E.roomOf(S.pos[0]));
    if (dist(g, S.pos) > 15) return g;
  }
}

async function loop() {
  for (;;) {
    if (!S.ready || !S.running || S.reached || S.dragging) {
      await sleep(60);
      continue;
    }
    const version = S.goalVersion;
    const z = await planner.encode(E.render(S.pos));
    if (S.zPredNext) {
      let e = 0;
      for (let k = 0; k < z.length; k++) e += (z[k] - S.zPredNext[k]) ** 2;
      S.surprise.push(e / z.length);
      ui.surpriseVal.textContent = (e / z.length).toFixed(3);
    }
    setStatus("Planning");
    const res = await planner.plan(z, S.zGoal, S.warm);
    if (version !== S.goalVersion) { S.zPredNext = null; continue; } // goal changed while planning
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
      if (version !== S.goalVersion || !S.running) { interrupted = true; break; } // new goal or paused
      S.pos = E.stepPos(S.pos, a);
      S.trail.push(S.pos.slice());
      S.steps++;
      ui.steps.textContent = String(S.steps);
      draw();
      if (dist(S.pos, S.goal) <= SUCCESS_PX) {
        S.reached = true;
        S.zPredNext = null;
        S.plan = null;
        setStatus(`Reached in ${S.steps} steps`);
        draw();
        const v = S.goalVersion;
        setTimeout(() => { if (S.reached && v === S.goalVersion && S.running) setGoal(randomGoal()); }, 2200);
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
  const p = toArena(ev);
  if (dist(p, S.pos) < E.AGENT_R + 3.5) {
    S.dragging = true;
    ui.overlay.classList.add("dragging");
    ui.overlay.setPointerCapture(ev.pointerId);
  } else if (E.isFree(p[0], p[1])) {
    setGoal(p);
  }
});
ui.overlay.addEventListener("pointermove", (ev) => {
  if (!S.dragging) return;
  const p = toArena(ev);
  if (E.isFree(p[0], p[1])) {
    S.pos = p;
    S.trail.push(p.slice());
    draw();
  }
});
const endDrag = () => {
  if (!S.dragging) return;
  S.dragging = false;
  S.warm = null;
  ui.overlay.classList.remove("dragging");
};
ui.overlay.addEventListener("pointerup", endDrag);
ui.overlay.addEventListener("pointercancel", endDrag);

ui.play.addEventListener("click", () => {
  S.running = !S.running;
  ui.play.textContent = S.running ? "Pause" : "Play";
  if (S.running && S.reached) setGoal(randomGoal());
});
ui.newTask.addEventListener("click", () => setGoal(randomGoal()));
ui.nudge.addEventListener("click", () => {
  S.pos = E.sampleFree(rand);
  S.trail.push(S.pos.slice());
  S.warm = null;
  draw();
});
ui.showPlan.addEventListener("change", draw);
ui.showCost.addEventListener("change", draw);
window.addEventListener("resize", () => { sizeOverlay(); draw(); });

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

async function load() {
  meta = await (await fetch("model/meta.json")).json();
  const total = meta.bytes["encoder.onnx"] + meta.bytes["rollout.onnx"] + meta.bytes["grid_z.bin"];
  let got = 0;
  const tick = (n) => {
    got += n;
    $("loadBar").style.width = `${Math.min(100, (100 * got) / total).toFixed(1)}%`;
    $("loadText").textContent = `Loading model ${(got / 1e6).toFixed(1)} / ${(total / 1e6).toFixed(1)} MB`;
  };
  const [encBytes, roBytes, gridBytes] = await Promise.all([
    fetchWithProgress("model/encoder.onnx", tick),
    fetchWithProgress("model/rollout.onnx", tick),
    fetchWithProgress("model/grid_z.bin", tick),
  ]);
  $("loadText").textContent = "Starting the runtime…";
  ort.env.wasm.numThreads = 1; // static hosting has no cross-origin isolation, so no wasm threads
  ort.env.wasm.wasmPaths = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/`;
  const opts = { executionProviders: ["wasm"], graphOptimizationLevel: "all" };
  const [enc, ro] = await Promise.all([
    ort.InferenceSession.create(encBytes, opts),
    ort.InferenceSession.create(roBytes, opts),
  ]);
  gridZ = new Float32Array(gridBytes.buffer, gridBytes.byteOffset, gridBytes.byteLength / 4);
  planner = new Planner(ort, enc, ro, meta, rand);
  const c = meta.cem, steps = (c.knots * c.hold) / meta.frameskip;
  ui.search.textContent = `${c.samples} plans × ${c.iters} rounds × ${steps} steps`;
  ui.backend.textContent = `ONNX Runtime Web ${ORT_VERSION}, wasm`;
  window.__lp = { S, meta, planner, E, Planner }; // for automated tests (tests/web_smoke.py, tests/web_eval.py)
  await setGoal(S.goal);
  $("loading").classList.add("done");
  S.ready = true;
  setStatus("Planning");
}

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
  const rows = r.rows.map((x) => `<tr class="${x.ours ? "ours" : ""}"><td>${x.label}</td>
      <td>${pct(x.success, x.success_std)}</td><td>${pct(x.success_same, x.success_same_std)}</td>
      <td>${pct(x.success_cross, x.success_cross_std)}</td><td>${Number.isFinite(x.steps) ? x.steps.toFixed(1) : "–"}</td></tr>`).join("");
  $("results").innerHTML = `<h3>Measured results</h3>
    <div class="table-wrap"><table><thead><tr><th>Method</th><th>Success</th><th>Same room</th><th>Other room</th><th>Steps</th></tr></thead>
    <tbody>${rows}</tbody></table></div><p class="note">${r.note}</p>`;
}

function init() {
  putFrame(ui.arena, E.render(S.pos));
  putFrame(ui.seeGoal, E.render(S.goal));
  sizeOverlay();
  draw();
  loadResults();
  load().catch((e) => {
    $("loadText").textContent = `Could not load the model: ${e.message}`;
    setStatus("Error");
    console.error(e);
  });
  loop();
}

init();
