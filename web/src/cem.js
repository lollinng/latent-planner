// CEM planning in latent space with onnxruntime-web. Mirrors latent_planner/planner.py:
// a plan is K knots of 2-D actions, each held for `hold` env steps; one rollout.onnx call imagines the whole
// horizon for every sample; cost = ||z_H - z_goal||^2 + running_w * mean_t ||z_t - z_goal||^2.

export function gaussian(rand) {
  // Box-Muller, one sample per call
  let u = 0;
  while (u === 0) u = rand();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rand());
}

export class Planner {
  constructor(ort, encoderSession, rolloutSession, meta, rand) {
    this.ort = ort;
    this.enc = encoderSession;
    this.ro = rolloutSession;
    this.meta = meta;
    this.cfg = { ...meta.cem };
    this.D = meta.latent_dim;
    this.steps = (meta.cem.knots * meta.cem.hold) / meta.frameskip; // predictor steps per plan
    this.rand = rand;
  }

  async encode(frameU8) {
    const t = new this.ort.Tensor("uint8", frameU8, [1, 64, 64, 3]);
    const out = await this.enc.run({ frames: t });
    return Float32Array.from(out.z.data);
  }

  async imagine(z0, knots, S) {
    const { D } = this;
    const K = this.cfg.knots;
    const z0s = new Float32Array(S * D);
    for (let s = 0; s < S; s++) z0s.set(z0, s * D);
    const out = await this.ro.run({
      z0: new this.ort.Tensor("float32", z0s, [S, D]),
      knots: new this.ort.Tensor("float32", knots, [S, K, 2]),
    });
    return out.zs.data; // (S, steps, D)
  }

  costs(zs, zg, S) {
    const { D, steps } = this;
    const rw = this.cfg.running_w;
    const cost = new Float64Array(S);
    for (let s = 0; s < S; s++) {
      let run = 0, last = 0;
      for (let t = 0; t < steps; t++) {
        let d2 = 0;
        const o = (s * steps + t) * D;
        for (let d = 0; d < D; d++) {
          const e = zs[o + d] - zg[d];
          d2 += e * e;
        }
        run += d2;
        if (t === steps - 1) last = d2;
      }
      cost[s] = last + (rw * run) / steps;
    }
    return cost;
  }

  // Returns {mean (K*2), zs (steps*D imagined latents of the mean plan), cost, ms}.
  async plan(z0, zg, warm = null) {
    const t0 = performance.now();
    const { knots: K, samples: S, elites: E, iters, init_std, min_std, momentum: m } = this.cfg;
    const n = K * 2;
    let mean = warm ? Float32Array.from(warm) : new Float32Array(n);
    let std = new Float32Array(n).fill(init_std);
    const cand = new Float32Array(S * n);
    for (let it = 0; it < iters; it++) {
      for (let s = 0; s < S; s++)
        for (let j = 0; j < n; j++) {
          const v = s === 0 ? mean[j] : mean[j] + std[j] * gaussian(this.rand);
          cand[s * n + j] = Math.min(1, Math.max(-1, v));
        }
      const zs = await this.imagine(z0, cand, S);
      const cost = this.costs(zs, zg, S);
      const idx = Array.from(cost.keys()).sort((a, b) => cost[a] - cost[b]).slice(0, E);
      const nm = new Float32Array(n), ns = new Float32Array(n);
      for (const s of idx) for (let j = 0; j < n; j++) nm[j] += cand[s * n + j] / E;
      for (const s of idx) for (let j = 0; j < n; j++) ns[j] += (cand[s * n + j] - nm[j]) ** 2 / (E - 1);
      for (let j = 0; j < n; j++) {
        mean[j] = m * mean[j] + (1 - m) * nm[j];
        std[j] = Math.max(m * std[j] + (1 - m) * Math.sqrt(ns[j]), min_std);
      }
    }
    const zs = await this.imagine(z0, mean, 1);
    const cost = this.costs(zs, zg, 1)[0];
    return { mean, zs: Float32Array.from(zs), cost, ms: performance.now() - t0 };
  }

  // Warm start for the next replan: drop the executed first knot, pad with zeros.
  static shift(mean) {
    const out = new Float32Array(mean.length);
    out.set(mean.subarray(2));
    return out;
  }

  // Visual only: linear probe z -> (x, y), fitted after training. The planner never uses it.
  probe(z) {
    const { W, b } = this.meta.probe;
    const p = [b[0], b[1]];
    for (let d = 0; d < this.D; d++) {
      p[0] += W[0][d] * z[d];
      p[1] += W[1][d] * z[d];
    }
    return p;
  }
}
