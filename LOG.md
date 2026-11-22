# LOG

## 2026-11-22
- 05:22 Started. Machine check: WSL2 laptop, RTX 3050 Laptop 4 GB, 10.7 GB RAM, 16 threads. No reverse proxy. -> research/SERVER.md
- 05:35 Licenses: le-wm MIT, stable-worldmodel MIT, stable-pretraining MIT, eb_jepa Apache-2.0, dino_wm MIT, PLDM MIT,
  vjepa2 MIT. **lejepa is CC BY-NC 4.0** and **keon/jepa has no license** -> read only, don't copy code.
- 05:50 Read LeWM source (jepa.py, module.py, train.py, configs). Paper config: ViT-tiny/14 at 224 px, embed 192,
  history 3, 6-layer AdaLN predictor, lambda=0.09, SIGReg 17 knots on [0,3], 1024 projections, AdamW 5e-5, bf16, CEM
  300 samples / 30 iters / top-30, horizon 5 x frameskip 5.
- 06:05 Decision: own 2-D env for the demo (exact JS port possible), LeWM-style objective, small CNN+MLP model. -> PLAN.md

## 2026-11-22 (session 2, resumed after the OOM kill)
- Previous session was OOM-killed in `python -m latent_planner.data`: `render_batch` was called on all 98k states at
  once and built a float64 copy of every frame (98k x 64x64x3 x 8 B = 9.6 GB) before converting to uint8.
  `research/NOTES.md` was never written; the research facts live in this log and PLAN.md.
- Fix: `render_batch` now starts from a uint8 copy of the static layer and only blends the 10x10 agent window in
  float64 (same arithmetic as `render()`, parity test passes). `data.generate` renders 64 episodes per chunk and
  appends raw bytes after a .npy header (no memmap, so written pages don't count toward RSS), gc between chunks, and
  aborts above 3 GB RSS.
- Result: data/train 2000 eps x 49 frames (1.20 GB on disk, 56% of episodes cross the door) in 11 s with
  **peak RSS 0.15 GB**; data/val 200 eps (seed 1). System used memory moved by ~64 MB during generation.
- 06:00 Training loop works. First full run (lam 0.09, seed 0, 6000 steps, 5 min): 16-step open-loop rollout
  error 0.022 of latent variance, effective rank 23/32, linear position probe RMSE 0.57 px (probe is diagnostic only).
  Host peak RSS 1.81 GB (mostly CUDA libs), GPU peak 2.55 GB.
- 06:02 Planner bug found with the oracle baseline: final-state-only cost under receding-horizon MPC never arrives
  (the goal is always at the end of a fresh window), oracle 0.24 success. Adding a running cost (final + 1.0 x mean
  over the horizon) fixes it: oracle 1.00 same-room / 0.75 cross-room on the tuning set.
- 06:04 Latent CEM (lam 0.09 s0) on the 100 **tuning** tasks: 1.00 success (same 1.00, cross 1.00), 9.9 steps,
  0.8 s per replan for all 100 tasks batched. Plotted cross-room trajectories: they go through the door on
  near-shortest paths. Planner settings locked from the tuning set: K=5 knots x hold 4, S=300, I=10, E=30, rw=1.0.
- 06:05 Launched the experiment matrix detached (`scripts/train_all.sh`, setsid/nohup): lam 0.09 x3 seeds,
  lam 0 x3 (no SIGReg), frameskip-4 lam 0.09 x3 (cheaper planning for the browser).
