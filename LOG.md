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
