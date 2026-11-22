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
- 06:10 Browser demo works end to end (Chrome and WebKit, desktop 1200x900 and phone 390x844, no horizontal
  overflow): loads, plans with ORT-web 1.31.1 wasm (1 thread), reaches goals. Plan time ~0.8 s per replan on this
  laptop with the frameskip-1 model (300 samples x 10 iters x 20 steps), too slow for phones -> frameskip-4 model.
- 06:12 JS env port is bit-exact with env.py (positions float64-equal, identical pixels; tests/test_js_parity.py).
- 06:14 Probes on lam0.09 s0: surprise AUROC 1.00 (median error 0.003 real vs 1.89 teleport); latent distance is
  not geodesic (Spearman 0.37 vs geodesic, 0.36 vs straight line) but a ~12 px basin around the goal, flat beyond.
  Door crossing comes from the 20-step horizon finding the basin, and the flat far field avoids the wall trap that
  the straight-line oracle falls into. Demo shows this as the "cost map" overlay (grid_z.bin) instead of PCA
  (top-2 PCs explain only 19%).
- 06:16 Test-set baselines (100 tasks): random 0.11, straight line 0.70 (same 1.00 / other room 0.40),
  oracle CEM 0.87 (1.00 / 0.74).
- 06:18 fp16 weight storage in the ONNX files (Cast back to fp32, folded at load): model download 5.3 -> 2.9 MB.
  Planning with fp16-rounded weights on the tuning set: 1.00 success either way (10.1 vs 9.9 steps).
- 06:21-06:32 lam 0 (no SIGReg) x3: collapse in every seed. Embedding std 0.0002-0.0003 (vs ~0.96), pred loss
  ~2e-7, position probe at chance (20-22 px). Effective rank reads ~31/32 for these collapsed models because it is
  scale-invariant (tiny isotropic noise); report embedding std as the collapse signal instead.
- 06:35 Added free Kaggle T4 GPUs as a second place to run jobs. Wrote scripts/kaggle_run.py: private
  T4 script kernel with the code embedded as a tarball (repo stays private), data regenerated from seed with
  sha256 printed. Pushed a private kernel: lam {0.01, 0.03, 0.3, 1.0} x 3 seeds + lam 0.09 s0 re-run on T4.
- 06:38 Research agent on free GPU providers: Colab free via `google-colab-cli` (T4, headless) and Modal ($30/mo
  credit, card needed) are the next two options; Studio Lab closed to new users, GCP trial can't use GPUs.
- 06:38 frameskip-4 lam 0.09 s0: tuning set 1.00 (same 1.00 / other 1.00, 9.3 steps). Browser plan time 220 ms
  (vs 800 ms for frameskip 1), ~0.87 s with 4x CPU throttling as a phone stand-in. 8/8 goals reached in Chrome.
- 06:40 Export bug: fp16 storage broke the fs4 encoder (0.07 max error; projector BatchNorm running var ~1e-7
  amplifies upstream rounding) and the failed export had already overwritten web/model. Now atomic (temp dir) and
  fp16 is kept per graph only if ORT parity < 2e-2: fs4 s0 ships fp32 encoder (4.4 MB) + fp16 rollout (0.2 MB).
- 06:49-06:53 Test-set results (100 fixed tasks; planner settings locked from the tuning set):
  lam 0.09 frameskip 1: 1.00 / 1.00 / 1.00 success (3 seeds), 10.1-10.7 steps; frameskip 4: 1.00 x3, 9.3-9.5 steps,
  ~225 ms/replan batched; lam 0: 0.06 / 0.06 / 0.05 (other room 0.00-0.02). Probes: surprise AUROC 1.00 (fs1),
  0.997 (fs4), 0.53 (lam 0, chance); latent-vs-geodesic Spearman 0.36 (fs1), 0.50 (fs4); basin 12 px / ~15 px.
- 06:53 Training curves show val rollout error stuck at 10-100 for 2/3 fs1 seeds and 3/3 fs4 seeds until the last
  ~1500 steps of the cosine schedule (eval-mode BatchNorm with running var 1e-7..7e-5). All final models good, but
  noted as a finding/limitation in WRITEUP.md.
- 06:54 RESULTS.md, plots, web/results.json, WRITEUP.md and README filled from the logged CSVs. Demo model:
  fs4 lam 0.09 seed 0 (5.0 MB). GIF recorded (docs/demo.gif, 0.76 MB).
- 06:55 Browser planner evaluated on the 100 test tasks in headless Chrome (tests/web_eval.py: cem.js + ORT-web +
  env.js, same rule and budget): 0.99 (same 1.00, other 0.98), 9.2 steps, 172 ms/replan. The miss (task 35) has its
  goal in the doorway corner (x 33.9, y 45.1); agent ended 8.7 px away. Reported as its own row.
- 07:02 Owner OK'd a public repo + demo. Before publishing, scrubbed machine/service details and private links from
  the whole (never pushed) git history with git-filter-repo; added MIT LICENSE and the demo/repo links.
- 07:05 Public repo https://github.com/lollinng/latent-planner created and pushed (scrubbed history; full-history
  scan clean: personal noreply identity on all commits, no emails, hostnames, IPs, tokens or Kaggle credentials).
  Pages via Actions; live at https://prathameshjadhav.in/latent-planner/ (no DNS change). Live smoke test in
  headless Chrome: loads in 3.4-4.5 s, ~180 ms per plan, 4/4 goals, no console errors, no overflow at 390 px.
- 08:00 Kaggle sweep merged (13 runs, Tesla T4, torch 2.11): dataset sha256 identical to local (frames, pos, actions,
  val). Cross-hardware re-run lam 0.09 s0: 1.00 (same as local). Sweep (3 seeds each): lam 0.01 1.00, 0.03 1.00,
  0.3 0.99, 1.0 0.93 (other room 0.86). Rollout error 0.010 -> 0.057 and probe 0.26 -> 2.93 px as lam goes
  0.01 -> 1. SIGReg needs to be on but not strong; 0.09 sits in the flat region. Added to RESULTS.md/WRITEUP.md.
- 08:05 Next goal (M8, stretch): Scene 2 "Push". Kinematic axis-separated push env (env_push.py), bit-exact JS port
  (web/src/env_push.js, parity test over 181 contact/wall states), push behaviour policy + wander for data
  (2000 + 200 episodes, 83% move the block > 5 px, peak RAM 0.15 GB), eval_push.py with tasks taken from real push
  rollouts (success = block within 3 px within 60 steps). Tuning-set baselines: random 0.01, scripted pusher (true
  positions) 0.85, oracle CEM (true simulator, state-distance cost) 0.57.
- 08:18 Teammate session asked for REPORT.md and for re-verifying the live page after every push. Re-check found the
  08:07 Pages deploy had failed ("No artifacts named github-pages"; artifact listing lag), live site still on the
  previous good build. Fixed by moving to current action majors and separate build/deploy jobs; live re-tested OK.
  REPORT.md written.
- 08:24 Push fs4 lam 0.09 s0 (6000 steps): rollout error 0.10-0.20, read-out 3.6 px (agent) / 4.3 px (block) vs
  0.7 px in two-room. Tuning tasks: 0.32 (short 0.53, long 0.23). Longer horizon doesn't help (0.32 at 32 steps,
  0.27 with 8-step holds). Failures do push the block but not toward the goal (median 30 -> 31 px), so the
  imagined contact dynamics are the bottleneck. Sent 4 training variants to Kaggle (lam 0.03 / 0.09, 6k / 15k
  steps, frameskip 4 / 1).
- 08:47 Push variant lam 0.01, 15k steps (local): rollout error 0.011-0.016 but block read-out 13.7 px and tuning
  success 0.17: the model ignores the rarely-moving block. Teammate session asked to bound the push work; agreed.
- 08:50 Push test (100 tasks, 3 seeds, lam 0.09 fs4): 0.40 / 0.37 / 0.39 (short 0.74, long 0.19). Baselines:
  scripted pusher 0.87, oracle CEM 0.64, random 0.01. Kaggle push-variant kernel was still QUEUED after 30 min;
  deleted it (never ran, no GPU used). Shipped push behind the scene toggle with the 39% stated on the page;
  WRITEUP/REPORT/README updated with the contact-dynamics limitation.
- 10:11 User asked for next goals in parallel with subagents (and suggested multiple Kaggle profiles; declined again:
  Kaggle/Colab terms allow one account per person). Parallel tracks: LayerNorm projector experiment (local GPU),
  push v2 on Kaggle (2x data, 90% pushing, LayerNorm, 12k steps), and two worktree subagents. Merged:
  (1) Colab CLI runner (scripts/colab_run.py, docs/COLAB.md; needs the owner's one-time sign-in);
  (2) longer-tasks experiment: plain CEM test success 100/99/99/81% at 20/12/8/4-step horizons; subgoals from a
  latent waypoint graph (5,200 training-frame embeddings) raise 4 steps to 98% (test) and far tasks 50->83%
  (4 steps), 83->100% (8 steps). It also showed the latent cost is not flat beyond the basin (weak slope,
  7.5 -> 9.4 from 20 to 110 px), so WRITEUP finding 2, REPORT and the demo text were corrected.
  Demo: scene captions under the tabs make the two scenes' goals explicit.
- 10:44 LayerNorm projector (next step 1): two-room fs4, 3 seeds. Val 16-step error < 0.2 by step 600 (BatchNorm:
  ~5,800); final read-out 0.50-0.62 px (vs 0.80); test 0.99 / 1.00 / 0.99 (vs 1.00 x3). Encoder fp16 error 0.00045
  (BatchNorm 0.07), so fp16 storage works: demo model download 5.0 -> 2.8 MB. Browser planner with the LayerNorm
  seed-0 model: 1.00 on the 100 test tasks (BatchNorm demo model: 0.99). Swapped the two-room demo model.
  Local GPU hit 86 C and thermal-throttled (one run took 33 min), so the push LayerNorm runs moved to Kaggle.
  Push v2 on Kaggle (2x data, 90% pushing, LayerNorm, 12k steps): test 0.40 x3 (short 0.80-0.86, long 0.15-0.18)
  vs v1 0.39 (short 0.74, long 0.19): more data helps short pushes only.
- 11:06 Push LayerNorm on the same data (Kaggle; data sha256 matches local): test 0.35 / 0.28 / 0.33 vs BatchNorm
  0.40 / 0.37 / 0.39, so LayerNorm helps two-room but hurts push. Push demo keeps the BatchNorm model (39%).
  GitHub Pages is slow to this laptop right now (~140 KB/s even on cache hits; jsDelivr 1.6 MB/s); live page
  load measured 21-25 s vs 3.5-7 s earlier. Transient CDN issue, not the model change; re-check later.
- 11:53 Owner AFK for a week. Colab sign-in can't be done without the owner (Google account in a browser); left for
  later, local GPU + Kaggle cover unattended work. Model files now served from jsDelivr pinned to the deployed
  commit (deploy.json written by the Pages workflow), GitHub Pages fallback tested (bogus commit -> fallback OK).
  Live load 1.3-3.3 s (was 21-25 s). Push subgoals (waypoint graph over 13k push frames, fs4 s0, tuning tasks):
  default graph 391 components -> 0.32 (= plain); connected graphs 0.31 / 0.35 (long 0.23 / 0.27 vs 0.23):
  within noise, not run on test. Push gap is contact dynamics, not routing.
