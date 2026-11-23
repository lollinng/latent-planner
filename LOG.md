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
- 13:57 Live check of both scenes (owner asked for the live demo). Two rooms fine (load 2.5 s alone; an earlier
  20 s reading was two test browsers plus a stuck download sharing the line). Push demo had a real problem: it
  chained each new push from where the last ended (agent pressed against the block) and never gave up on a missed
  push, so it stalled: 0 of 12 chained tasks reached. Browser planner on the 100 push test tasks: 0.37 (Python,
  same model: 0.40), so the planner port is fine. Fixes: push tasks from a JS port of the test-set generator
  (makeTask; push-length distribution matches Python: median 25.9 vs 25.7 px, 66% long in both), a 2x-budget
  give-up (100 / 120 steps) with "Missed" and a running score, and the loop now waits for the new goal's
  embedding (it could plan one knot toward the old goal). Push demo loop: 31 of 64 reached (48%, 120 steps);
  browser planner on 100 JS-generated tasks: 0.30 (60 steps; within noise of 0.37). Two rooms: 8 of 8.
  Colab still not signed in (no browser tools in this session; WSL Chrome has no Google sign-in).
- 15:20 Push decomposition (tuning tasks, same CEM K5xh4 S300 I10): learned predictor + latent cost 0.32; true sim +
  true-state cost (oracle) 0.57; true sim + latent cost 0.86 (short 1.00, long 0.80; every candidate rendered and
  encoded with a pixel-exact batched torch port of the push renderer, 23k frames/s vs 0.8k). So the embedding is
  a very good cost and the learned contact dynamics are the weak part. More search: oracle 0.57 -> 0.81 (K8xh4
  S1000 I15); learned models 0.24-0.36 over 8 settings (no gain; doubling the step budget doesn't help the oracle
  either: stuck, not out of time). Bigger predictor (seed 0): 512x2 0.36, 512x3 0.40, 1024x2 0.32, 1024x3 0.39 vs
  0.32. Queued: 512x3 seeds 1-2 + test, 12-step rollout loss, frameskip 1, 3x data with 18k steps.
  Live-check workflow (GitHub runner): both scenes load in 1-2 s, no console errors; push 3 of 12 tasks there,
  34 of 76 over all demo-loop runs (45%).
- 16:20 Push follow-ups. 512x3 predictor, 3 seeds, test (running cost): 0.47 / 0.47 / 0.40 (vs 0.40 / 0.37 / 0.39).
  12-step rollout loss 0.25, frameskip 1 0.37 (tuning, seed 0; vs 0.40). Final-state cost only (running_w 0):
  same demo models 0.50 / 0.53 / 0.52 on test (long 0.34-0.35 vs 0.19), 512x3 0.54 / 0.59 / 0.52; oracle on tuning
  0.57 -> 0.70; with a 120-step budget 0.52-0.55 vs 0.38-0.41 (extra steps add 1-2 points either way). Browser
  planner with running_w 0: 0.52 on test, 0.46 on 100 JS-generated tasks (vs 0.37 / 0.30). Push demo switched to
  running_w 0 (meta.json only; two rooms keeps 1.0). Demo-loop headless runs were noisy (20 of 48 with the new
  cost vs 31 of 64 before; local page loads stalled 30-50 s on jsDelivr from this connection), so the page note
  quotes the test numbers only. Literature review with 4 research subagents: reports/World models for block
  pushing.md (notes in research_notes/). Top suggestions: retrain the predictor on the planner's own rollouts
  (queued), 2-step rollout loss + predictor dropout 0.1, more contact data, final-state cost (done).
- 00:35 (23 Nov) 3x contact data (6000 episodes, 90% scripted pushes) + 18k steps, 512x3 predictor, seed 0:
  tuning 0.45 with the running cost (1x data 512x3 seed 0: 0.40; demo model 0.32). Single seed, modest. Note the
  data seed (11) equals push_val's seed, so its validation error may be optimistic; tuning/test tasks are separate
  seeds. Laptop is on battery since the evening (AC offline, GPU capped at 10 W / 210 MHz: 9 s per replan), so the
  local DAgger queue is stopped and moved to Kaggle (lp-push-dagger): for seeds 0-2, train the demo recipe, then 3
  rounds of predictor retraining on the planner's own attempts (2000 episodes each, final-state cost, noise 0.3),
  each round scored on tuning tasks with both costs, then all rounds on test. kaggle_run.py gained raw-command
  jobs and --include; eval_push names nested run dirs uniquely.
- 01:50 Prediction-loss rollout length matters a lot for push (Kaggle lp-push-reg, tuning tasks, 3 seeds, demo
  recipe otherwise). 2-step rollout loss (vs 8): final-state cost 0.69 / 0.66 / 0.72, running cost 0.72 / 0.76 /
  0.78 (8-step: 0.40 / 0.48 / 0.41 and about 0.32). With predictor dropout 0.1 on top: 0.57 / 0.53 / 0.54 and
  0.79 / 0.77 / 0.82. Dropout alone (8-step) hurts: 0.24-0.37. Long pushes go from about 0.20-0.35 to 0.57-0.79.
  Consistent with the 12-step loss being worse (0.25). Kaggle data hashes equal the local data; push_r2_s0
  re-scored locally gives 0.71 vs 0.72 (GPU numerics). Same predictor size, so same browser speed. Queued on
  Kaggle before any test-set run: rollout 1 and 4, each with and without dropout, 3 seeds (lp-push-rollout).
- 01:55 Retraining the predictor on the planner's own attempts works (Kaggle lp-push-dagger; encoder frozen,
  2000 planner episodes per round with the final-state cost and action noise 0.3, predictor trained with the
  usual 8-step loss). Fresh demo-recipe models, tuning, final-state cost: 0.42 / 0.51 / 0.37 -> round 1 0.72 / 0.68
  / 0.74 -> round 3 0.71 / 0.73 / 0.66. Round 0 (same fine-tuning, no new data) 0.52 / 0.51 / 0.42, so the new data
  is what helps. Test, final-state cost: base 0.52 / 0.52 / 0.49, round 3 0.72 / 0.80 / 0.78 (long 0.60-0.71 vs
  0.34-0.38). The planner reaches the goal in 45-49% of collection episodes in round 1 and 64-69% by round 3.
  Per-round logs in results/push_dagger/. Next: DAgger on top of the short-rollout loss.
- 02:05 Push demo switched to the retrained model: Kaggle seed 0 after 2 rounds of predictor retraining on the
  planner's own attempts (round picked by mean tuning score across seeds, seed fixed in advance), final-state
  cost. Test, 3 seeds: 0.72 / 0.77 / 0.77 (mean 0.75; short 0.95, long 0.65) vs 0.52 for the previous demo model.
  Browser planner (JS + ONNX Runtime Web) on the 100 test tasks: 0.73 (Python 0.72). Same predictor size, so plan
  time is unchanged (0.33 s per replan in headless Chrome). Page note, WRITEUP, REPORT and RESULTS updated;
  report.py can now select retraining rounds (runs/x_s0/round2); export.py accepts retraining summaries.
- 03:30 Rollout-length sweep (Kaggle lp-push-rollout + lp-push-reg, tuning tasks, 3 seeds each; final-state /
  running cost): 8 steps 0.43 / 0.31, 4 steps 0.63 / 0.45, 2 steps 0.69 / 0.75, 1 step 0.56 / 0.82, 1 step +
  dropout 0.1 0.40 / 0.88 (2 + dropout 0.55 / 0.79, 4 + dropout 0.56 / 0.48, 8 + dropout 0.32 / 0.26). Picked
  1 step + dropout with the running cost on tuning; test: 0.87 / 0.90 / 0.87 (mean 0.88, short 0.96, long 0.84),
  level with the hand-written pusher (0.87). Browser planner, seed 0: 0.84 on the 100 test tasks (Python 0.87).
  Push demo switched to it (running_w 1 in meta.json). The cost preference flips with the loss: 8-step models want
  the final-state cost, 1-step models the running cost. RESULTS gains the sweep table; report.py lists it.
- 03:40 Short loss + retraining on the planner's attempts stack (Kaggle lp-push-dagger-r2; collection with the
  running cost, retraining with the 2-step loss). Running cost, tuning tasks (start -> round 1 / 2 / 3), 3 seeds:
  2-step 0.75 -> 0.91 / 0.90 / 0.88; 2-step + dropout 0.79 -> 0.92 / 0.92 / 0.92. Test, running cost, round 1:
  2-step 0.89 / 0.89 / 0.91, 2-step + dropout 0.93 / 0.93 / 0.93 (fresh models 0.85 / 0.84 / 0.79; round 0,
  fine-tuning without new data, 0.86 / 0.86 / 0.80). Waiting for the 1-step + dropout version (lp-push-dagger-r1)
  before choosing a demo model by tuning score.
- 04:20 1-step + dropout + retraining on the planner's attempts (Kaggle lp-push-dagger-r1, running cost). Tuning:
  start 0.85 / 0.86 / 0.87, round 0 (no new data) 0.95 / 0.92 / 0.88, round 1 0.94 / 0.92 / 0.93, round 2 0.91 /
  0.92 / 0.91. Test: start 0.84 / 0.85 / 0.84, round 0 0.88 / 0.94 / 0.91, round 1 0.91 / 0.91 / 0.90, round 2 0.92 /
  0.96 / 0.94. Chosen by mean tuning score over the combined recipes (1-step + dropout round 1: 0.930; 2-step +
  dropout round 1: 0.923; 2-step round 1: 0.907), seed 0 fixed in advance. Browser planner on the 100 test tasks:
  0.90 (Python 0.91). Push demo switched to it. Caveats in WRITEUP: for 1-step models most of the retraining gain is
  the extra training (control 0.91), and the same recipe re-run on the same hardware moved 0.88 -> 0.84.
- 05:45 Push failure analysis (owner's question: can the block get stuck?). Yes: push-only, so a block within 6 px
  of a wall can never come away from it. `scripts/push_deadends.py` on the saved test trajectories: 15 of 28 demo
  recipe failures (3 seeds) are such dead ends, mostly the block shoved against the wrong wall around step 11; the
  rest are near misses (~8 px). Hand-written pusher and true-simulator CEM: 0 dead ends. Also found: 94/100 test
  goals have the block against a wall (30 in a corner). New `--taskset interior` (seed 5050, 100 goals in the open,
  12-step pushes): demo recipe 0.91 / 0.91 / 0.93, hand-written pusher 1.00, oracle CEM 0.94, random 0.00. Ran
  locally on battery (6 min). WRITEUP and REPORT updated.
