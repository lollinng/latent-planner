# LatentPlanner plan

Goal: a public page where a small JEPA world model, trained by us from pixels with the LeWM recipe
(next-embedding prediction + SIGReg, no EMA, no stop-grad), plans live in the browser with CEM.

## Key decisions (from Phase 0/1)
- **Environment for the demo: our own 2-D env** written twice (NumPy for training, JS for the browser) with a
  shared deterministic renderer, so the browser sees exactly the pixels the model was trained on. Push-T runs on
  pymunk physics that can't be reproduced exactly in JS, and LeWM-size training at 224 px is slow on a 4 GB GPU.
- **Scene 1 "Two rooms"** (navigate through a door; same task family as PLDM / EB-JEPA / LeWM Two-Room).
  **Scene 2 "Push"** (agent pushes a block) is a stretch goal once scene 1 works end to end.
- **Model**: CNN encoder (64x64 RGB) + BatchNorm projector -> z (D=32..64); MLP action-conditioned predictor
  (one frame of history is enough, the env is Markov in the frame). Loss = MSE(pred, next z) + lambda * SIGReg.
  SIGReg written from the LeJEPA paper and checked against LeWM's MIT implementation (the `lejepa` package is
  CC BY-NC, so we don't vendor it).
- **Planner**: CEM in latent space, receding-horizon MPC, cost = distance to goal latent. In the browser the
  predictor is exported with the whole horizon unrolled, so one ONNX call scores all samples for one CEM iteration.
- **Deploy**: GitHub Pages project site in `lollinng` -> `https://prathameshjadhav.in/latent-planner/` (no DNS
  change; needs the repo public -> ask first).

## Milestones (estimates are wall-clock for me)
| # | Milestone | Est. |
|---|---|---|
| M0 | Research notes, server check, plan | 1.5 h |
| M1 | Env in NumPy + matching torch/JS renderers, parity test, data generator | 2 h |
| M2 | Model + SIGReg + training loop, first run, sanity plots (loss, eff. rank) | 2 h |
| M3 | CEM planner + eval harness (success rate, baselines: random, greedy, oracle-CEM on true sim) | 2 h |
| M4 | Experiments: 3 seeds lambda>0, 3 seeds lambda=0, (pixel-recon baseline if cheap), surprise probe, latency | 3 h |
| M5 | Post-hoc decoder + linear position probe (visualisation only), ONNX export + parity test | 1.5 h |
| M6 | Browser demo: env, ORT-web, CEM, imagined-trajectory overlay, PCA panel, surprise meter, stats | 5 h |
| M7 | Mobile testing (Chrome emulation, WebKit via Playwright), load-time budget | 1.5 h |
| M8 | Scene 2 "Push" (stretch) | 3 h |
| M9 | README, WRITEUP, GIF, REPORT, repo hygiene, deploy (after OK) | 3 h |

## Evaluation protocol (all numbers from logged runs)
- 3 training seeds per config. Each model evaluated on the same 100 fixed (start, goal) tasks per seed.
- Tasks: start and goal sampled uniformly in free space, half of them in opposite rooms (needs the door).
- Success: agent centre within 4 px of the goal centre within the step budget.
- Report mean +- std over seeds. Logs as CSV in `results/`, plots from those CSVs.

## Ask-first checkpoints
- Making the GitHub repo public / enabling Pages.
- Any DNS change (should not be needed).
- Uploading weights to the HF Hub (would publish under the user's account).
