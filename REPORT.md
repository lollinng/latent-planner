# LatentPlanner: report

**Live demo:** https://prathameshjadhav.in/latent-planner/ · **Code:** https://github.com/lollinng/latent-planner
(public, MIT) · Technical write-up: [WRITEUP.md](WRITEUP.md) · All numbers: [results/RESULTS.md](results/RESULTS.md)

## What was built

A small world model that learned a two-room world from pixels and plans routes in its own embedding space,
live in the browser:

- **Environment:** our own 2-D two-room world, written in Python for training and ported to JS for the browser.
  The port is bit-exact (same positions, same pixels; tested), so the browser shows the model the same kind of
  frames it trained on.
- **Model:** a 1.1M-parameter CNN encoder (64×64 frame → 32 numbers) and a 93k-parameter predictor (embedding +
  action → next embedding). Trained with the LeWM recipe: next-embedding prediction plus SIGReg, with no
  decoder, no EMA target and no stop-gradient. No rewards, positions or map are ever given.
- **Planner:** the cross-entropy method (CEM) in embedding space toward the embedding of a goal image. It
  re-plans every 4 steps.
- **Browser demo:** ONNX Runtime Web (wasm). Tap to set a goal, drag the agent to "surprise" it, and toggle the
  planner's cost map. It works on phones.

## Headline results (100 fixed test tasks, half need the door, 3 training seeds each)

| | Success | Other room |
|---|---|---|
| Latent planner (λ = 0.09) | **100% on every seed** | 100% |
| Frameskip-4 model (the one in the browser) | **100% on every seed** | 100% |
| The browser's own planner (JS + ONNX Runtime Web, LayerNorm demo model), run headless on the same tasks | 100% | 100% |
| Without SIGReg (λ = 0): the embedding collapses | 6% | 1% |
| Oracle: same planner, true simulator, straight-line cost | 87% | 74% |
| Straight line to goal / random actions | 70% / 11% | 40% / 2% |

- **SIGReg weight sweep:** λ = 0 collapses (6%); every λ from 0.01 to 0.09 reaches 100%; 0.3 gives 99% and 1.0
  gives 93%. SIGReg has to be on, but too strong a weight hurts prediction.
- **Reproducibility:** the λ = 0.09 seed-0 model retrained on a Kaggle T4 also scores 100%. The dataset
  regenerated there is byte-identical to the local one (sha256 match).
- **Load time:** model files now come from jsDelivr pinned to the deployed commit, with GitHub Pages as the
  fallback. The live page loads in 1.3-3.3 s, down from 21-25 s while GitHub Pages was slow to this laptop.
- **Latency:** ~180-220 ms per re-plan in Chrome on this laptop with the frameskip-4 model (vs ~800 ms with
  frameskip 1), and ~0.9 s with 4× CPU throttling as a phone stand-in. The live page loads in 3.4-7 s from this
  connection; the download is 2.8 MB of model (5.0 MB before the LayerNorm swap) plus 3.2 MB of ONNX Runtime.
- **Surprise:** the predictor's error separates real next frames from teleports with AUROC 1.00, which drives the
  demo's surprise meter. The collapsed models are at chance (0.53).

## Findings worth knowing

1. **The latent cost is a basin, not a map.** Latent distance to the goal is a sharp ~12 px basin with only a weak
   slope beyond it; it doesn't trace the rooms. The planner gets through the door because the learned dynamics
   stop plans at walls and its lookahead reaches the basin. The weak far field also avoids the trap that catches the
   straight-line oracle (hugging the wall next to a goal in the other room). An earlier version of this report said
   the far field was flat; the horizon experiment showed a weak slope that CEM can follow.
2. **Final-state-only cost fails under re-planning.** The agent kept pushing the goal to the end of a fresh
   window. A running-cost term fixed it; we found this with the oracle first.
3. **Training only settles late.** Validation rollouts stay unstable until the learning rate decays in the last
   ~1,500 of 6,000 steps, for 5 of the 6 SIGReg runs. The likely cause is the projector BatchNorm, which sees
   features with variance ~1e-7. Every final model is good, but the result leans on the schedule.
4. **Effective rank is a bad collapse detector here.** It reads ~31/32 for collapsed models (it ignores scale);
   embedding std (0.0002 vs 0.96) is the honest signal.

## Time and compute

- **Wall-clock:** about 3 hours. Session 1 (05:22-~05:50) did the research, the machine check and the first
  code, and was OOM-killed. Session 2 resumed at 05:52 and fixed the OOM first (data generation peak RAM
  9.2 GB → 0.15 GB). It reached test results by 06:53, the public deploy by 07:05, and the merged sweep by 08:00.
- **Local GPU (RTX 3050 Laptop, 4 GB):** 9 main models at ~5-6 min each (about 50 min), test evaluations of about
  1.5 min each, plus 4 push models (3 × 5 min, plus one 13-min variant). Host RAM peaked at 1.8 GB and VRAM at 2.6 GB.
- **Kaggle (free T4, private kernel):** one kernel, about 84 min of GPU time, for 13 runs (λ sweep plus the
  cross-hardware re-run) including data generation, evaluation and probes. That is about 1.4 of the ~30 free
  GPU-hours per week.
- **Cost:** $0. No paid cloud, no DNS change, no opened ports. The repo went public only after your OK.
- **Version control:** 30+ small commits. Before the first push, the history was scrubbed of machine details and
  private links, and a full-history scan for secrets, emails and hostnames came back clean.

## LayerNorm projector (next step 1, done)

Swapping the projector's BatchNorm for LayerNorm fixed the late-settling training: validation error drops below
0.2 by step 600, against ~5,800 before. Test success is 99/100/99% (vs 100/100/100%). The encoder now survives
fp16 weights, so the demo runs the LayerNorm model. Its browser planner scores 100% on the test tasks, and the
model download fell from 5.0 to 2.8 MB.

## Longer tasks: subgoals (next step 2, done)

A subagent measured the horizon limit and added subgoal planning from a graph of training-frame embeddings
(embeddings only, no positions). Test success at the shortest horizon (4 steps) goes from 81% to **98%**. On 100
extra long cross-room tasks it goes from 50% to 83% at 4 steps and from 83% to **100%** at 8 steps. At the default
20 steps nothing changes, because the goal is already in reach. Details in WRITEUP finding 8 and
`results/horizon/SUBGOAL.md`.

## Scene 2: Push (stretch goal, shipped behind a toggle)

- A second scene where the agent pushes a block to a target. Same model, objective and planner, with its own data,
  bit-exact JS port and test set.
- **Test result (3 seeds): 52% ± 1%** (84% short pushes, 34% long), up from 39% (74% / 19%) by one planner change:
  CEM now scores only the final imagined state. The running cost, needed in two rooms, penalised the detour a long
  push needs (walking away from the target to get behind the block). Same model, no retraining; the browser
  planner scores 52% on the test tasks. For comparison: a hand-written pusher using true positions gets 87%, the
  oracle CEM (true simulator) 64%, random actions 1%.
- **Where it loses:** with the true simulator and the model's own latent cost, the same planner reaches 86% on the
  tuning tasks, so the embedding is a good cost and the learned contact dynamics are the weak part. More search
  doesn't help the learned model (it does help the true simulator: 57% → 81%), a sign the planner exploits model
  errors. A 512×3 predictor adds a little (55% with the final-state cost) at 2.5× the plan time, so the demo keeps
  the small one.
- **How others do it:** a literature review (`reports/World models for block pushing.md`) found published PushT
  results use ~40× our data, short prediction losses, predictor dropout and final-state costs, and that
  retraining the predictor on the planner's own rollouts is the best-supported fix for model exploitation.
- **Why it's weaker:** the model's imagined contact dynamics are inaccurate. Prediction error is ~4× the two-room
  model's, and the block position is encoded only to ~4 px. A longer planning horizon didn't help.
  Weaker SIGReg (λ = 0.01) made the model ignore the block: good prediction loss, useless for pushing (17%).
- Later variants on the Kaggle T4: a LayerNorm projector on the same data scored 32%. Twice the data (90% pushing)
  with LayerNorm and 12k steps scored 40%: short pushes 83%, long ones still 17%. Subgoals from a waypoint graph
  of push frames didn't help either (31-35% vs 32% on the tuning tasks). The demo keeps the 39% model and states
  that number on the page.
- **The push demo now behaves like the test.** Its browser planner scores 37% on the 100 test pushes (Python, same
  model: 40%). Before this fix the demo chained each new push from wherever the last one ended, with the agent
  pressed against the block, and when a push failed it kept trying forever: 0 of 12 chained tasks were reached in
  a headless run. New tasks now come from the test-set generator (ported to JS; same distribution), and a task ends
  after 120 steps with "Missed" and a running score. Measured in the real demo loop: 31 of 64 reached (48%; 120
  steps vs the test's 60).

## Known limitations

- The two-room world is tiny, fully observed and deterministic: one frame is the whole state, and there are no
  distractors. It shows the recipe working end to end, not that it scales.
- With a short lookahead, long tasks need subgoals (see "Longer tasks" below); the waypoint graph relies on a dense
  bank of training frames, easy in a world this small.
- One planning seed per trained model; the error bars are training variance only.
- The demo runs one model per scene (seed 0); the evaluation numbers cover all three seeds.

## Suggested next steps

1. ~~LayerNorm projector~~: done. Training is stable ~10× earlier, fp16 now works, and the demo download fell
   from 5.0 to 2.8 MB. Test success is 99-100% (see WRITEUP finding 9).
2. ~~Longer tasks~~: done with subgoals from a latent waypoint graph (see above). Next would be a learned value
   function instead of a frame bank, for worlds too big to cover densely.
3. Harder scenes: distractors, partial observability or stochastic dynamics, where pixel-free prediction should
   matter more.
4. Use WebGPU when available, for more CEM samples per re-plan on phones.
