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
| The browser's own planner (JS + ONNX Runtime Web), run headless on the same tasks | 99% | 98% |
| Without SIGReg (λ = 0): the embedding collapses | 6% | 1% |
| Oracle: same planner, true simulator, straight-line cost | 87% | 74% |
| Straight line to goal / random actions | 70% / 11% | 40% / 2% |

- **SIGReg weight sweep:** λ = 0 collapses (6%); every λ from 0.01 to 0.09 reaches 100%; 0.3 gives 99% and 1.0
  gives 93%. SIGReg has to be on, but too strong a weight hurts prediction.
- **Reproducibility:** the λ = 0.09 seed-0 model retrained on a Kaggle T4 also scores 100%. The dataset
  regenerated there is byte-identical to the local one (sha256 match).
- **Latency:** ~180-220 ms per re-plan in Chrome on this laptop with the frameskip-4 model (vs ~800 ms with
  frameskip 1), and ~0.9 s with 4× CPU throttling as a phone stand-in. The live page loads in 3.4-7 s from this
  connection; the download is 5.0 MB of model plus 3.2 MB of ONNX Runtime.
- **Surprise:** the predictor's error separates real next frames from teleports with AUROC 1.00, which drives the
  demo's surprise meter. The collapsed models are at chance (0.53).

## Findings worth knowing

1. **The latent cost is a basin, not a map.** Latent distance to the goal is a sharp ~12 px basin and flat
   elsewhere; it doesn't trace the rooms. The planner gets through the door because its 20-step lookahead is
   long enough to find the basin, and the flat far field avoids the trap that catches the straight-line oracle
   (hugging the wall next to a goal in the other room).
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
  1.5 min each, plus the push-scene runs. Host RAM peaked at 1.8 GB and VRAM at 2.6 GB.
- **Kaggle (free T4, private kernel):** one kernel, about 84 min of GPU time, for 13 runs (λ sweep plus the
  cross-hardware re-run) including data generation, evaluation and probes. That is about 1.4 of the ~30 free
  GPU-hours per week.
- **Cost:** $0. No paid cloud, no DNS change, no opened ports. The repo went public only after your OK.
- **Version control:** 30+ small commits. Before the first push, the history was scrubbed of machine details and
  private links, and a full-history scan for secrets, emails and hostnames came back clean.

## Scene 2: Push (in progress)

<!-- PUSH -->

## Known limitations

- The two-room world is tiny, fully observed and deterministic: one frame is the whole state, and there are no
  distractors. It shows the recipe working end to end, not that it scales.
- Goals must be within the planner's 20-step lookahead, because the cost gives no signal beyond the basin.
- One planning seed per trained model; the error bars are training variance only.
- The demo runs one model (frameskip 4, seed 0). Its fp32 encoder (fp16 weights broke it) makes the download
  5 MB instead of ~2.6 MB.

## Suggested next steps

1. Replace the projector's BatchNorm with LayerNorm (or lower the peak learning rate) and check whether training
   stabilises early. This would also let fp16 weights halve the download.
2. Longer tasks: hierarchical planning or a learned value on top of the latent cost, to reach goals beyond the
   horizon.
3. Harder scenes: distractors, partial observability or stochastic dynamics, where pixel-free prediction should
   matter more.
4. Use WebGPU when available, for more CEM samples per re-plan on phones.
