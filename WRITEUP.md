# LatentPlanner: a JEPA world model that plans in the browser

## Summary

We trained a small world model (1.1M-parameter CNN encoder, 93k-parameter MLP predictor) on 98k pixel frames
from a two-room environment, with the LeWM objective: predict the next embedding, plus SIGReg to keep the
embeddings from collapsing, with no EMA target, no stop-gradient and no pixel reconstruction. Planning is CEM in
embedding space toward the embedding of a goal image. The model never sees positions, rewards or a map.

On 100 fixed test tasks, half of which need the door, the planner reaches the goal in **100% of tasks for all
three training seeds**, both with one-step prediction and with 4-step frameskip (the browser model). The same
planner given the true simulator and straight-line distance reaches 87% (74% on other-room tasks). Without SIGReg
(λ = 0) the embedding collapses and success drops to 6%, below random actions (11%).

Everything runs on a 4 GB laptop GPU in about 5 minutes per model, and the planner runs in the browser with ONNX
Runtime Web.

A second scene, pushing a block onto a target, is harder: the same recipe gets 39%, rising to 52% with a
final-state-only planning cost, to 88% by training the predictor on one-step predictions instead of 8-step
rollouts, and to 91% (3 seeds; 90% in the browser) after retraining the predictor on the planner's own attempts,
above a hand-written pusher that uses true positions (87%).

## Setup

**Environment.** A 64×64 arena with a 2 px border and a vertical wall with a 14 px door. The agent is a radius-3
disc; actions in [-1, 1]² move it up to 5 px per step (5 substeps, axis-separated collision against
axis-aligned walls). The renderer anti-aliases the disc with 4×4 supersampling. The environment is written twice,
NumPy for training and JS for the browser, with the same float64 operations in the same order: positions match
exactly and frames match pixel for pixel (`tests/test_js_parity.py`), so the browser shows the model exactly the
kind of frames it was trained on.

**Data.** 2,000 training and 200 validation episodes of 48 steps from two behaviour policies: a sticky random walk
(60%) and a noisy go-to-random-waypoint controller that routes through the door (40%). 56% of episodes cross the
door. The policies' targets are never shown to the model. Generation streams frames to disk in chunks (peak RAM
0.15 GB; see "Engineering notes").

**Model.** Encoder: four stride-2 4×4 convolutions (32-64-128-128 channels, GELU), a linear layer to 256, and a
LeWM-style projector (Linear-BatchNorm-GELU-Linear) to a 32-d embedding z. Predictor:
z_{t+k} = z_t + MLP([z_t, emb(a_t..a_{t+k-1})]) with two 256-unit LayerNorm-GELU layers, where k is the frameskip
(1 or 4).

**Objective.** loss = mean over one-step and open-loop multi-step (up to 8 predictor steps) prediction errors
‖ẑ − z‖² + λ·SIGReg(z). Gradients flow into both sides of the prediction loss; SIGReg alone prevents collapse.
SIGReg follows the LeJEPA paper: project the batch of embeddings onto 512 random unit directions and compare each
1-D empirical characteristic function with N(0, 1)'s using the Epps–Pulley statistic (17 trapezoid knots on
[0, 3], Gaussian window), applied per time step over the batch. We wrote it from the paper and checked it against
LeWM's MIT implementation; the `lejepa` package is CC BY-NC, so it isn't used.

**Training.** 6,000 steps, batch 64 windows of 9 frames (frameskip 1) or 9 frames spaced 4 apart (frameskip 4),
AdamW (lr 1e-3, cosine, 200 warm-up steps, weight decay 1e-4), gradient clipping 1.0. The uint8 dataset lives on
the GPU (1.2 GB). 5 minutes per run on an RTX 3050 Laptop; host RAM peak 1.8 GB, GPU peak 2.6 GB.

**Planner.** CEM over K = 5 action knots, each held for 4 env steps (20 env steps of lookahead, enough for the
longest path through the door), 300 samples, 30 elites, 10 iterations, momentum 0.1. Cost of a plan:
‖z_H − z_goal‖² + 1.0 · mean_t ‖z_t − z_goal‖² over the imagined latents. Model-predictive control: execute the
first knot (4 steps), re-encode the new frame, re-plan, warm-starting from the shifted previous plan.

**Evaluation.** 100 fixed test tasks (seeded, saved in `results/tasks_test.json`); half start and end in
different rooms. Success: agent centre within 4 px of the goal centre at any step within 50 env steps. All
planner settings were chosen on a separate set of 100 tuning tasks. Three training seeds per configuration, one
planning seed each. Baselines: random actions; a straight line to the goal using the true positions; and oracle
CEM, the same CEM with the true simulator and the true straight-line distance as cost.

## Results

### Goal reaching (100 fixed test tasks, 50-step budget, success = within 4 px)

| Method | Seeds | Success | Same room | Other room | Steps (successes) | Per-seed success |
|---|---|---|---|---|---|---|
| Latent CEM, LeWM-style (λ=0.09) | 3 | 100 ± 0% | 100 ± 0% | 100 ± 0% | 10.3 | 1.00, 1.00, 1.00 |
| Latent CEM, frameskip 4 (λ=0.09) | 3 | 100 ± 0% | 100 ± 0% | 100 ± 0% | 9.4 | 1.00, 1.00, 1.00 |
| Latent CEM, frameskip 4, LayerNorm projector | 3 | 99 ± 0% | 100 ± 0% | 99 ± 1% | 9.4 | 0.99, 1.00, 0.99 |
| Same planner in the browser (JS + ONNX Runtime Web, ln_fs4_lam0.09_s0) | 1 | 100% | 100% | 100% | 9.8 | 1.00 |
| Latent CEM, no SIGReg (λ=0) | 3 | 6 ± 0% | 10 ± 2% | 1 ± 1% | 16.2 | 0.06, 0.06, 0.05 |
| Oracle CEM: true simulator, straight-line cost | - | 87% | 100% | 74% | 6.8 | - |
| Straight line to goal (true positions) | - | 70% | 100% | 40% | 5.6 | - |
| Random actions | - | 11% | 20% | 2% | 28.2 | - |

Mean ± std over training seeds (one planning seed each). Baselines run once (seeded).

"Steps" averages over each method's own successes, so it isn't comparable across rows with different success
rates (the oracle and straight-line baselines mostly succeed on the easier tasks).

![Success by method](results/plots/success.png)

### Probes (validation data; diagnostics, never used by the planner)

| Config | 16-step rollout error | Embedding std | Effective rank (of 32) | Position probe RMSE (px) | Surprise AUROC | Latent vs geodesic ρ | Latent vs straight-line ρ | Cost basin radius (px) |
|---|---|---|---|---|---|---|---|---|
| Latent CEM, LeWM-style (λ=0.09) | 0.021 ± 0.002 | 0.96 | 23.7 ± 0.3 | 0.77 ± 0.15 | 1.000 ± 0.000 | 0.36 ± 0.01 | 0.35 ± 0.01 | 12.0 ± 0.0 |
| Latent CEM, frameskip 4 (λ=0.09) | 0.047 ± 0.000 | 0.93 | 16.7 ± 0.1 | 0.81 ± 0.03 | 0.997 ± 0.000 | 0.50 ± 0.01 | 0.52 ± 0.01 | 14.7 ± 0.9 |
| Latent CEM, no SIGReg (λ=0) | 4.888 ± 0.997 | 0.0002 | 30.6 ± 0.3 | 21.33 ± 0.90 | 0.533 ± 0.026 | 0.17 ± 0.03 | 0.17 ± 0.03 | 2.0 ± 0.0 |

Rollout error is normalised by the latent variance (1.0 = no better than the mean embedding). Surprise AUROC separates real next frames from teleports using the prediction error. Effective rank is scale-invariant, so it stays high for a collapsed model (tiny isotropic noise); embedding std shows the collapse.

![Training curves](results/plots/training.png)

## Findings

**1. A final-state cost doesn't work under receding-horizon control.** With only ‖z_H − z_goal‖², every re-plan
puts the goal at the end of a fresh 20-step window, so the agent drifts and rarely passes within 4 px. The oracle
with true dynamics showed it first (24% success on the tuning tasks). Adding the running term fixed it for both
the oracle and the latent planner.

**2. The latent cost is not a map, and that turns out to help.** Latent distance correlates only weakly with
geodesic or straight-line distance (Spearman ρ in the table above). Around any goal it forms a sharp basin about
12 px wide; beyond it the distance keeps rising, but only slowly. For the frameskip-4 model the median latent
distance goes from 1.6 at 0-4 px of path to 7.5 at 16-20 px, then only to 9.4 at 80-110 px
(`results/horizon/latent_vs_px.txt`). The planner crosses rooms because the learned dynamics stop plans at walls
and the 20-step horizon reaches the basin; with shorter horizons CEM can still follow the weak far-field slope
(see finding 8). The weak far field also avoids the trap that catches the straight-line oracle: pressing against
the wall right next to a goal in the other room earns little in latent space, while in true straight-line distance
it looks almost as good as arriving. (An earlier draft called the far field flat and said goals beyond the
horizon give no signal; the horizon experiment below showed that was too strong.) The demo's "cost map" toggle
shows this landscape for whatever goal you click.

**3. Without SIGReg the embedding collapses.** With λ = 0 every seed's embedding shrinks to a standard deviation
of ~3e-4 (vs ~0.95), the prediction loss falls to ~2e-7 because predicting a constant is trivial, and a linear
probe can't recover position (20-22 px RMSE, chance level). Effective rank is misleading here: it reads ~31/32
for the collapsed models because it is scale-invariant and the leftover noise is isotropic.

**4. Surprise is a reliable signal.** The predictor's error on the real next frame is ~600× smaller than its
error against a frame from a random other state (median 0.003 vs 1.9 for frameskip 1), so a teleport is
detected with AUROC 1.00 (0.997 for frameskip 4). The collapsed λ = 0 models are at chance (0.53). The demo's surprise meter plots this error live; dragging the agent makes it spike.

**5. Validation rollouts are unstable until the learning rate decays.** For seed 0 of the frameskip-1 model the
16-step validation error falls early; for the other two seeds and all three frameskip-4 seeds it stays at 10-100×
the latent variance for most of training and only drops in the last ~1,500 steps of the cosine schedule (see the
training curves). Every final model is good, but the result leans on the schedule. Validation runs the encoder in
eval mode, and the projector's BatchNorm sees pre-normalisation features with running variance of only 1e-7 to
7e-5, so small drifts in its running statistics get amplified; the same amplification is why fp16 weights break
the frameskip-4 encoder. Replacing that BatchNorm with LayerNorm fixed both (finding 9).

**6. Frameskip 4 is the browser model.** Predicting 4 env steps at a time (LeWM uses 5) cuts the predictor calls
per plan by 4×. Planning in Chrome on this laptop drops from ~800 ms to ~220 ms per re-plan (~0.9 s with 4× CPU
throttling as a phone stand-in) with the same CEM budget, and test success stays at 100%. Its latent space is
also somewhat more map-like (ρ ≈ 0.50 vs 0.36, basin ~15 px vs 12 px).

**7. SIGReg needs to be on, but not strong.** A sweep of the SIGReg weight (3 seeds each, the extra runs on a
free Kaggle T4) shows a cliff and then a slow decline: λ = 0 collapses (6%), every λ from 0.01 to 0.09 reaches 100%,
and larger weights make the embedding fit the Gaussian target at the expense of prediction: 16-step rollout error
grows from 0.010 (λ = 0.01) to 0.057 (λ = 1), the position read-out degrades from 0.26 to 2.93 px, and success
drops to 99% (λ = 0.3) and 93% (λ = 1, 86% on other-room tasks). The LeWM default of 0.09 sits in the flat region;
on this environment 0.01-0.03 is slightly better on every probe and gives shorter paths (8.6-9.1 vs 10.3 steps).

### SIGReg weight sweep (frameskip 1, same test tasks)

| λ | Seeds | Success | Other room | Embedding std | 16-step rollout error | Position probe RMSE (px) | Surprise AUROC | Trained on |
|---|---|---|---|---|---|---|---|---|
| 0 | 3 | 6 ± 0% | 1 ± 1% | 0.0002 | 4.888 | 21.33 | 0.533 | GeForce RTX 3050 Laptop GPU |
| 0.01 | 3 | 100 ± 0% | 100 ± 0% | 0.9187 | 0.010 | 0.26 | 1.000 | Tesla T4 |
| 0.03 | 3 | 100 ± 0% | 100 ± 0% | 0.9421 | 0.014 | 0.39 | 1.000 | Tesla T4 |
| 0.09 | 3 | 100 ± 0% | 100 ± 0% | 0.9614 | 0.021 | 0.77 | 1.000 | GeForce RTX 3050 Laptop GPU |
| 0.3 | 3 | 99 ± 1% | 99 ± 1% | 0.9776 | 0.038 | 1.64 | 1.000 | Tesla T4 |
| 1 | 3 | 93 ± 2% | 86 ± 3% | 0.9876 | 0.057 | 2.93 | 1.000 | Tesla T4 |

λ = 0.09 is the LeWM default used for the main results. Runs other than λ ∈ {0, 0.09} were trained and evaluated on a free Kaggle T4 from the same seeds and code (dataset regenerated there; sha256 matches the local data).

### Cross-hardware re-run

λ = 0.09, seed 0, retrained on a Kaggle T4 with the same code and data: success 100% (same room 100%, other room 100%), 10.2 steps; the local RTX 3050 run of the same seed scored 100%.

![Success vs SIGReg weight](results/plots/sweep.png)

**8. Longer tasks: subgoals from a latent waypoint graph.** Shrinking the planning horizon shows where it breaks:
with frameskip-4 models (3 seeds) test success is 100% at 20 steps, 99% at 12 and 8 steps, and 81% at 4 steps. On
100 extra cross-room tasks with paths of at least 70 px ("far") it falls to 83% at 8 steps and 50% at 4. The fix
uses only embeddings: a graph over 5,200 embeddings of training frames (edges between consecutive frames, plus
near neighbours under a latent-distance threshold calibrated on those transitions), a shortest-path search from
the current embedding to the goal's, and CEM aimed at the first node at or beyond one horizon along that path.

| Tasks, horizon | Plain CEM | With subgoals |
|---|---|---|
| test, 4 steps | 81% | **98%** |
| far, 4 steps | 50% | **83%** |
| far, 8 steps | 83% | **100%** |
| test or far, 20 steps | 99-100% | 99-100% (unchanged) |

How the subgoal is chosen matters more than the graph: aiming at the farthest node *within* one horizon often
picked a node ~1 px away, the agent stalled, and success fell to 2% on far tasks at 4 steps. The "beyond" rule was
chosen on the tuning tasks; the far task set was designed after seeing the short-horizon numbers. Details:
[results/horizon/SUBGOAL.md](results/horizon/SUBGOAL.md).

**9. A LayerNorm projector fixes the late-settling training.** Swapping the projector's BatchNorm for LayerNorm
(frameskip 4, 3 seeds, everything else equal) makes the 16-step validation error drop below 0.2 by step 600,
against step ~5,800 with BatchNorm (0.11 vs 25 at step 1,000). The final models are slightly sharper (read-out
0.50-0.62 px vs 0.80 px; rollout error 0.043-0.046 vs 0.047), and test success is 99/100/99% vs 100/100/100%:
two misses in 300 tasks, within noise. Its encoder also survives fp16 weight storage (error 0.00045 vs 0.07),
which roughly halves the browser download, so the demo now runs the LayerNorm model (seed 0): its browser planner
scores 100% on the test tasks with 2.8 MB of model files, against 99% and 5.0 MB for the BatchNorm model it
replaced.

## Scene 2: Push (stretch goal, works)

The second scene is a walled room with an 8×8 px block that the agent pushes. Pushing is kinematic and
axis-separated: the block moves with the agent and stops at walls, with no momentum or rotation. Like the
two-room env it is ported to JS bit-exactly. The data is 2,000 episodes (70% from a policy that walks behind the
block and pushes it in random directions, 30% sticky random walk), and the model, objective, training and planner
are unchanged (frameskip 4, λ = 0.09, 3 seeds). Each test goal is the end state of a real 30-step push, so the goal
image shows the agent where a pusher would be.

### Test results (100 fixed test tasks, 60-step budget, success = block within 3 px)

| Method | Seeds | Success | Short push (< 20 px) | Long push | Steps (successes) | Per-seed |
|---|---|---|---|---|---|---|
| 1-step loss + dropout, predictor retrained on 1 round of the planner's own attempts, running cost: in the demo | 3 | 91 ± 0% | 98 ± 1% | 87 ± 1% | 18.0 | 0.91, 0.91, 0.90 |
| Browser planner (JS + ONNX Runtime Web, seed 0, 1 round), running cost | 1 | 90% | 100% | 85% | 17.6 | 0.90 (Python: 0.91) |
| Same, 2 rounds | 3 | 94 ± 2% | 99 ± 1% | 91 ± 2% | 17.8 | 0.92, 0.96, 0.94 |
| Same fine-tuning without new data (control) | 3 | 91 ± 2% | 97 ± 2% | 88 ± 3% | 19.2 | 0.88, 0.94, 0.91 |
| Starting models: 1-step loss + dropout (Kaggle), running cost | 3 | 84 ± 0% | 97 ± 2% | 77 ± 1% | 18.9 | 0.84, 0.85, 0.84 |
| 2-step loss + dropout, retrained on 1 round, running cost | 3 | 93 ± 0% | 99 ± 1% | 90 ± 1% | 17.6 | 0.93, 0.93, 0.93 |
| 2-step loss, retrained on 1 round, running cost | 3 | 90 ± 1% | 100 ± 0% | 84 ± 1% | 16.4 | 0.89, 0.89, 0.91 |
| 1-step loss + dropout, running cost (no retraining) | 3 | 88 ± 1% | 96 ± 1% | 84 ± 2% | 19.4 | 0.87, 0.90, 0.87 |
| Browser planner (seed 0), running cost | 1 | 84% | 97% | 77% | 19.1 | 0.84 (Python: 0.87) |
| 8-step loss, predictor retrained on 2 rounds of the planner's own attempts, final-state cost | 3 | 75 ± 2% | 95 ± 1% | 65 ± 3% | 19.8 | 0.72, 0.77, 0.77 |
| Browser planner (seed 0, 2 rounds), final-state cost | 1 | 73% | 94% | 62% | 18.8 | 0.73 (Python: 0.72) |
| Same, 3 rounds | 3 | 77 ± 3% | 96 ± 1% | 66 ± 5% | 20.0 | 0.72, 0.80, 0.78 |
| Same fine-tuning without new data (control) | 3 | 55 ± 1% | 85 ± 3% | 38 ± 2% | 17.0 | 0.53, 0.55, 0.56 |
| Starting models (8-step loss, Kaggle), final-state cost | 3 | 51 ± 1% | 79 ± 3% | 36 ± 2% | 16.6 | 0.52, 0.52, 0.49 |
| First models (8-step loss), final-state cost | 3 | 52 ± 1% | 84 ± 4% | 34 ± 1% | 15.4 | 0.50, 0.53, 0.52 |
| Browser planner, push_fs4_s0, final-state cost | 1 | 52% | 74% | 40% | 17.0 | 0.52 (Python: 0.50) |
| Same models, running cost | 3 | 39 ± 1% | 74 ± 0% | 19 ± 2% | 11.2 | 0.40, 0.37, 0.39 |
| Browser planner, push_fs4_s0, running cost | 1 | 37% | 71% | 18% | 10.9 | 0.37 (Python: 0.40) |
| 512×3 predictor, final-state cost | 3 | 55 ± 3% | 86 ± 4% | 38 ± 5% | 17.2 | 0.54, 0.59, 0.52 |
| 512×3 predictor, running cost | 3 | 45 ± 3% | 84 ± 5% | 24 ± 3% | 11.4 | 0.47, 0.47, 0.40 |
| LayerNorm projector, running cost | 3 | 32 ± 3% | 58 ± 11% | 18 ± 2% | 10.7 | 0.35, 0.28, 0.33 |
| LayerNorm, 2× data (90% pushing), 12k steps, running cost | 3 | 40 ± 0% | 83 ± 2% | 17 ± 1% | 9.4 | 0.40, 0.40, 0.40 |
| Hand-written pusher (true positions) | - | 87% | 100% | 80% | 23.3 | - |
| Oracle CEM: true simulator, state-distance cost | - | 64% | 91% | 49% | 11.1 | - |
| Random actions | - | 1% | 3% | 0% | 9.0 | - |

100 fixed test tasks; each goal is the end of a 30-step push from the start. Success = block within 3 px of its target within 60 steps; long = the block must move 20 px or more. Mean ± std over 3 training seeds. Final-state
cost = CEM scores only the last imagined state; the running cost also averages distance along the plan. The
retrained models are the "Starting models" row after predictor retraining (next sections); those starting models
are the demo recipe re-trained on the Kaggle T4, so they differ slightly from push_fs4_s0-s2.

#### Goals away from the walls (found late)

94 of the 100 test goals have the block touching a wall, 30 of them in a corner: the 30-step scripted push that
makes each goal almost always runs the block into a wall. So the test mostly asks for the block at the right place
along a wall, and the wall does part of the stopping. To check that the result isn't an artefact of that, a second
set of 100 tasks (`--taskset interior`, seed 5050, never used for any choice) keeps only goals with the block in the
open (centre 12-52 px, where it can be pushed from all four sides). Its goals come from shorter 12-step pushes
(median block move 13 px; 14 long pushes).

| Method | Success | Short | Long |
|---|---|---|---|
| Demo recipe (1-step loss + dropout, 1 round of retraining, running cost), 3 seeds | 92 ± 1% (0.91, 0.91, 0.93) | 91% | 95% |
| Hand-written pusher (true positions) | 100% | 100% | 100% |
| Oracle CEM: true simulator, state-distance cost | 94% | 95% | 86% |
| Random actions | 0% | 0% | 0% |

The learned planner stops the block in the open as well as it does against a wall. These pushes are shorter, so
the hand-written pusher gets all of them. 10 of the planner's 25 failures here are dead ends (next section).

#### Prediction-loss rollout length (tuning tasks, mean ± std over seeds, (n) seeds)

| Prediction loss | Final-state cost | Running cost |
|---|---|---|
| 8 steps (two-room default) | 43 ± 4% (3) | 31 ± 1% (3) |
| 8 steps + dropout 0.1 | 32 ± 4% (3) | 26 ± 1% (3) |
| 4 steps | 63 ± 1% (3) | 45 ± 2% (3) |
| 4 steps + dropout | 56 ± 3% (3) | 48 ± 2% (3) |
| 2 steps | 69 ± 2% (3) | 75 ± 2% (3) |
| 2 steps + dropout | 55 ± 2% (3) | 79 ± 2% (3) |
| 1 step (teacher forcing only) | 56 ± 2% (3) | 82 ± 3% (3) |
| 1 step + dropout | 40 ± 2% (3) | 88 ± 1% (3) |

Rollout length = open-loop predictor steps per training window in the prediction loss. Same encoder, predictor size (256×2), data, SIGReg weight and planner (5 knots × 4 steps, 300 samples, 10 iterations).

#### Variants tried on the tuning tasks (seed 0 unless noted)

| Run | Planner | Success | Short | Long | Block read-out error (px) |
|---|---|---|---|---|---|
| push_fs4_s0 | K5xh4 S300 I10 rw1.0 | 32% | 53% | 23% | 4.3 |
| push_fs4_s0 | K8xh4 S300 I10 rw1.0 | 32% | 57% | 21% | 4.3 |
| push_fs4_s0 | K4xh8 S300 I10 rw1.0 | 27% | 50% | 17% | 4.3 |
| push_fs4_s1 | K5xh4 S300 I10 rw1.0 | 31% | 57% | 20% | 4.0 |
| pushv_l0.01_fs4_15k_s0 | K5xh4 S300 I10 rw1.0 | 17% | 40% | 7% | 13.7 |
| pushv2_ln_s0 | K5xh4 S300 I10 rw1.0 | 29% | 53% | 19% | 5.5 |
| pushv2_ln_s1 | K5xh4 S300 I10 rw1.0 | 28% | 50% | 19% | 4.6 |
| pushv2_ln_s2 | K5xh4 S300 I10 rw1.0 | 30% | 57% | 19% | 5.2 |
| push_ln_fs4_s0 | K5xh4 S300 I10 rw1.0 | 20% | 33% | 14% | 4.0 |
| push_ln_fs4_s1 | K5xh4 S300 I10 rw1.0 | 27% | 40% | 21% | 4.8 |
| push_ln_fs4_s2 | K5xh4 S300 I10 rw1.0 | 22% | 40% | 14% | 3.7 |

**What didn't work, and why (first models).** The planner gets 74% of short pushes but only 19% of long ones, against 87% / 80% for
a hand-written pusher with true positions. In failures it does push the block, just not toward the target, so the
problem is the model's imagined contact dynamics, not the search:

- **Prediction is much worse than in two rooms.** 16-step rollout error is ~0.18 vs 0.048, and a linear read-out
  finds the block only to ~4 px (0.8 px for the agent in two rooms). Contacts are rare in the data (the block moves
  on ~26% of push-policy steps and ~4% of wander steps), so the predictor sees few of them.
- **A longer horizon doesn't help:** 32% (20 steps), 32% (32 steps), 27% (32 steps with coarser holds) on the
  tuning tasks.
- **Weaker SIGReg makes it worse, in an instructive way.** λ = 0.01 with 2.5× more training gives excellent
  rollout error (0.011-0.016) but the block read-out degrades to 13.7 px and success drops to 17%: the model learns
  to mostly ignore the rarely-moving block, which makes its embeddings easy to predict and useless for pushing.
  Good prediction loss is not evidence of a good planning representation.
- **LayerNorm, which fixed two-room training (finding 9), hurts here:** 32% vs 39% on the same data.
- **Twice the data, 90% of it pushing, with LayerNorm and 12k steps:** 40%. Short pushes rise to 83% and long ones
  stay at 17%, so more contact data helps the easy case only. Long pushes, which need the agent to walk around the
  block, are where the remaining gap to the hand-written pusher (80%) lies.
- **Subgoals (finding 8) don't fix it either.** The same waypoint graph built from 13,000 push frames is badly
  fragmented at the default settings (391 pieces; push states are 4-D, so the bank covers them sparsely) and scores
  32% on the tuning tasks, the same as plain CEM. With looser neighbour links the graph connects (4 pieces, then 1)
  and scores 31% and 35% (long pushes 23% and 27% vs 23%), within noise, so it was not run on the test tasks.
  Subgoals help when the route is the problem (long navigation), not when the model's contact predictions are.
- Not tried: a bigger latent, and LeWM's longer history (3 frames). History adds little here: the scene is fully
  observed and has no momentum, so one frame is the whole state.

#### Where the push planner loses: dynamics, not the cost

Swapping one part at a time for the real thing, on the 100 tuning tasks with the same CEM (5 knots × 4 steps,
300 samples, 10 iterations) unless noted:

| Dynamics | Cost | Success | Short | Long |
|---|---|---|---|---|
| Learned predictor (push_fs4_s0) | Latent distance to the goal image | 32% | 53% | 23% |
| True simulator | True-state distance (oracle) | 57% | 80% | 47% |
| True simulator | Latent distance (every candidate rendered and encoded) | **86%** | **100%** | **80%** |
| True simulator | True-state distance, 8 knots, 1000 samples × 15 iterations | 81% | 93% | 76% |
| Learned predictor | Latent distance, same bigger search | 36% | 57% | 27% |

- **The embedding is a very good cost.** With true dynamics it gets 86%, more than the hand-written true-state cost
  (57%) and about the same as the hand-written pusher (85% on these tasks). Its distance seems to weigh block and agent
  in a way that avoids the local minima of plain state distance (which counts the agent as much as the block).
- **The learned dynamics are the weak part.** Same cost, learned predictor: 32%. More search, which lifts the
  oracle from 57% to 81%, does nothing for the learned model (24-36% across 8 settings and two models): extra
  search mostly finds plans that look good only inside the model.
- **A bigger predictor helps a little at most** (seed 0, tuning tasks): 256×2 (current) 32%, 512×2 36%, 512×3 40%,
  1024×2 32%, 1024×3 39%. Plan time grows with it (Python, 100 tasks batched: 0.49 s → 0.96 s for 512×3).

- **What did help: score only the final imagined state.** The running cost (mean distance to the goal along the
  plan, added because final-state-only planning procrastinated in two rooms, finding 2) penalises the detour a
  long push needs: walking away from the target to get behind the block. Dropping it (running_w 0) lifts the
  test score of the same models from 39% to 52% (long pushes 19% → 34%; 3 seeds) and the 512×3 predictor from 45%
  to 55%; the true-simulator oracle goes from 57% to 70% on the tuning tasks. LeWM, DINO-WM and V-JEPA 2-AC all
  plan with a final-state cost. The push demo used it for a day (same model, browser planner 52% on the test
  tasks). This holds for models trained with the 8-step loss; with the 1-step loss below, the running cost wins.
- **Other training changes** (512×3 predictor, seed 0, tuning tasks, running cost): a 12-step rollout loss 25%
  (vs 40% with 8 steps; published JEPA ablations also favour short rollout losses), frameskip 1 37% at 3.6× the
  plan time.

#### What fixed it: train the predictor on short predictions, then on the planner's own attempts

Both top suggestions from the literature review below worked, with the same encoder size, predictor size, data and
planner. Together they take push from 52% to 91-94% on the test tasks, above the hand-written pusher that uses
true positions (87%).

- **Shorten the prediction loss's rollout.** The training loss, copied from two rooms, chains the predictor 8 steps
  open-loop on each window and penalises every step. With fewer steps (table above), push success on the tuning
  tasks rises steadily: 31% (8 steps) → 45% (4) → 75% (2) → 82% (1 step, pure teacher forcing) with the running
  cost, and 88% with 1 step plus predictor dropout 0.1. On the test tasks that model scores 88% ± 1% (0.87 / 0.90
  / 0.87; long pushes 84%). Dropout alone, with the 8-step loss, hurts (26-32%).
- **The planning cost interacts with the loss.** With the 8-step loss the final-state cost is better (43% vs 31%),
  but with 1-step models it is much worse (40% vs 88% with dropout). A plausible reading: a model trained only on
  one-step predictions drifts over a 5-step imagined rollout, so its last imagined state is the least reliable
  part of the plan, and the running cost leans on the earlier, more accurate states. A model trained on long
  open-loop rollouts keeps the end state accurate but, as found in "What did help" above, the running cost then
  penalises the detour long pushes need. Why the long loss hurts contact at all is not settled; one guess is that
  it pulls the predictor towards a smooth average over what could happen at contact, which blurs the moment the
  block starts or stops moving.
- **Retrain the predictor on the planner's own attempts** (DAgger-style data aggregation, `scripts/push_dagger.py`).
  Freeze the encoder, run the planner on 2,000 fresh random tasks with Gaussian action noise (σ = 0.3), encode the
  frames of what really happened, and retrain the predictor (6,000 steps) on the original episodes plus these;
  repeat. No rewards or positions are used. With the 8-step loss it lifts test success from 51% to 75% after 2
  rounds, and the same fine-tuning without the new data gives only 55%, so there the planner's data is what helps.
  On top of the short losses (running cost, test, after 1 round): 2-step 90%, 2-step + dropout 93%, 1-step +
  dropout 91% (94% after 2 rounds). For the 1-step models most of that gain is the extra predictor training
  itself: the no-new-data control already reaches 91% (from 84%), and the planner's data adds 1-3 points (tuning
  tasks: 0.92 control, 0.93 after 1 round).
- **Demo model:** 1-step loss + dropout, retrained on 1 round of the planner's attempts, seed 0 (recipe and round
  picked by mean tuning score across seeds: 0.93, against 0.92 for 2-step + dropout; the seed was fixed in
  advance). Test 91% ± 0% over 3 seeds; the browser planner scores 90% on the same tasks (Python, same model: 91%).
- **Run-to-run noise is about 3-4 points.** The 1-step + dropout recipe trained twice on the same T4 type with the
  same seeds scored 0.87 / 0.90 / 0.87 and 0.84 / 0.85 / 0.84 on test (GPU kernels are not deterministic), so
  differences of that size between recipes are not meaningful.

#### Remaining failures: half are dead ends

The agent can only push. To move the block away from a wall it has to stand between block and wall, which needs a
6 px gap, so once the block centre is within 6 px of a wall (x or y below 12 or above 52) it can never come away
from that wall, and in a corner it is stuck for good. When the goal lies the other way by more than the 3 px success
radius, the task can no longer be solved. `scripts/push_deadends.py` checks the saved test trajectories for this
(a sufficient condition, so it undercounts; it never flags a task that later succeeded).

- Demo recipe, 3 seeds: 28 of 300 test episodes fail. **15 of them are dead ends** (2, 7 and 6 per seed), almost
  all from shoving the block against a wall the goal isn't on (14; 1 into a corner), at a median step of ~11 of 60,
  while the block is still ~40 px from its goal. The other 13 are near misses that end a median 7.7 px away.
- On the open-room goals above it is 10 of 25 failures (2, 4 and 4 per seed).
- The hand-written pusher and the true-simulator planner never cause a dead end. With the first push models about
  40% of failures were dead ends (23-27 of ~61 per seed).
- The planner can't see this coming. Its cost is a plain distance between embeddings, which is symmetric: it has no
  notion that a move can be one-way. A cost that measures "how many steps from here to the goal" (which can differ
  from goal to here) would rank a pinned block as far away; learning such an asymmetric distance is one direction
  for future work.

A literature review of how others handle pushing (`reports/World models for block pushing.md`) points the same
way: published PushT results use ~40× our data, one-step or 2-step prediction losses, predictor dropout, and
final-state costs; retraining the predictor on the planner's own rollouts is the best-supported fix for the
model exploitation seen here.

Scripts: `scripts/push_cost_vs_dynamics.py` (uses a batched torch port of the push renderer, checked pixel-exact
against `env_push.render`), `scripts/push_oracle_horizon.py`, `scripts/push_search.sh`, `scripts/train_push_pred.sh`,
`scripts/push_dagger.py` (predictor retraining; per-round logs in `results/push_dagger/`).

## Browser demo

Model files are fetched from jsDelivr pinned to the deployed commit (written to `deploy.json` by the Pages
workflow), falling back to GitHub Pages on any error; GitHub Pages delivered only ~140 KB/s to the test laptop
at one point (page load 21-25 s), jsDelivr brought it to 1.3-3.3 s.

`web/` is a static page: `env.js` (bit-exact port of the environment), `cem.js` (the same CEM as
`planner.py`, with one ONNX call per CEM iteration scoring all 300 samples over the full horizon), and `main.js`
(loop, drawing, input). The encoder and the unrolled predictor are exported to ONNX and run with ONNX Runtime Web
1.31.1 on wasm (one thread, since static hosting has no cross-origin isolation). Large weights are stored as fp16
with a Cast back to fp32 when ONNX Runtime's output stays within 2e-2 of PyTorch. The original BatchNorm
frameskip-4 encoder failed that check (its projector BatchNorm sees features with running variance ~1e-7, which
amplifies rounding upstream) and shipped as fp32; the LayerNorm model now in the demo passes it (finding 9). The imagined path is drawn with a linear probe from z to (x, y) fitted after training; the
planner never uses it. The cost map uses precomputed embeddings of a 1 px grid of positions.

Tested headless in Chrome and WebKit at 1200×900 and 390×844 (`tests/web_smoke.py`): the page loads, plans and
reaches goals with no horizontal overflow.

Each demo task ends when it is reached or after 2× the eval budget (100 steps in two rooms, 120 for push), with a
running score in the status line, and the demo moves on to a new task. Push tasks come from a JS port of the
test-set generator (`makeTask` in `env_push.js`: random start, then 30 steps of the scripted pusher; same
distribution of push lengths as the Python test set). Chaining each push from wherever the last one ended was much
harder than the test set, because the agent ends every push pressed against the block: 0 of 12 chained tasks were
reached. With the first push model the demo loop then reached 31 of 64 tasks (48%) with its 120-step limit, and the
browser planner scored 37% on the 100 test tasks with the test's 60 steps (`tests/web_eval.py --scene push`); the
current model (1-step loss + dropout, one round of retraining, running cost) scores 90% there.

## Limitations

- The environment is tiny and fully observed: one frame is a complete state, the dynamics are deterministic,
  and there are no distractors. This shows the recipe working end to end, not that it scales.
- With short horizons the planner fails on long tasks unless it plans through subgoals (finding 8); the waypoint
  graph needs a dense bank of training frames, which is easy in a world this small.
- One planning seed per trained model; the spread across seeds is training variance only.
- The demo uses one model (frameskip 4, LayerNorm projector, seed 0); the evaluation numbers cover all three
  seeds. Its browser planner (same algorithm, JS random numbers, ONNX Runtime Web, fp16 weights) scores 100% on
  the test tasks. The BatchNorm model it replaced scored 99% in the browser; that one miss had its goal in the
  corner of the doorway.

## Engineering notes

- **The first run of the data generator was OOM-killed** (9.2 GB RSS on a 10.7 GB machine): `render_batch` was
  called on all 98k states at once and built a float64 copy of every frame. It now renders 64 episodes at a time
  into a uint8 buffer, appends the bytes to `frames.npy` behind a pre-written header (no memmap, so written pages
  don't count toward RSS), and aborts above 3 GB. Peak RSS is 0.15 GB.
- Training reads the frames with plain file reads and uploads them to the GPU in chunks, so host RSS stays at
  ~1.8 GB (most of it CUDA libraries).
- Extra runs (the λ sweep) ran on a free Kaggle T4 through a private script kernel with the code embedded
  (`scripts/kaggle_run.py`). The kernel regenerates the dataset from its seeds and prints sha256 hashes, which
  match the local data.

## References

- LeWM: Maes et al., code at github.com/lucas-maes/le-wm (MIT). Objective, projector design and SIGReg settings.
- LeJEPA / SIGReg: Balestriero & LeCun, arXiv 2511.08544.
- Related planning-with-latent-dynamics work on two-room navigation: PLDM, DINO-WM, EB-JEPA (all read for
  context; no code used).
