# Train the predictor where CEM actually goes

The most promising fixes for the push planner change what the predictor learns from, not how hard CEM searches. The best-supported fix is the one already queued: freeze the encoder and retrain only the predictor on the planner's own rollouts (DAgger, which one recent paper calls "Online World Modeling"). In the closest published analogue, a JEPA model on PushT, this lifted open-loop CEM from 78% to 90% ([Closing the Train-Test Gap](https://arxiv.org/abs/2512.09929)). Our decomposition has three parts: true simulator plus our latent cost reaches 86%, the learned predictor with the same cost reaches 32%, and extra search stays flat at 24-36%. That is the standard signature of a planner exploiting model error, and two 2026 PushT studies observed it directly ([Chen 2026](https://arxiv.org/abs/2610.37891), [Min et al. 2026](https://arxiv.org/abs/2610.05550)). After DAgger, the ranked list is:

- Replace the 8-step rollout loss with a 2-step one and add predictor dropout 0.1.
- Fine-tune the predictor adversarially (AWM, 78% to 94% on PushT).
- Generate about 10x more contact-dense data.
- Draw CEM candidates from proposals backed by the training data.
- Add an ensemble disagreement penalty.
- Only if a contact probe fails, rebuild the encoder around explicit positions.

The top four add nothing at inference, which is the deciding constraint for a few-MB ONNX model on a 100-200 ms budget. The published 90-96% PushT numbers come from models trained on about 40x our data and tested on goals 25 steps away. At 75-step goals they drop to 17% ([G-JEPA](https://arxiv.org/html/2610.28483)). The groups that make long pushes work add grounding signals or keep the planner's candidates on recorded data. Our own uncommitted runs point the same way: a 12-step rollout loss cut tuning success from 40% to 25%, and terminal-only cost scored 40% against 32% ([results/eval_push.csv](../results/eval_push.csv)). Both deltas sit near the noise floor of a 100-task evaluation.

## Published PushT wins rest on 40x more data and 25-step goals

The strongest public result, LeWM, gets **96.0 ± 2.83%** on PushT with:

- **Model:** a ViT-tiny encoder (one 192-d CLS embedding) and a 6-layer, roughly 10M-parameter transformer predictor with AdaLN action conditioning, 3 frames of history and frameskip 5 ([LeWM](https://arxiv.org/html/2603.19312)).
- **Loss:** predictor dropout 0.1 and **pure one-step teacher forcing with no multi-step rollout loss** ([train.py](https://github.com/lucas-maes/le-wm/blob/main/train.py), [config](https://github.com/lucas-maes/le-wm/blob/main/config/train/lewm.yaml)).
- **Data:** about **20,000 expert episodes (≈3.9M frames)**.
- **Planner:** CEM with 300 samples, 30 elites and 30 iterations over 5 action blocks, terminal-only cost.

DINO-WM reaches 0.90-0.92 with frozen DINOv2 patch features, 18,500 noisy-expert trajectories and the same CEM ([DINO-WM](https://arxiv.org/html/2411.04983)). Both test goals 25 environment steps away with a budget of about 50 steps, and LeWM samples its start/goal pairs from training clips ([le-wm #111](https://github.com/lucas-maes/le-wm/issues/111)). Our test protocol is close to that 25-step setting: the goal is the end of a real 30-step push, with a 60-step budget and a 3 px tolerance. Our scene is also simpler, with no rotation and no momentum. So the gap between our 39% and their ~90% is real.

Their ablations show which knobs move PushT:

- **Data scale, the largest and best-quantified lever.** DINO-WM goes from 0.08 at 200 trajectories to 0.48 at 1,000, 0.72 at 5,000, 0.88 at 10,000 and 0.92 at 18,500 ([DINO-WM](https://arxiv.org/html/2411.04983)).
- **Predictor regularization, which beats predictor size.** LeWM scores 78% with dropout 0 and 96% with dropout 0.1. A larger "Base" predictor scores 86.7%, below the "Small" one ([LeWM](https://arxiv.org/html/2603.19312)).
- **Rollout-loss length has a short optimum.** Meta's JEPA-WM design study finds a 2-step rollout loss best in simulation, with longer rollouts worse. Only on real robot data does the optimum move to 6 ([JEPA-WMs](https://arxiv.org/html/2512.24497v3)).

Horizon is the hidden variable. LeWM's success falls from **90% to 53% to 17%** as the goal moves from 25 to 50 to 75 steps away ([G-JEPA](https://arxiv.org/html/2610.28483)). It halves from 0.98 to 0.53 when the planning horizon doubles past the trained one ([LeWM+I](https://arxiv.org/html/2610.03137v1)).

The 2026 follow-ups that fix long or hard pushes use three ideas:

- **Grounding signals that force the latent to track action and physical state.** LeWM+I adds an inverse-dynamics loss and lifts PushT at double horizon from 0.53 to 0.83 ([LeWM+I](https://arxiv.org/html/2610.03137v1)). G-JEPA adds physical-state supervision, a photometric-invariance loss and a training-only temporal module, and raises 75-step success from 17% to 48% ([G-JEPA](https://arxiv.org/html/2610.28483)).
- **Targets and candidate actions drawn from recorded data.** Anchored Planning takes PushT at offset 140 from 7.0% to 69.5% ([AP](https://arxiv.org/html/2609.30036v4)). HWM takes 75-step PushT from 17% to 61% ([HWM](https://arxiv.org/html/2604.03208)). SAGE takes offset-150 PushT from 4.7% to 64.7% ([SAGE](https://arxiv.org/abs/2607.17973)).
- **Closing the gap between training data and the states the planner reaches**, by on-policy aggregation or adversarial fine-tuning ([Closing the Train-Test Gap](https://arxiv.org/abs/2512.09929)).

| System | What it trains on | Predictor / loss | PushT result | The change that mattered |
|---|---|---|---|---|
| [LeWM](https://arxiv.org/html/2603.19312) | 20k expert episodes | 10M transformer, teacher forcing, dropout 0.1 | 96% at 25-step goals | Dropout (78 to 96), CEM over gradient methods |
| [DINO-WM](https://arxiv.org/html/2411.04983) | 18.5k noisy-expert trajectories | 19M ViT on frozen DINOv2 patches | 0.90-0.92 | Data scale; patches over CLS (0.44 to 0.90) |
| [JEPA-WMs](https://arxiv.org/html/2512.24497v3) | 18.5k trajectories | ViT-S, AdaLN, proprioception, 2-step rollout | 70.2% (stricter protocol) | 2-step rollout loss; proprioception |
| [LeWM+I](https://arxiv.org/html/2610.03137v1) | LeWM data | LeWM + inverse-dynamics head | 0.53 to 0.83 at 2x horizon | Encoder-side inverse dynamics |
| [OWM / AWM](https://arxiv.org/abs/2512.09929) | DINO-WM + planner rollouts | Predictor-only fine-tune | Open-loop CEM 78 to 90 / 94 | Training on the planner's own states |
| [Anchored Planning](https://arxiv.org/html/2609.30036v4) | Frozen LeWM, no training | Unchanged | 7.0 to 69.5 at offset 140 | Near targets and recorded action blocks |

## The decomposition shows CEM exploiting a predictor that gets contact wrong

Three facts from our own runs fix the diagnosis.

First, with the same latent cost, swapping the learned predictor for the true simulator takes success from 32% to 86% (long pushes 23% to 80%). The model is the weak part, not the cost ([WRITEUP.md](../WRITEUP.md)).

Second, a much bigger search lifts the true-simulator planner from 57% to 81%, but leaves the learned model at 24-36% across eight settings. When more optimization stops helping, the optimizer is finding plans that look good only inside the model. ME-TRPO saw predicted returns rise while real returns fell ([ME-TRPO](https://arxiv.org/abs/1802.10592)). On PushT specifically:

- DINO-WM's one-step latent error is higher on planner-generated trajectories than on expert ones ([Closing the Train-Test Gap](https://arxiv.org/abs/2512.09929)).
- Giving a LeWM-class planner 4, 8 and 12 updates turned 9 successes into 8 and then 7, while lowering the model's cost for the chosen plans ([Chen 2026](https://arxiv.org/abs/2610.37891)).
- The Spearman correlation between predicted latent cost and physical outcome across 300 candidates is only **0.193** for LeWM. It rises to 0.631 when the true endpoints are encoded instead of predicted ([Min et al. 2026](https://arxiv.org/abs/2610.05550)).

One practitioner found CEM-chosen plans whose predicted cost was 27x below their realized cost ([le-wm #107](https://github.com/lucas-maes/le-wm/issues/107)).

Third, doubling the data with 90% pushing helped only short pushes. Comparing like with like (both with LayerNorm), short pushes went from 58% to 83% and long pushes from 18% to 17% ([WRITEUP.md](../WRITEUP.md)).

Contact makes exploitation worse in our scene for structural reasons:

- **Contact is rare in the data.** The block moves on about 26% of scripted-pusher steps and about 4% of random-walk steps.
- **The latent localizes the block coarsely.** A linear read-out finds it only to **4.3 px**, coarser than the 3 px success radius and more than half the 8 px block.
- **Failures move the block the wrong way.** Failed runs push the block, just not toward the goal ([LOG.md](../LOG.md)). That is what a predictor does when it cannot tell which side of the block the pusher is on.

Low prediction loss does not rescue this. Our λ = 0.01 run reached rollout error 0.011-0.016 while block read-out degraded to 13.7 px and success fell to 17%. TailReg cut LeWM's contact-step prediction error from 0.428 to 0.247 without fixing planning ([NeurIPS 2026 workshop](https://neurips.cc/virtual/2026/159691)). On a pushing benchmark with small objects, plain LeWM's latent was nearly blind to actions. Every configuration below an action-sensitivity score of about 0.05 failed to plan ([LeWM+I](https://arxiv.org/html/2610.03137v1)).

Four cheap diagnostics decide which half of the ranked list to run. All of them are less noisy than a 100-task success rate, whose 95% binomial interval is about ±9 points at 35% success (about ±13 for the difference of two unpaired runs).

1. **Predicted vs realized cost of executed plans**, tracked across CEM iterations. A ratio that grows with iterations means exploitation. Clipping parity is already handled: `cem_plan` clamps candidates before imagining them.
2. **The train-test gap.** Compare the predictor's one-step error on CEM-chosen action chunks, executed in the true simulator, with its error on dataset chunks ([Closing the Train-Test Gap](https://arxiv.org/abs/2512.09929)).
3. **Candidate-ranking Spearman** against the batched true renderer we already have.
4. **A contact probe.** Train a small MLP on (z, action chunk) to predict whether and which way the block moves in the next chunk.

If the probe is accurate while diagnostics 1 and 2 show large gaps, the encoder holds the contact information and the predictor-only fixes ranked 1-6 apply. If the probe fails, no amount of predictor retraining recovers information the latent lacks, and ranks 7-8 move to the top.

## Eight fixes ranked by evidence per browser millisecond

The ranking weighs four things: how strong the external evidence is, whether our own runs agree, whether the fix targets exploitation and contact specifically, and what it costs inside a few-MB ONNX model planning in 100-200 ms. The current browser planner takes about 0.17 s per replan ([results/eval_push.csv](../results/eval_push.csv)).

| Rank | Fix | What it fixes | Strongest external evidence | Our evidence so far | Browser cost |
|---|---|---|---|---|---|
| 1 | DAgger / Online World Modeling: frozen encoder, retrain predictor on planner rollouts mixed with original data | CEM steering into states the data never covered | PushT open-loop CEM 78 to 90 ([OWM](https://arxiv.org/abs/2512.09929)) | Running ([push_dagger.py](../scripts/push_dagger.py)) | None |
| 2 | Teacher forcing + 2-step rollout loss instead of 8; predictor dropout 0.1 | Blurred contact predictions, overfitting | 2-step best in simulation ([JEPA-WMs](https://arxiv.org/html/2512.24497v3)); dropout 78 to 96 ([LeWM](https://arxiv.org/html/2603.19312)) | 12-step loss: 25% vs 40% | None |
| 3 | Adversarial World Modeling (AWM) fine-tune | Predictor brittleness on the inputs CEM picks | PushT open-loop CEM 78 to 94 ([AWM](https://arxiv.org/abs/2512.09929)) | Untested | None |
| 4 | ~10x contact-dense data, oversample block-moving transitions | Too few contact transitions | 1k to 10k trajectories: 0.48 to 0.88 ([DINO-WM](https://arxiv.org/html/2411.04983)) | 2x: short 58 to 83, long flat | None (training time only) |
| 5 | Data-backed candidates: goal-conditioned proposal policy in ~20% of CEM samples, or retrieved action blocks | Exploitation; long pushes | 79.4 to 87.1 ([G-JEPA](https://arxiv.org/html/2610.28483)); 7.0 to 66.4 ([AP](https://arxiv.org/html/2609.30036v4)) | Graph subgoals (targets, not actions) within noise | Tiny MLP or <1 MB bank |
| 6 | 3-5 predictor ensemble with disagreement penalty | Exploitation | MOPO out-of-distribution: 18.6 to 2531 with penalty ([MOPO](https://arxiv.org/abs/2005.13239)) | Untested | 3-5x predictor calls; pay with iCEM |
| 7 | Encoder grounding: inverse-dynamics loss, latent 64-128-d | Latent that ignores actions or the block | PushT 2x horizon 0.53 to 0.83 ([LeWM+I](https://arxiv.org/html/2610.03137v1)) | λ = 0.01 ignored the block | Near zero; encoder retrain |
| 8 | Explicit positions: spatial-softmax keypoints + CoordConv, or a 2-entity latent with a pairwise MLP | Imprecise agent-block geometry at contact | Keypoint MLP pushes as well as one on ground-truth state ([Manuelli et al.](https://arxiv.org/abs/2009.05085)) | Read-out 4.3 px > 3 px tolerance | Similar size; full rebuild |

### Retrain the predictor on the planner's own states first

DAgger-style aggregation has the strongest theory and the most direct empirical match. With data aggregation, Ross and Bagnell's performance bound depends only on how well the exploration data covers a good policy. Training on a fixed batch instead carries a mismatch factor that can scale with the size of the state-action space ([Ross & Bagnell](https://arxiv.org/abs/1203.1007)).

Online World Modeling is exactly our planned setup: frozen DINOv2 encoder, predictor-only fine-tuning on planner rollouts executed in the simulator. It raised PushT open-loop CEM from **78% to 90%** ([OWM](https://arxiv.org/abs/2512.09929)). Two caveats temper this:

- **Under MPC its PushT result stayed at 92%**, but that baseline was already at the ceiling. Where MPC had headroom it helped: Wall went from 82 to 90 and PointMaze from 90 to 96.
- **It cost 28 points on PointMaze open-loop** (90 to 62).

Since we replan after every knot, we are in the regime where OWM never hurt. The guards from older work apply directly. Keep the original data in the mix: Ross and Bagnell drew half of each round from exploration data, and Nagabandi trained on 10% random and 90% aggregated data ([MB-MF](https://arxiv.org/abs/1708.02596)). Run several rounds rather than one. The train-test gap metric (diagnostic 2) should shrink after each round. If it shrinks but success does not move, the bottleneck is the representation, not coverage.

### Change the predictor's training recipe before its size

Our 8-step open-loop rollout loss covers 32 environment steps, far beyond the 2-step simulation optimum in JEPA-WMs. That study's best variant mixes encoder and predictor outputs as inputs (a random ground-truth prefix) and backpropagates only the last step's error ([JEPA-WMs](https://arxiv.org/html/2512.24497v3)). LeWM and DINO-WM use no rollout loss at all. Our own 12-step run fell from 40% to 25% on the tuning tasks (512x3 predictor, seed 0). The direction is clear: sweep 8, 4, 2 and 0 next.

Add dropout at the same time. Our predictor has none, and LeWM's PushT score swings from 78% to 96% to 85% as dropout goes from 0 to 0.1 to 0.2 ([LeWM](https://arxiv.org/html/2603.19312)). Retrain the predictor on the frozen encoder so the validated 86% cost stays intact.

Adversarial World Modeling is the third free change. It fine-tunes the predictor for one epoch on worst-case perturbations of input latents and actions. On PushT the radius was 0.05 of the batch standard deviation for the visual latent and 0.02 for actions, and it lifted open-loop CEM from **78% to 94%** and Adam from 54% to 82% without any new environment data ([AWM](https://arxiv.org/abs/2512.09929)). Its one bad number, Wall open-loop at 30%, sits next to a DINO-WM baseline the authors measured at 32% themselves. Noise injection in the style of Graph Network Simulators is the cheaper cousin. It traded a little one-step accuracy for better rollouts at an intermediate noise scale ([GNS](https://arxiv.org/abs/2002.09405)).

The bigger 512x3 predictor scores 45% on test across three seeds against 39%, mostly on short pushes ([results/eval_push.csv](../results/eval_push.csv)). Keep it if the browser absorbs its roughly doubled plan time (0.49 to 0.96 s batched in Python), but stop scaling there. Both LeWM and JEPA-WMs found larger predictors no better in simulation.

One free planner check belongs here too: dropping the running cost scored 40% against 32% on one seed. LeWM, DINO-WM and V-JEPA 2-AC all plan with terminal-only costs ([LeWM](https://arxiv.org/html/2603.19312), [V-JEPA 2](https://arxiv.org/html/2506.09985v1)). But LeWM+I found a path cost beating terminal-only cost 0.335 to 0.083 on a PushT checkpoint ([LeWM+I](https://arxiv.org/html/2610.03137v1)). Treat it as a three-seed ablation, not a settled fix.

### Feed the predictor more pushes, and expect short pushes to gain most

Data is free in our simulator and costs nothing at inference. DINO-WM's curve is steepest between 1,000 and 10,000 trajectories ([DINO-WM](https://arxiv.org/html/2411.04983)), and our 96k steps sit at its low end. The 2x run moved short pushes only, so 10x data is a short-push fix first.

Two refinements raise contact density per gradient step:

- **Prioritize badly modeled or rarely seen samples.** Curious Replay did this when building batches and lifted DreamerV3 on Crafter from 14.5 to 19.4 ([Curious Replay](https://arxiv.org/abs/2306.15934)).
- **Oversample transitions where the block moves.** Our share is about 26% of pusher steps.

TailReg is the warning: better contact-step prediction error alone did not move planning success ([NeurIPS 2026 workshop](https://neurips.cc/virtual/2026/159691)). Judge these changes by success and by the ranking diagnostic, not by contact-step MSE.

### Keep CEM's candidates where the model was trained

Once the predictor is better, the next lever is to stop CEM from proposing action sequences the data never contained. The best browser fit is a tiny goal-conditioned proposal policy. It would be trained on frozen latents with hindsight goals: the frame 30 steps later in the same episode, which matches how our test goals are built. It would then fill a fraction of each CEM population. The evidence:

- **G-JEPA** filled 20% of CEM candidates from a learned action prior and rose from 79.4% to 87.1% ([G-JEPA](https://arxiv.org/html/2610.28483)).
- **Fast-JEPA** fell from 90% to 45% on PushT when its proposal generator was removed. With the generator it planned in **10.88 ms against 1,730 ms for plain LeWM** ([Fast-JEPA](https://arxiv.org/html/2611.25927)).
- **Goal-conditioned behavior cloning alone** scores 75% on PushT in stable-worldmodel's baselines ([stable-worldmodel](https://galilai-group.github.io/stable-worldmodel/baselines/)).
- **On PushT, direct inverse-dynamics control** beat the best CEM configuration, 35.5% to 21.5% ([NeurIPS 2026 workshop](https://neurips.cc/virtual/2026/159691)).

The training-free alternative is Anchored Planning's ranking variant. It retrieves recorded action blocks whose (start, end, displacement) key matches (current, goal, goal minus current) and uses the model only to rank them. That reached 66.4% at PushT offset 140, against 7.0% for flat LeWM. A memory of 25% of the data kept 80.1% of the 83.8% mean ([AP](https://arxiv.org/html/2609.30036v4)). By my estimate, a quarter of our episodes at one key per 4-step block fits under 1 MB in fp16. This is not the waypoint graph that failed for push: that changed CEM's target, while the ranking variant executes recorded actions that cannot be exploited. V-JEPA 2-AC applies the same principle more crudely, limiting sampled actions to a small L1 ball because larger ones are "largely out-of-distribution" ([V-JEPA 2](https://arxiv.org/html/2506.09985v1)).

An ensemble penalty is the direct guard against exploitation, but it costs compute. Without its uncertainty penalty, MOPO collapsed on out-of-distribution tasks (ant-angle 2531 to 18.6, halfcheetah-jump 4017 to −981), while in-distribution results barely moved (6406 against 6409) ([MOPO](https://arxiv.org/abs/2005.13239)). PLDM, the JEPA planner closest to ours, adds ensemble variance to its planning cost ([PLDM](https://arxiv.org/html/2502.14819)). No PushT number for the penalty exists. Each 256x2 member is about 85k parameters (my estimate), so size is fine, but three to five members multiply predictor calls. iCEM's keep-and-shift elites and colored noise can pay for it: with learned PlaNet models, 366 samples over 3 iterations came close to CEM with 10,000 samples over 10 ([iCEM](https://arxiv.org/html/2008.06389v1)).

### Rebuild the latent around positions only if the contact probe fails

If the contact probe shows the 32-d latent cannot say which side of the block the pusher is on, the predictor-side fixes above hit a ceiling. The representation itself has to change.

The cheapest step is grounding the encoder. LeWM+I's encoder-side inverse-dynamics term alone took its small-object pushing benchmark from 0.003 to 0.290, while the predictor-side term alone gave 0.010 ([LeWM+I](https://arxiv.org/html/2610.03137v1)). Removing inverse dynamics cost PLDM 23 points on Diverse Maze ([PLDM](https://arxiv.org/html/2502.14819)). Two related options have mixed or modest support:

- **A state-regression head.** Our simulator supplies the targets for free, but G-JEPA found state supervision alone did nothing (56.9 against 55.7). It only helped combined with an invariance loss ([G-JEPA](https://arxiv.org/html/2610.28483)).
- **A larger latent.** LeWM reports success dropping below about 184 dimensions ([LeWM](https://arxiv.org/html/2603.19312)). Sweeping 64 and 128 dimensions costs little in size.

Any encoder change means re-checking the true-simulator-plus-latent-cost number.

The structural fix makes positions explicit. "Keypoints into the Future" pushed boxes as accurately with a 2x500 MLP on keypoints as on ground-truth state (1.12-1.19 cm against 1.14 cm). An autoencoder latent was about twice as bad in position and three to four times worse in angle ([Manuelli et al.](https://arxiv.org/abs/2009.05085)). In C-SWM, removing object factorization dropped 10-step accuracy on a pushing grid world from 99.9% to 24.1% ([C-SWM](https://arxiv.org/abs/1911.12247)). CoordConv shows plain CNNs struggle to map pixels to coordinates, and adding coordinate channels fixes that with 10-100x fewer parameters ([CoordConv](https://arxiv.org/abs/1807.03247)). Our scene has exactly two visually distinct entities, so it avoids the slot-binding failures that hit identical cubes. A spatial-softmax latent of 16 keypoints is still 32-d.

The open risk is SIGReg. Pushing bounded coordinates toward an isotropic Gaussian is untested, so apply it to a projection head or replace it with a prediction loss plus a position probe.

## Six popular fixes the evidence says to skip

**A pixel decoder** lowered PushT success in both systems that tried it: DINO-WM fell from 0.92 to 0.80 and LeWM from 96% to 86% ([DINO-WM](https://arxiv.org/html/2411.04983), [LeWM](https://arxiv.org/html/2603.19312)).

**Gradient-based planning** follows the model's gradient into exactly the regions where the model is wrong. On PushT it never beats CEM: LeWM gets 26% with SGD and 84% with Adam against 96% with CEM, and JEPA-WMs gets 9.0% with Adam against 70.2% with CEM ([JEPA-WMs](https://arxiv.org/html/2512.24497v3)).

**More CEM search or iCEM for success**, rather than for speed, is already ruled out by our eight-setting sweep. Elsewhere, an adapted iCEM solved 2 of 16 PushT scenes ([Chen 2026](https://arxiv.org/abs/2610.37891)).

**Frame history and frameskip 1** matter on PushT because its block has momentum ([DINO-WM](https://arxiv.org/html/2411.04983)). Our block is kinematic and fully observed. Our frameskip-1 run scored 37% against 40% and planned 3.6x slower.

**Hierarchy and learned temporal-distance costs** fix objectives. Our latent cost already reaches 86% with true dynamics. HWM's gain at 25-step goals is only 84% to 89% ([HWM](https://arxiv.org/html/2604.03208)), and the large temporal-cost gains are on navigation ([Singh 2026](https://arxiv.org/abs/2608.12959)).

**Weaker SIGReg** made the model ignore the block (λ = 0.01: 17%).

## Conclusion

The 86% result changes the problem. A browser-sized 32-d latent is already a good enough goal metric. What is missing is a predictor that is accurate on the states CEM actually visits, which is a training-distribution problem more than a capacity problem. That makes the browser budget less binding than it looks. The four best-supported fixes (planner-rollout aggregation, a 2-step rollout loss with dropout, adversarial fine-tuning, more contact data) change only training. The two that touch inference (data-backed proposals, ensembles) can be paid for by cutting CEM samples, which our own sweep shows are not buying anything with the learned model.

The simulator should also become a measuring instrument, not just a source of data. Candidate-ranking correlation against the pixel-exact batched renderer, and the train-test error gap on planner-chosen chunks, are much less noisy than a 100-task success rate whose interval is about ±9 points. Most deltas logged so far, including the bigger predictors and terminal-only cost, sit inside that interval.

Expect the gains to come in an order that is itself diagnostic. Data and training-side fixes have so far lifted short pushes first. If long pushes stay near 20% after DAgger and the recipe changes, while the contact probe shows the latent cannot place the pusher relative to the block, then the remaining gap is representational. An explicit-position encoder is the fix that fits a few-MB model.
