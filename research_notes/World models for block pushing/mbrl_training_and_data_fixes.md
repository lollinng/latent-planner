# Training-side and data-side fixes for inaccurate learned dynamics and planner exploitation (focus: pushing / contact)

Context used to judge relevance: LeWM-style JEPA (CNN encoder 64x64 -> 32-d, residual MLP predictor over 4-step action chunks, next-embedding MSE + 8-step open-loop rollout loss + SIGReg, no decoder), CEM in latent space, 2-D kinematic block push, 2000 eps x 48 steps offline (70% scripted pusher, 30% random walk, block moves on ~26% of pusher steps). True sim + our latent cost = 86%, learned predictor + latent cost = 32%; more CEM search helps the true sim (57% -> 81%) but not the learned model. "Recent" = 2023-2026. Reported results sit under "Cited Findings"; my own reasoning sits under "Inferences". A ranking of fixes for our failure mode is in the Inferences of the first question.

## 1. On-policy data aggregation / iterative model refinement (DAgger for models, PlaNet/Dreamer/MBPO-style online collection, model exploitation)

### Takeaway
Retraining the model on the states and actions the planner actually reaches is the best-supported fix for planner exploitation, in theory (Ross & Bagnell 2012) and in a recent JEPA/PushT experiment that matches our plan: frozen encoder, retrain only the predictor on planner rollouts (Online World Modeling, Dec 2025). There it raised open-loop CEM on PushT from 78% to 90%, but it also hurt open-loop CEM on PointMaze (90% -> 62%). Mix the new data with the original data, and evaluate with replanning (MPC) as well as open loop.

### Cited Findings
**Classic DAgger-for-models theory and results**
- Ross & Bagnell 2012 ("Agnostic System Identification for Model-Based RL"): each iteration collects transitions from a 50/50 mixture of an exploration distribution ν and the current policy's state-action distribution, adds them to one growing dataset, refits the model, and re-solves the control problem. Their bound for this DAgger loop depends only on how well ν covers a good comparison policy. The Batch baseline (fixed i.i.d. data from ν) instead carries a train/test mismatch factor that can scale with |S||A|. — [arXiv 1203.1007](https://arxiv.org/abs/1203.1007)
- Same paper, experiments: helicopter simulator (21-d state, 4-d control), 20 seeds. Hover used 100 samples per iteration, for 50 iterations (no delay) or 100 iterations (delay 1). DAgger "significantly" beat Batch in both settings and converged faster. DAgger worked with every exploration distribution tried; Batch worked with only one per setting. DAgger slightly beat the expert's linearized LQR controller. On the nose-in funnel task (500 samples/iteration, 100 iterations), DAgger again beat Batch with every ν. Exact costs are only in figures; model-free policy gradient needed 500 samples/iteration to reach a cost of about 15,000 after 100 iterations. — [arXiv 1203.1007](https://arxiv.org/abs/1203.1007)
- Nagabandi et al. 2018 (MB-MF): iterative aggregation of MPC rollouts into a random-data set; "training using mostly the aggregated on-policy rollouts significantly improves performance" (Fig. 6b; no numbers in text). Aggregation lets runs that start with little random data reach high final performance. Settings: 5–7 aggregation iterations, 9–10 new rollouts per iteration, retraining with a 10/90 random-to-aggregated mini-batch split. — [arXiv 1708.02596](https://arxiv.org/abs/1708.02596)
- PlaNet (Hafner et al. 2019): "online data collection helps for all tasks and is necessary for the cartpole, finger, and walker tasks" (Fig. 5, versus a model trained on random data only). Collection: 5 random seed episodes, then one new episode every 100 updates with N(0, 0.3) action noise. The Fig. 2 caption states that a deterministic transition model "makes it easy for the planner to exploit inaccuracies". — [arXiv 1811.04551](https://arxiv.org/abs/1811.04551)
- MBPO (Janner et al. 2019): models trained on more data from a policy's distribution have lower error and are less sensitive to policy shift (Fig. 1). With short branched rollouts, the policy "may not exploit the model at all"; k=1 "retains much of the benefit", 500-step rollouts were "too inaccurate". Ensemble of B=7 probabilistic MLPs. Ant: MBPO at 300k steps matches SAC at about 3M steps. — [arXiv 1906.08253](https://arxiv.org/abs/1906.08253)
- ME-TRPO (Kurutach et al. 2018): policy optimization "tends to exploit regions where insufficient data is available to train the model, leading to catastrophic failures". With a single model on Swimmer, predicted returns rose while real returns fell. Ensembles of 5/10/20 models beat a single model. — [arXiv 1802.10592](https://arxiv.org/abs/1802.10592)
- PDDM (Nagabandi et al. 2019), contact-rich dexterous manipulation: iterative on-policy collection with a 3-model ensemble and MPPI (horizon 7). Simulated Baoding balls were solved with 100,000 data points (2.7 h). Real Baoding: about 100% success on 90° turns after under 2 h of data, 54% on 180° turns. — [arXiv 1909.11652](https://arxiv.org/abs/1909.11652)

**Recent (2025-2026): DAgger-style refinement of JEPA world models on PushT**
- "Closing the Train-Test Gap in World Models for Gradient-Based Planning" (Dec 2025; NeurIPS 2026): Online World Modeling (OWM) plans from expert start/goal pairs, runs the planned actions in the true simulator, encodes the real outcomes, and fine-tunes **only the transition model with the DINOv2 encoder frozen**, on next-state prediction. PushT: 6,000 rollouts, batch 32, 1 epoch, LR 1e-5. PointMaze: 500 rollouts. Wall: 1,920 trajectories. — [arXiv 2512.09929](https://arxiv.org/abs/2512.09929)
- Same paper, Table 1, success % as open-loop / MPC:
  - PushT CEM: DINO-WM 78/92, OWM 90/92, AWM (adversarial fine-tuning, see question 2) 94/92.
  - PushT GD: 38/56 -> OWM 34/50 -> AWM 56/66.
  - PointMaze CEM: 90/90 -> OWM 62/96 (open-loop got worse) -> AWM 88/98.
  - Wall CEM: 74*/82 -> OWM 54/90 -> AWM 30/94. *The authors could not reproduce DINO-WM's 74% open-loop Wall result and measured 32% themselves.
  - Rope/Granular: not reported for OWM. IRIS on Wall: OWM gave 0 for both GD and CEM.
  - The paper says "only Online World Modeling supports successful completion" in some hard environments. — [arXiv 2512.09929](https://arxiv.org/abs/2512.09929)
- Same paper, Fig. 4 (PushT): DINO-WM's one-step latent error Δ = ||f(Φ(s),a) − Φ(h(s,a))||² is higher on planner-generated trajectories than on expert trajectories, and OWM and AWM remove this gap. Exact values are only in the figure. — [arXiv 2512.09929](https://arxiv.org/abs/2512.09929)
- DINO-WM: replanning helps even without retraining. On PushT, open-loop CEM 0.86 vs MPC (CEM with replanning) 0.90. — [arXiv 2411.04983](https://arxiv.org/abs/2411.04983)
- LeWM independent evaluation (NeurIPS 2026 workshop, Zi et al.): on PushT with frameskip 5, direct inverse-dynamics (IDM) control reached 35.5% mean success versus 21.5% for the best CEM configuration tested. So CEM through the latent predictor was weaker than direct action decoding in a contact-rich task. — [NeurIPS 2026 workshop page](https://neurips.cc/virtual/2026/159691)

### Inferences
- Our diagnosis (true sim plus more search helps; learned model plus more search does not) is the pattern described by Ross & Bagnell, ME-TRPO and Closing-the-Gap Fig. 4: the planner drives the predictor into action sequences the offline data does not cover. The cheapest check is to copy their Fig. 4 measurement. Compute the predictor's error on CEM-chosen action chunks (executed in the true sim) versus dataset action chunks. A large gap confirms exploitation and gives the metric DAgger should shrink.
- Our planned setup (encoder frozen, retrain only the predictor) is exactly OWM. Expected size of the gain, from the closest analog (DINO-WM PushT, open-loop CEM): about +12 points. The same method lost 28 points on PointMaze open-loop. Practical guards drawn from the cited methods:
  - Keep the original data in the mix: Ross & Bagnell used 50% exploration data, Nagabandi 10% random / 90% aggregated.
  - Run several rounds, not one.
  - Evaluate with MPC replanning, where OWM never hurt in Table 1.
- Ranking for our failure mode (my judgment, combining the evidence across all five questions):
  1. On-policy aggregation (DAgger/OWM). Best theory plus a direct PushT/JEPA result.
  2. Adversarial fine-tuning of the predictor on perturbed latents and actions (AWM, question 2). No environment interaction needed, and it was the best single fix for CEM on PushT (78 -> 94 open-loop).
  3. A predictor ensemble with a disagreement penalty in the CEM cost (PETS/MOPO/PLDM, question 3). It attacks exploitation directly at plan time.
  4. More data, or contact-weighted data and loss (question 4). By DINO-WM's PushT scaling, our ~96k steps is in the steep part of the curve.
  5. Regularizing the predictor (dropout 0.1 helped LeWM 78 -> 96 on PushT; GNS-style noise injection).
  6. Changing the multi-step loss horizon. We already train 8 steps; evidence favors 2–6 (question 2).
  7. Adding a decoder or reconstruction loss. Evidence on PushT is negative (question 5).

### Gaps
- Ross & Bagnell's DAgger vs Batch cost numbers are only in figures, not text. No exact effect size.
- I found no published DAgger-for-models result on a pure planar block-pushing task other than OWM on PushT. PlaNet, Dreamer and MBPO report "online helps" mostly qualitatively, or against SAC/PPO rather than against a fixed offline set.
- OWM's number of outer iterations N is not stated in the text I could read.

## 2. Multi-step losses: latent overshooting, multi-step consistency, rollout loss, scheduled sampling, noise injection

### Takeaway
Gains from longer training horizons are small and saturate early. A recent systematic JEPA study found 2-step rollout loss best in simulated tasks and 6-step best on real robot data, with longer horizons hurting in simulation. TD-MPC2 uses a horizon of 3 with 0.5 decay, and PlaNet's latent overshooting slightly hurt its best model. Perturbation-based training (GNS noise injection, adversarial latent/action perturbation in AWM) and predictor dropout have stronger reported effects. With an 8-step rollout loss already in place, horizon is unlikely to be the lever for us.

### Cited Findings
- **PlaNet latent overshooting:** averages KL bounds over distances 1..D with β_d weights (all β_{>1} set equal; values not given in the text). It "can substantially improve the performance of the DRNN" but "slightly reduces performance of our RSSM", so the final agent does not use it. Observation overshooting was judged "typically too expensive in image domains". The stochastic latent component matters more: "the agent does not learn without it". — [arXiv 1811.04551](https://arxiv.org/abs/1811.04551)
- **TD-MPC2:** the latent consistency (joint-embedding) loss is summed over the planning horizon H=3 with temporal decay λ=0.5 (later steps weigh less). Loss coefficients: consistency 20, reward 0.1, value 0.1. No decoder ("predicting raw future observations does not necessarily lead to effective control"). Horizon and SimNorm ablations are plots only. Size scaling on the 80-task data (normalized score): 1M 16.0, 5M 49.5, 19M 57.1, 48M 68.0, 317M 70.6. — [arXiv 2310.16828](https://arxiv.org/abs/2310.16828)
- **JEPA world-model design study (Terver, Yang, Ponce, Bardes, LeCun; Dec 2025, v3 May 2026):**
  - Rollout loss with K unrolled steps: "performance increases when going from pure teacher-forcing models to 2-step rollout loss models", then decreases in simulated environments. On DROID real-robot data "the optimal tradeoff point shifts to K=6". Final choice: K=2 for simulated navigation, K=6 for real manipulation.
  - Uses truncated backprop through time: only the error of the last prediction is backpropagated.
  - L2 planning cost "consistently outperforms" L1. CEM with L2 cost is best overall; gradient planners fail on 2-D navigation.
  - Predictor depth 6 is best in most simulated environments; a bigger encoder does not help in simulation.
  - Proprioception is "consistently better".
  - Final model vs DINO-WM (success %): Push-T 70.2 vs 66.0, Wall 78.8 vs 64.1, Maze 83.9 vs 81.6, Metaworld-Reach 58.2 vs 44.8, DROID 48.2 vs 39.4. — [arXiv 2512.24497](https://arxiv.org/abs/2512.24497)
- **DINO-WM:** trained with teacher forcing on H+1 frame segments with frame history H=3 (PushT) and frameskip 5. A causal mask is critical: PushT success with history h=3 is 0.08 without the mask and 0.92 with it. — [arXiv 2411.04983](https://arxiv.org/abs/2411.04983)
- **LeWM (2026):**
  - One-step teacher-forced loss on 4-frame sub-trajectories, frameskip 5, no multi-step rollout loss; still 96.0% on PushT.
  - Predictor dropout ablation (PushT): p=0.0 78%, p=0.1 96%, p=0.2 85%, p=0.5 67%.
  - Predictor size: tiny 80.7%, small 96.0%, base 86.7%, so bigger is not better. — [arXiv 2603.19312](https://arxiv.org/abs/2603.19312)
- **Noise injection (Graph Network Simulator, Sanchez-Gonzalez et al. 2020):** random-walk noise (σ=3e-4) is added to input velocities during one-step training, so the model sees inputs like its own drifted rollouts. Rollout accuracy is best at an intermediate noise scale; one-step accuracy falls as noise grows. The authors judge that two-step training losses are more expensive and "may not generalize to longer sequences as well as noise injection". Adding noise also improved the CConv baseline's rollouts on most datasets. — [arXiv 2002.09405](https://arxiv.org/abs/2002.09405)
- **Adversarial World Modeling (AWM, Dec 2025):**
  - Fine-tunes the predictor on worst-case bounded perturbations of input latents and actions: one signed-gradient step with radius ε = λ × the batch std.
  - PushT radii: 0.05 (visual latent), 0.02 (proprio), 0.02 (action); batch 16, 1 epoch. No new environment data needed.
  - PushT results (open-loop / MPC): CEM 78/92 -> 94/92, GD 38/56 -> 56/66, Adam 54/76 -> 82/92.
  - Wall open-loop CEM fell to 30, though the authors' own DINO-WM baseline measured 32. With MPC, AWM improved every planner. — [arXiv 2512.09929](https://arxiv.org/abs/2512.09929)
- **Data as Demonstrator (DaD; Venkatraman et al., AAAI 2015):** a DAgger-style fix for multi-step prediction. Roll the model out on training sequences, and train it to map its own predicted states back to the true next states. The abstract reports "significant improvement" on dynamical-system and video-texture prediction. A 2016 ISER follow-up applies it to learned dynamics for control (LQR, iLQR, open-loop control on simulated and physical robots). — [AAAI 2015](https://ojs.aaai.org/index.php/AAAI/article/view/9590); [ISER 2016 (CMU RI)](https://ri.cmu.edu/publications/improved-learing-of-dynamics-for-control-2)
- **MPC horizon and compounding error:** Nagabandi 2018 found half-cheetah open-loop predictions diverge and longer MPC horizons do worse. PDDM found horizons both too short and too long detrimental (default H=7). PlaNet with the true simulator found horizon 6 "not sufficient", about 8 best, and much longer horizons hurt. — [arXiv 1708.02596](https://arxiv.org/abs/1708.02596); [arXiv 1909.11652](https://arxiv.org/abs/1909.11652); [arXiv 1811.04551](https://arxiv.org/abs/1811.04551)

### Inferences
- Our 8-chunk rollout covers 32 env steps, beyond the K=2 (sim) to K=6 (real) best points in the JEPA study. It is worth one ablation at K=2–4, or with TD-MPC2-style 0.5^t decay, since too long a horizon was found to hurt in simulation. I do not expect horizon changes to close a 32% vs 86% gap.
- AWM is effectively structured noise injection aimed at the inputs a planner would pick. It needs no new data, takes one epoch on PushT, and gave the largest single CEM gain on PushT. It is a cheap complement to DAgger: fine-tune once with AWM, then aggregate.
- If our residual MLP has no dropout or noise, the LeWM dropout result (78 -> 96 on PushT) and the GNS noise result suggest regularization is a cheap test. A bigger predictor barely helping us is consistent with LeWM's base model scoring below its small one.

### Gaps
- No numeric values found for the latent-overshooting β, the TD-MPC2 horizon ablation, or the per-K success rates in the JEPA study (all in figures).
- I found no quantitative scheduled-sampling result for latent world models used for planning. DaD's exact numbers were not retrieved (abstract only).
- I found no study of Gaussian noise on latents during JEPA predictor training, other than AWM's adversarial perturbation.

## 3. Ensembles and uncertainty penalties against planner exploitation (PETS, MOPO/MOReL/LOMPO, latent disagreement)

### Takeaway
Ensembles are the most consistently supported guard against exploitation. PETS ranked probabilistic ensembles first in every task except cartpole. In MOPO, removing the uncertainty penalty collapses performance on out-of-distribution tasks (ant-angle: 2531 -> 19; halfcheetah-jump: 4017 -> −981) while changing little on in-distribution data. In latent space, PLDM (a JEPA planner, 2025) adds ensemble variance to its MPPI cost. A 5-member predictor ensemble with a disagreement penalty in CEM is cheap for a small MLP predictor.

### Cited Findings
- **PETS (Chua et al. 2018):** model ranking probabilistic ensemble > probabilistic > deterministic ensemble > deterministic. Probabilistic ensembles "perform best in all tasks, except cartpole"; the choice of propagation method gives "only minor improvements". Ensemble B=5, P=20 particles. PETS reaches PPO's asymptote in under 100 trials; on half-cheetah it uses 8x fewer samples than SAC and 125x fewer than PPO. Per-task returns, including pusher, are only in figures. — [arXiv 1805.12114](https://arxiv.org/abs/1805.12114)
- **MOPO (Yu et al. 2020), uncertainty-penalized offline model-based RL:**
  - Penalized reward r̃ = r̂ − λ · max_i ||Σ_i(s,a)||_F, the largest predicted std across a 7-model ensemble (best 5 kept). λ and the rollout length h were searched over {1, 5}.
  - Ablation (average return), MOPO vs no penalty:
    - halfcheetah-mixed: 6406 vs 6409
    - walker2d-mixed: 1916 vs 1421
    - halfcheetah-jump (out of distribution): 4017 vs −981 ± 5625
    - ant-angle (out of distribution): 2531 vs 18.6
  - An ensemble-disagreement penalty instead of learned variance: 6449 / 1924 / 3577 / 2256, comparable in distribution, weaker out of distribution. An oracle true-error penalty: 6984 (halfcheetah-mixed), 3819 (halfcheetah-jump).
  - MBPO without an ensemble: 2247 vs 5598 with one (halfcheetah-mixed).
  - D4RL normalized scores, MOPO vs MBPO: halfcheetah-med-expert 63.3 vs 9.7, walker2d-med-expert 44.6 vs 7.6, but hopper-med-expert 23.7 vs 56.0. The penalty is not uniformly better. — [arXiv 2005.13239](https://arxiv.org/abs/2005.13239)
- **ME-TRPO:** ensembles of 5, 10 and 20 models all beat a single model, with larger gains on HalfCheetah and Ant. Training also stops updating the policy once fewer than 70% of ensemble members agree it is improving, which is an exploitation guard. — [arXiv 1802.10592](https://arxiv.org/abs/1802.10592)
- **MBPO:** the authors state that ensembles help prevent exploitation; bootstrap ensemble B=7. — [arXiv 1906.08253](https://arxiv.org/abs/1906.08253)
- **PDDM:** ensembles help "especially earlier in training when non-ensembled models can overfit badly"; M=3 in all tasks. — [arXiv 1909.11652](https://arxiv.org/abs/1909.11652)
- **PLDM (Sobal et al., Feb 2025), JEPA latent planning from reward-free offline data:**
  - MPPI cost = mean over a predictor ensemble of the latent distance to the encoded goal, plus β × the ensemble variance per latent dimension (with γ^t discount). Ant U-Maze uses an ensemble of 5 MLP predictors.
  - Separate ablation: removing the inverse-dynamics auxiliary loss dropped Diverse Maze from 98.7% to 75.5% (Two-Rooms unchanged at 98.0%).
  - I did not find an isolated number for the variance penalty. — [arXiv 2502.14819](https://arxiv.org/abs/2502.14819)
- **PlaNet:** a deterministic model "makes it easy for the planner to exploit inaccuracies". Stochastic latents "might also add a safety margin to the planning objective that results in more robust action sequences". — [arXiv 1811.04551](https://arxiv.org/abs/1811.04551)
- **MBPO:** short model rollouts mean the policy "may not exploit the model at all". — [arXiv 1906.08253](https://arxiv.org/abs/1906.08253)

### Inferences
- MOPO's pattern (the penalty matters little in distribution and a lot out of distribution) fits our case, where CEM search drives the predictor out of distribution. A 5-member ensemble of the small residual MLP, sharing the frozen encoder, with cost = mean latent distance + β × disagreement, is a direct test. If more CEM samples then start helping the learned model (as they do for the true sim), exploitation was the cause. Tune β over roughly {0.1, 1, 10} × the typical cost scale; MOPO searched λ over {1, 5}.
- Ensemble disagreement on executed plans is also a natural signal for which DAgger rollouts to keep or upweight.

### Gaps
- MOReL and LOMPO effect sizes were not retrieved. LOMPO is the most relevant (latent ensemble on images), so it would be worth reading.
- No isolated number for PLDM's variance penalty, and no penalty study on a pushing task.
- PETS pusher returns are only in figures.

## 4. Data collection and balancing for rare contact events (oversampling, prioritized replay, contact-seeking exploration, data quantity for pushing)

### Takeaway
Data quantity is a first-order effect on PushT. DINO-WM's success rises from 8% (200 trajectories) to 48% (1k), 72% (5k), 88% (10k) and 92% (18.5k), and LeWM trains on 20k expert episodes of about 196 steps (~3.9M frames), roughly 40x our 96k steps. Contact-seeking exploration (CEE-US) and prediction-error or tail-error prioritization (Curious Replay, DreamerV3-XP, TailReg) improve modeling of rare interactions. The only contact-specific numbers are TailReg's, and its prediction gains did not turn into control success.

### Cited Findings
**Data quantity for pushing**
- **DINO-WM PushT data scaling (Table 5):** 200 trajectories -> success 0.08; 1,000 -> 0.48; 5,000 -> 0.72; 10,000 -> 0.88; 18,500 -> 0.92. Data: 18,500 samples made by replaying the released expert trajectories with added noise, 100–300 steps each, frameskip 5. PushObj: 20,000 random trajectories of 100 steps. Wall: 1,920 random trajectories of 50 steps. — [arXiv 2411.04983](https://arxiv.org/abs/2411.04983)
- **LeWM:**
  - Data: PushT 20,000 expert episodes (average length 196, from the DINO-WM setup, 10 epochs); TwoRoom 10,000 episodes; OGBench-Cube and Reacher 10,000 episodes of 200 steps.
  - Results: 96.0 ± 2.8% on PushT, vs DINO-WM 92.0 and PLDM 78.0.
  - CEM settings: 300 samples, 30 elites, 30 iterations on PushT, horizon 5 action blocks (25 env steps).
  - The authors list reliance on "offline datasets with sufficient coverage" as a limitation. — [arXiv 2603.19312](https://arxiv.org/abs/2603.19312)
- **PushT numbers differ across evaluators:** DINO-WM's self-reported 0.90–0.92; 66.0% for DINO-WM in the JEPA design study; 78% open-loop / 92% MPC in Closing-the-Gap. Evaluation protocols differ. — [arXiv 2411.04983](https://arxiv.org/abs/2411.04983); [arXiv 2512.24497](https://arxiv.org/abs/2512.24497); [arXiv 2512.09929](https://arxiv.org/abs/2512.09929)
- **Planar pushing with a state-based GP model (Bauza, Hogan, Rodriguez, CoRL 2018):**
  - "Around two hundred points are sufficient" to match the analytical model's accuracy. Tested 10 to 5,000 points; tracking improves as model accuracy improves. With fewer than 10 random points the controller was unstable.
  - The 5,000-point model beat the analytical model on one track at 80 mm/s (8.50 vs 9.56 mm error), but was worse at 20 mm/s (6.53 vs 2.89 mm) and on the square track (6.60 vs 4.95 mm). — [arXiv 1807.09904](https://arxiv.org/abs/1807.09904)
- **PLDM data-quality study (Two-Rooms, 2025):**
  - With about 3M transitions (~30k trajectories), all methods approach 100%. As the share of good-quality data goes to zero, methods fail, with PLDM among the most robust.
  - PLDM reaches about 80% with only a few thousand transitions.
  - Uniform-random actions travel an average maximum distance of about 10, versus about 28 for Von-Mises-sampled (direction-persistent) actions, so random-walk data covers little. — [arXiv 2502.14819](https://arxiv.org/abs/2502.14819)
- **TD-MPC2 data scale:** 80-task multitask dataset of 545M transitions; single-task Meta-World budget of 2M env steps. — [arXiv 2310.16828](https://arxiv.org/abs/2310.16828)

**Contact-seeking exploration**
- **CEE-US (Sancaktar et al., NeurIPS 2022):**
  - Method: ensemble of 5 graph-network models plus planning (iCEM) to maximize ensemble disagreement; 2,000 transitions per iteration.
  - Interaction: objects get moved repeatedly after about 8 iterations, while policy-based curiosity baselines "rarely interact with objects, even after 300 iterations". An MLP ensemble needs "about ten times more interactions".
  - Zero-shot pushing of 4 objects: 0.70 vs 0.08 for MLP+iCEM after 100 iterations (~200k transitions); 0.91 vs 0.90 after 250 iterations (~500k).
  - Offline RL on the collected data, 1-object relocate: 0.90 (CEE-US data) vs 0.06 (Disagreement, RND and ICM data). — [arXiv 2206.11403](https://arxiv.org/abs/2206.11403)
- **PlayWorld (Mar 2026):** world models trained on autonomous robot self-play, which "captures long-tailed physical interactions", give more physically consistent contact-rich predictions than models trained on human demonstrations. Reported: up to 40% better failure prediction and policy evaluation, and a 65% gain in real-world policy success after RL in the world model. The abstract does not isolate a play-vs-demo number for contact accuracy. — [arXiv 2603.09030](https://arxiv.org/abs/2603.09030)

**Reweighting and prioritized replay by model error**
- **Curious Replay (Kauvar et al., ICML 2023):** samples world-model training batches by a priority combining how few times an experience has been trained on and how badly it is modeled. DreamerV3 on Crafter: 19.4 vs 14.5 with uniform replay. — [arXiv 2306.15934](https://arxiv.org/abs/2306.15934); [code](https://github.com/AutonomousAgentsLab/curiousreplay)
- **DreamerV3-XP (Oct 2025):** prioritizing replay by model prediction error "consistently reduced dynamics loss" (from the abstract/snippet; numbers not retrieved). — [arXiv 2510.21418](https://arxiv.org/abs/2510.21418)
- **TailReg on LeWM (NeurIPS 2026 workshop, Zi et al.):** label-free reweighting of tail prediction errors.
  - Contact-conditioned latent prediction MSE: 0.428 -> 0.247 (p95: 0.605 -> 0.333).
  - IDM gripper-action correlation: 0.450 -> 0.577.
  - But no strategy solved Cube in 50 episodes, and the authors do not report that TailReg fixes planning. — [NeurIPS 2026 workshop page](https://neurips.cc/virtual/2026/159691)
- **CAER (Aug 2026):** reweights the world-model loss toward tokens whose predicted future changes when the action is removed from the conditioning, keeping total weight constant. The abstract claims better physical consistency and controllability than uniform MSE; no numbers retrieved. — [arXiv 2608.30897](https://arxiv.org/abs/2608.30897)

### Inferences
- **Our data is small on the DINO-WM scale.**
  - By episode count (2,000) we sit near DINO-WM's 1k–5k region (48–72%). By env steps (96k vs roughly 200k at 1k DINO-WM trajectories) we are below its 1k point, about where it scored 8–48%.
  - Block-moving transitions are fewer still: about 1,400 pusher episodes × 48 steps × 26% ≈ 17k, if the random-walk episodes rarely touch the block.
  - Caveats: different task (a T-shape that must also rotate), DINO-WM uses a frozen DINOv2 encoder, and episode lengths differ.
- **Cheap data-side tests:**
  - Oversample block-moving transitions, or weight them by predictor error (Curious Replay / TailReg style), for example to a 50/50 contact / no-contact mix in each batch.
  - Generate 5–10x more scripted-pusher data from the simulator. It is cheap here and, by the DINO-WM curve, likely the largest single gain.
  - Replace the uniform random walk with direction-persistent random actions (PLDM's Von Mises result) or contact-seeking scripted pushes.
- A DAgger loop also concentrates new data where the planner wants contacts, so it partly doubles as contact-focused collection.
- TailReg is a warning: better contact prediction error alone did not turn into planning success. Measure control success, not only MSE on contact steps.

### Gaps
- No paper found that cleanly ablates oversampling of "object moved" transitions for a pushing world model with success-rate numbers.
- No data-scaling curve for LeWM itself, or for TD-MPC2 on push tasks.
- CAER and DreamerV3-XP numbers were not retrieved.

## 5. Reconstruction-based world models on pushing (DreamerV3, IRIS, DIAMOND, PWM, TD-MPC2/Meta-World): does a decoder help object dynamics?

### Takeaway
On offline PushT, reconstruction-based baselines plan poorly (IRIS 0.32, DreamerV3 0.30, TD-MPC2 0.00 vs DINO-WM 0.90). In both DINO-WM and LeWM, adding a decoder or reconstruction loss lowered PushT success (0.92 -> 0.80 and 96% -> 86%). The supporting evidence for reconstruction is older: decoupled masked-reconstruction representations beat end-to-end Dreamer on Meta-World (81.7% vs 67.9%), with an explicit "small objects" motivation. Since our latent cost already reaches 86% with the true simulator, the encoder sees the block well enough, and a decoder is unlikely to fix the predictor.

### Cited Findings
- **DINO-WM Table 1 (offline PushT success):** IRIS 0.32, DreamerV3 0.30, TD-MPC2 0.00, DINO-WM 0.90. Rope / Granular Chamfer distance (lower is better): IRIS 1.11 / 0.37, DreamerV3 2.49 / 1.05, TD-MPC2 2.52 / 1.21, DINO-WM 0.41 / 0.26. — [arXiv 2411.04983](https://arxiv.org/abs/2411.04983)
- **DINO-WM Table 7:** passing the decoder's reconstruction loss gradient into the predictor dropped PushT success from 0.92 to 0.80. — [arXiv 2411.04983](https://arxiv.org/abs/2411.04983)
- **LeWM Table 7:** PushT 96.0 ± 2.8% without a decoder loss vs 86.0 ± 7.5% with one; the authors conclude reconstruction does not help control. — [arXiv 2603.19312](https://arxiv.org/abs/2603.19312)
- **LeWM linear-probe correlation r for PushT block angle:** LeWM 0.902, PLDM 0.745, DINO-WM 0.979. The paper states all methods struggle with block orientation, and DINO-WM keeps an edge on fine details. — [arXiv 2603.19312](https://arxiv.org/abs/2603.19312)
- **TD-MPC2:** decoder-free by design; "predicting raw future observations does not necessarily lead to effective control". — [arXiv 2310.16828](https://arxiv.org/abs/2310.16828)
- **Masked World Models (Seo et al., 2022):** end-to-end training of the representation and the dynamics together makes it "difficult to accurately model the interaction between robots and small objects". Decoupling a masked convolutional-feature autoencoder (with an auxiliary reward head) from the latent dynamics gives 81.7% vs 67.9% success on 50 Meta-World visual manipulation tasks. The baseline and sample budget were not visible on the abstract page. — [arXiv 2206.14244](https://arxiv.org/abs/2206.14244)
- **Spectral-target physical latent structuring (Sep 2026):** reports a LeWM/SIGReg failure called "physical representation laziness": latents that do not collapse but miss key physical properties, especially in highly dynamic environments. A training-only Fourier auxiliary head "substantially improves planning success" there, with the largest gains in low-data regimes. No numbers in the abstract. — [arXiv 2609.04264](https://arxiv.org/abs/2609.04264)
- **PLDM:** a non-reconstruction auxiliary loss (inverse dynamics) matters. Removing it dropped Diverse Maze success from 98.7% to 75.5%. — [arXiv 2502.14819](https://arxiv.org/abs/2502.14819)

### Inferences
- Two independent JEPA results on PushT show a decoder loss hurting (−12 and −10 points). Our true-sim test (86% with our latent cost) also suggests the encoder already captures the block's position. A decoder is therefore low priority for our predictor problem.
- If anything is added on the representation side, auxiliary losses that force action-relevant or physical structure are better supported than pixel reconstruction: inverse dynamics (PLDM), physical-target heads (the Fourier head paper, largest gains in low data). They are most likely to matter if retraining the encoder is back on the table; with a frozen encoder, they do not affect the predictor.

### Gaps
- No pushing results found for DIAMOND, IRIS or PWM beyond DINO-WM's PushT baseline table (IRIS 0.32). I did not verify their evaluation domains in this pass.
- No per-task TD-MPC2 or DreamerV3 numbers retrieved for Meta-World push, push-wall or push-back. TD-MPC2 reports aggregate Meta-World curves; per-task data may be in its GitHub results CSVs.
- MWM's baseline identity and sample budget, and the Fourier head's effect sizes, were not retrieved.
