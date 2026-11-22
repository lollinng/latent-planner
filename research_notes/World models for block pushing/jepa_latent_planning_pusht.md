# JEPA / latent-planning world models (no pixel decoder) on PushT and block pushing: reported numbers and the design choices that matter

Context for the reader: our model is a 64x64 RGB -> 32-d CNN encoder, residual MLP predictor (z_t + 4-step action chunk -> z_{t+4}), loss = next-embedding MSE + 8-step open-loop rollout loss + SIGReg (lambda 0.09), CEM in latent space, trained on 2000 x 48-step episodes (70% scripted pusher, 30% random walk) of a kinematic block scene. We get 32-39% success (~20% on long pushes); true simulator + our latent cost gives 86%, so the learned dynamics are the weak part. "Inferences" below flag where a finding bears on that setup; everything in "Cited Findings" is what the source reports.

## Q1. LeWM (LeWorldModel) and SIGReg: PushT numbers, predictor, encoder, data, planner, ablations

### Takeaway
LeWM reports 96.0 +/- 2.83% on PushT (3 seeds x 50 episodes). The setup: a ViT-tiny encoder (224x224, CLS token -> 192-d), a 6-layer, ~10M-parameter causal transformer predictor with AdaLN action conditioning, 3-frame history, frameskip 5 and dropout 0.1, trained with teacher forcing only (no rollout loss) on 20k expert episodes (about 3.9M frames). Planning is CEM 300/30/30 over 5 action blocks, with goals only 25 env steps away. In its ablations, the choices that move PushT most are predictor dropout (0.0 -> 78%, 0.1 -> 96%), predictor size (tiny 81%, small 96%, base 87%), embedding dim (drops below ~184) and the solver (CEM 96% vs Adam 84% vs SGD 26%). History length, frameskip and data size were not ablated in the text I could read. Third-party evaluations show success falling steeply with goal distance: about 90/53/17% at 25/50/75-step goals.

### Cited Findings
Reported results
- PushT, 3 training seeds, 50 trajectories: DINO-WM 92.0 +/- 1.63, PLDM 78.0 +/- 5.0, LeWM 96.0 +/- 2.83 (Table 5) — [LeWM paper, arXiv 2603.19312](https://arxiv.org/html/2603.19312)
- Text claims LeWM achieves "an 18% higher success rate" than PLDM on PushT and beats DINO-WM even when DINO-WM gets proprioception. In the main comparison DINO-WM is run without proprio "for fair comparison". — [LeWM paper](https://arxiv.org/html/2603.19312)
- Under fixed planning FLOPs, LeWM "significantly outperforms DINO-WM" on PushT and OGBench-Cube. A search snippet of the v2 figure reads 90 vs 48 for PushT at fixed FLOPs; I did not verify these values in the text. — [LeWM paper](https://arxiv.org/html/2603.19312), [arXiv PDF v2](https://arxiv.org/pdf/2603.19312v2)
- LeWM uses ~200x fewer tokens than DINO-WM and plans up to ~48x faster, with a full plan under 1 s. The model has ~15M parameters and trains on one GPU in a few hours. — [LeWM paper](https://arxiv.org/html/2603.19312), [le-wm README](https://github.com/lucas-maes/le-wm)
- On Two-Room, LeWM is worse than PLDM and DINO-WM. The authors attribute this to low intrinsic dimensionality making the isotropic-Gaussian (SIGReg) prior harder to match. On OGBench-Cube DINO-WM slightly beats LeWM. — [LeWM paper](https://arxiv.org/html/2603.19312)
- Probing on PushT (MLP probe, r): block location 0.999 (LeWM) vs 0.994 (PLDM) vs 0.999 (DINO-WM); block angle 0.990 / 0.972 / 0.995; agent location 0.998 / 0.993 / 0.999. — [LeWM paper, Table 1](https://arxiv.org/html/2603.19312)

Architecture (paper plus released config)
- Encoder: ViT-tiny (~5M parameters), patch 14, 12 layers, 3 heads, hidden 192, 224x224 frames, trained from scratch (`pretrained: false`). The embedding is the last-layer CLS token followed by a 1-layer MLP projector with BatchNorm (hidden 2048). The paper says this projector is needed because the ViT's final LayerNorm hampers the anti-collapse objective. — [LeWM paper](https://arxiv.org/html/2603.19312), [config/train/model/lewm.yaml](https://github.com/lucas-maes/le-wm/blob/main/config/train/model/lewm.yaml)
- Predictor (`ARPredictor`): transformer, depth 6, 16 heads, dim_head 64, mlp_dim 2048, width 192 (= embed_dim), dropout 0.1, learned positional embedding over `num_frames = history_size = 3`, causal attention. Actions enter every block through AdaLN-zero modulation, zero-initialized. A second BatchNorm-MLP head (`pred_proj`) follows the predictor. — [config/train/model/lewm.yaml](https://github.com/lucas-maes/le-wm/blob/main/config/train/model/lewm.yaml), [module.py](https://github.com/lucas-maes/le-wm/blob/main/module.py)
- Paper: ~10M predictor parameters, 6 layers, 16 heads, 10% dropout. Appendix D calls it a "ViT-S backbone", which conflicts with the main text. — [LeWM paper](https://arxiv.org/html/2603.19312)
- Frameskip 5: the action encoder input is `frameskip * action_dim` (5 x 2 = 10 for PushT), passed through a Conv1d + MLP `Embedder`. History is 3 frames for PushT and Cube and 1 for TwoRoom in the paper; the released TwoRoom config may use 3 ([issue #100](https://github.com/lucas-maes/le-wm/issues/100)). Batch size is 128. — [config/train/data/pusht.yaml](https://github.com/lucas-maes/le-wm/blob/main/config/train/data/pusht.yaml), [train.py](https://github.com/lucas-maes/le-wm/blob/main/train.py), [LeWM paper](https://arxiv.org/html/2603.19312)
- Optimizer: AdamW, lr 5e-5, weight decay 1e-3, bf16, gradient clipping 1.0, linear-warmup cosine schedule, `max_epochs: 100` in the config. The paper's App. E says 10 epochs; the two sources disagree. — [config/train/lewm.yaml](https://github.com/lucas-maes/le-wm/blob/main/config/train/lewm.yaml), [LeWM paper](https://arxiv.org/html/2603.19312)

Training loss as implemented
- `num_preds: 1`, `history_size: 3`, so each sample has 4 frames. The predictor gets embeddings 0..2 and actions 0..2 and is regressed onto embeddings 1..3. Every causal position is supervised, so this is pure one-step teacher forcing with no multi-step rollout loss. Loss = MSE + 0.09 x SIGReg(emb), with SIGReg using 1024 random projections and 17 knots, computed per time step over the batch and averaged. No EMA and no stop-gradient. — [train.py](https://github.com/lucas-maes/le-wm/blob/main/train.py), [config/train/lewm.yaml](https://github.com/lucas-maes/le-wm/blob/main/config/train/lewm.yaml), [module.py SIGReg](https://github.com/lucas-maes/le-wm/blob/main/module.py)
- The paper states λ = 0.1 as the default and calls λ the only tuned hyperparameter, found by bisection. The released config uses 0.09. — [LeWM paper](https://arxiv.org/html/2603.19312), [config](https://github.com/lucas-maes/le-wm/blob/main/config/train/lewm.yaml)
- A user who retrained the predictor supervising only the last of 3 context positions got far worse planning: TwoRoom 36/100 vs 85/100. The cause was that the planner starts from a single frame, and the retrained predictor's relative error at 1-frame context was 1.18, vs 0.006 at 3 frames. Retraining with the official all-position loss restored 84-89%. The same thread also found that CEM chose plans whose predicted cost was 27x lower than their true cost, i.e. the optimizer exploited model error. — [le-wm issue #107](https://github.com/lucas-maes/le-wm/issues/107)

Data
- PushT: 20,000 expert episodes, average 196 steps. TwoRoom: 10,000 episodes from a noisy heuristic policy. OGBench-Cube: 10,000 x 200-step episodes from the benchmark heuristic. Reacher: 10,000 x 200-step episodes, described as SAC, though a user reports the released Reacher data looks random ([issue #98](https://github.com/lucas-maes/le-wm/issues)). — [LeWM paper App. E](https://arxiv.org/html/2603.19312)
- The authors list as limitations: offline data needs sufficient coverage, low diversity weakens SIGReg in simple environments, planning is short-horizon, and fine details such as end-effector orientation get lost. — [LeWM paper](https://arxiv.org/html/2603.19312)

Planner and evaluation protocol
- CEM: 300 samples, top-30 elites, initial variance 1, 30 iterations on PushT (10 elsewhere). The horizon is 5 action blocks x frameskip 5 = 25 env steps. With `receding_horizon: 5` the whole 25-step plan is executed before replanning. — [config/eval/solver/cem.yaml](https://github.com/lucas-maes/le-wm/blob/main/config/eval/solver/cem.yaml), [config/eval/pusht.yaml](https://github.com/lucas-maes/le-wm/blob/main/config/eval/pusht.yaml), [LeWM paper](https://arxiv.org/html/2603.19312)
- Cost: squared distance between the final predicted embedding and the goal-image embedding (terminal cost only). — [jepa.py `criterion`](https://github.com/lucas-maes/le-wm/blob/main/jepa.py)
- Eval: 50 episodes. The goal is the dataset state 25 env steps after the start, with a 50-step budget. Starts and goals are sampled from `pusht_expert_train`, the training set. — [config/eval/pusht.yaml](https://github.com/lucas-maes/le-wm/blob/main/config/eval/pusht.yaml), [issue #111](https://github.com/lucas-maes/le-wm/issues/111)
- Lucas Maes says evaluating on validation gave "roughly the same performance". A DINO-WM author replied that DINO-WM evaluates on a held-out validation split, not on training clips. — [le-wm issue #21](https://github.com/lucas-maes/le-wm/issues/21)

Ablations (PushT success %, from App. G)
- Predictor size: Tiny 80.67 +/- 6.54, Small (default) 96.0 +/- 2.83, Base 86.7 +/- 3.06. — [LeWM paper Table 6](https://arxiv.org/html/2603.19312)
- Predictor dropout: p=0.0 78 +/- 6.54, p=0.1 96.0 +/- 2.83, p=0.2 85.33 +/- 5.74, p=0.5 66.67 +/- 4.11. — [Table 9](https://arxiv.org/html/2603.19312)
- Adding a decoder/reconstruction loss hurts: 96.0 without, 86.0 +/- 7.54 with. — [Table 7](https://arxiv.org/html/2603.19312)
- Encoder: ViT 96.0 vs ResNet-18 94.0 +/- 3.27. — [Table 8](https://arxiv.org/html/2603.19312)
- Solver (LeWM / PLDM): CEM 96.0 / 78.0, Adam 84 / 80, RMSProp 67.33 / 49.33, SGD 26 / 4.67. — [Table 10](https://arxiv.org/html/2603.19312)
- Embedding dim: performance drops below ~184 dims and saturates above (figure only). SIGReg λ: success stays >80% for λ in [0.01, 0.2], peaks near 0.09 and degrades sharply at 0.5. The number of SIGReg projections hardly matters. — [LeWM paper App. G](https://arxiv.org/html/2603.19312)

Reproductions and third-party re-evaluations of LeWM on PushT
- Same checkpoint, 5 eval seeds x 50 episodes: 88 / 84 / 96 / 92 / 80% (a ±5-6 pp noise floor from CEM sampling). — [le-wm issue #82](https://github.com/lucas-maes/le-wm/issues/82)
- Released weights: 45/50 = 90% on Apple MPS once `transformers<5` is pinned. — [issue #80](https://github.com/lucas-maes/le-wm/issues/80)
- One user retraining from scratch got only 60% on PushT, while others matched the paper on PushT and Cube. — [issue #72](https://github.com/lucas-maes/le-wm/issues/72)
- Goal-distance sweep (G-JEPA paper), LeWM success at goals 25 / 50 / 75 steps away: PushT-Distance 90.49 / 52.99 / 16.54; PushT-Challenging 80.69 / 42.42 / 10.97. On "PushT-Ada" (256 fixed tasks, block within 20 px and 20°, 100-step budget) LeWM scores 56.90 +/- 2.35. — [G-JEPA, arXiv 2610.28483](https://arxiv.org/html/2610.28483)
- At the trained planning horizon (Ha=5) plain LeWM gets 0.98 +/- 0.01 on PushT; at twice that (Ha=10) it gets 0.53 +/- 0.03. — [Keeping JEPA World Models Plannable, arXiv 2610.03137](https://arxiv.org/html/2610.03137v1)
- LeWM 90.67 +/- 0.58 on PushT (goal offset 25, 50-step cap, 100 paired episodes, 3 seeds). — [Fast-JEPA, arXiv 2611.25927](https://arxiv.org/html/2611.25927)
- TwoRoom with the same released weights scores 82% / 61% / 34% depending only on goal construction and budget. The random-policy floor moves from 24% to 8% with it, and a 50-episode 95% CI is about ±13 points. Giving the planner 3 context frames instead of 1 changed 1 of 200 outcomes, because replanning happens only every 25 steps. — [omelas-tech/wm-eval](https://github.com/omelas-tech/wm-eval), [issue #101](https://github.com/lucas-maes/le-wm/issues/101)
- On TwoRoom, squared latent distance to the goal tracks true distance only weakly (r = 0.426), saturates around 80 units and decreases beyond 120. Replacing only the planning objective raised offset-100 success from 26.0% to 98.0%. — [Singh, "The Objective Is the Bottleneck", arXiv 2608.12959](https://arxiv.org/abs/2608.12959)

### Inferences
- The headline PushT 90-96% is a short-goal number: goals 25 env steps (5 planner steps) away, a 50-step budget, and starts/goals drawn from the training trajectories. At 50- and 75-step goals LeWM drops to ~53% and ~17%. Our ~20% on long pushes is in the same range as LeWM's long-goal numbers, so our gap to the "96%" may be smaller than it looks. Bin our success by goal distance before comparing.
- Data scale differs by about 40x. LeWM's PushT has 20,000 x 196 ≈ 3.9M env steps; we have 2,000 x 48 = 96k. That is the largest single difference from our setup (see DINO-WM's data-scaling table in Q2).
- LeWM uses no multi-step rollout loss at all (pure teacher forcing with 3-frame causal context), and its dynamics are good enough for 96% at short goals. Our 8-step rollout loss is therefore not something LeWM shows is needed. JEPA-WMs (Q3) found more than 2 rollout steps hurt in simulation.
- Predictor regularization mattered more than size: dropout 0.1 vs 0 was worth 18 points, and the "Base" predictor was worse than "Small". That fits our observation that a bigger MLP (512x3) did not help, and suggests dropout or another regularizer on our predictor is worth trying.
- Our latent is 32-d. LeWM reports degradation below ~184 dims, and notes that SIGReg struggles in low-intrinsic-dimension scenes (Two-Room). Our 2-D block scene is low intrinsic dimension and small latent, both risk factors this paper names. Our 86% (true sim + latent cost) says the cost geometry is usable, but a 32-d isotropic-Gaussian latent may make contact transitions harder to fit. Testing 64/128/192-d is cheap.
- LeWM's projector design (BatchNorm-MLP after the CLS token, BatchNorm-MLP after the predictor) is there because a final LayerNorm fights SIGReg. If our CNN ends in a LayerNorm, that is worth checking.
- For our evaluation: with 50 episodes the 95% CI is about ±13 points, so the difference between our 32% and 39% runs is probably within noise. Use ≥200 episodes per configuration.

### Gaps
- No history-length, frameskip, dataset-size, rollout-loss or EMA/stop-grad ablation for LeWM appeared in the paper text I could read. The figure values for embedding dim and λ are not given numerically.
- I did not open the LeJEPA paper (the SIGReg origin). The LeWM README lists a "LeJEPA" baseline checkpoint, but I found no PushT number for it.
- The exact Figure 6 values (GCBC, GCIQL, GCIVL on PushT) were not in the extracted text.

## Q2. DINO-WM: PushT results, predictor, history, frameskip, action encoding, data, ablations

### Takeaway
DINO-WM reaches 0.90-0.92 on PushT. It uses frozen DINOv2 ViT-S patch features (14x14x384) and a 6-layer ViT predictor with frame-level causal attention, 3 frames of history and frameskip 5. Action embeddings (10-d) and proprio (agent position + velocity) are concatenated onto every patch token. Training is teacher-forced on 18.5k noisy-expert trajectories, and planning is MPC-CEM (300 samples, top-30, 30 iterations, horizon 5) toward a goal 25 env steps away. The ablations that matter are patch features vs CLS (0.90 vs 0.44), the causal mask with history (h=3 without mask 0.08, with mask 0.92, vs 0.76 at h=1) and data scale (200 trajectories 0.08, 1k 0.48, 5k 0.72, 10k 0.88, 18.5k 0.92).

### Cited Findings
Results
- PushT success, offline world models (Table 1): IRIS 0.32, DreamerV3 0.30, TD-MPC2 0.00, DINO-WM 0.90. — [DINO-WM, arXiv 2411.04983](https://arxiv.org/html/2411.04983)
- Encoder comparison on PushT (Table 2): R3M 0.42, ResNet 0.20, DINO CLS 0.44, DINO patch 0.90. — [DINO-WM](https://arxiv.org/html/2411.04983)
- PushObj (PushT with unseen shapes, 20,000 random trajectories of 100 steps; Table 3): IRIS 0.14, DreamerV3 0.18, R3M 0.16, ResNet 0.14, DINO CLS 0.18, DINO-WM 0.34. — [DINO-WM](https://arxiv.org/html/2411.04983)
- Planning mode on PushT (Table 8): open-loop CEM 0.86, open-loop gradient descent 0.28, MPC-CEM 0.90. — [DINO-WM](https://arxiv.org/html/2411.04983)
- The paper reports 0.90 in Tables 1/2/8 but 0.92 in Tables 5/7 for the same full-data model. — [DINO-WM](https://arxiv.org/html/2411.04983)

Architecture and training
- Frozen DINOv2 patch features (14x14 patches x 384). Predictor: a ViT with the tokenizer removed, depth 6, 16 heads, MLP 2048, ~19M parameters, dropout 0.1. Causal attention is at the frame level: all patches of a frame attend to that frame and earlier frames. — [DINO-WM](https://arxiv.org/html/2411.04983), [conf/predictor/vit.yaml](https://github.com/gaoyuezhou/dino_wm/blob/main/conf/predictor/vit.yaml)
- `num_hist: 3`, `num_pred: 1` ("only supports 1"), `frameskip: 5`, `action_emb_dim: 10`, `proprio_emb_dim: 10`, `concat_dim: 1`. The action and proprio embeddings are tiled and concatenated onto each patch vector. `encoder_lr: 1e-6` with `train_encoder: False`, predictor lr 5e-4, 100 epochs, batch 32. — [conf/train.yaml](https://github.com/gaoyuezhou/dino_wm/blob/main/conf/train.yaml), [models/visual_world_model.py](https://github.com/gaoyuezhou/dino_wm/blob/main/models/visual_world_model.py)
- PushT proprio is the agent's 2-D position plus, with `with_velocity: True` (the default in `conf/env/pusht.yaml`), the agent velocity, giving 4-d proprio. Actions are relative (`rel_actions.pth`). — [datasets/pusht_dset.py](https://github.com/gaoyuezhou/dino_wm/blob/main/datasets/pusht_dset.py), [conf/env/pusht.yaml](https://github.com/gaoyuezhou/dino_wm/blob/main/conf/env/pusht.yaml)
- Loss: teacher forcing on H+1-frame segments with latent MSE on each predicted frame. The decoder is trained separately and is optional for planning. Backpropagating a decoder loss into the predictor drops PushT from 0.92 to 0.80 (Table 7). — [DINO-WM](https://arxiv.org/html/2411.04983)

Data
- PushT: "18500 samples replayed the original released expert trajectories with various level of noise", trajectory length 100-300 (Table 11), `pusht_noise` data with a 0.9 train/val split. — [DINO-WM](https://arxiv.org/html/2411.04983), [conf/env/pusht.yaml](https://github.com/gaoyuezhou/dino_wm/blob/main/conf/env/pusht.yaml)
- Data scaling on PushT (Table 5), trajectories -> success: 200 -> 0.08, 1,000 -> 0.48, 5,000 -> 0.72, 10,000 -> 0.88, 18,500 -> 0.92. — [DINO-WM](https://arxiv.org/html/2411.04983)

History and causal mask (Table 6, PushT success)
- h=1: 0.76 without mask, 0.76 with mask. h=2: 0.36 / 0.88. h=3: 0.08 / 0.92. — [DINO-WM](https://arxiv.org/html/2411.04983)

Planner and evaluation
- `plan_pusht.yaml`: MPCPlanner with `n_taken_actions: 5` and a CEM sub-planner with horizon 5, topk 30, num_samples 300, var_scale 1, opt_steps 30. Objective `mode: last` (terminal latent distance). `n_evals: 50`, `goal_source: dset`, `goal_H: 5` (5 frameskip-5 steps = 25 env steps). — [conf/plan_pusht.yaml](https://github.com/gaoyuezhou/dino_wm/blob/main/conf/plan_pusht.yaml)
- DINO-WM plans on held-out validation trajectories, per a DINO-WM author. — [le-wm issue #21](https://github.com/lucas-maes/le-wm/issues/21)

### Inferences
- Data scale is the most clearly quantified lever for PushT in this literature. Even with a frozen, internet-pretrained DINOv2 encoder, 1,000 trajectories (roughly 0.1-0.3M env steps at 100-300 steps each) give only 0.48 and 200 give 0.08. Our 96k env steps sit in the low end of that curve, so 32-39% is about what DINO-WM's own curve would predict at our data scale. Our simulator is cheap, so 10-40x more data is the most supported next step.
- The history ablation (h=1 0.76 -> h=3 0.92 with causal mask) is on pymunk PushT, where the T-block has momentum and proprio carries agent velocity. Our block is kinematic, so a single frame plus the action chunk should in principle be Markov. History may therefore matter less for us, though a 2-3 frame context is cheap to try. If we add it, every context length must be supervised (see Q1, issue #107).
- DINO-WM's predictor sees 196 patch tokens per frame, i.e. spatially resolved features. The CLS-only variant drops to 0.44. Our single 32-d vector from a small CNN is the analogue of the CLS setting, while LeWM shows a single 192-d CLS can work when the encoder is trained end-to-end. The paper does not test whether a low-dimensional global vector limits contact prediction, but it is a plausible contributor.
- Giving the predictor proprio (agent position/velocity) is something DINO-WM does on PushT. If our env exposes the pusher position, feeding it as an extra input is a cheap test (JEPA-WMs, Q3, found proprio consistently helps).

### Gaps
- No frameskip ablation, and no PushT-specific ablation of CEM sample count, in DINO-WM.
- The paper does not state whether the CEM settings in Table 10 ("100 samples per iteration and 10 optimization steps") apply to PushT. The released `plan_pusht.yaml` uses 300 samples and 30 steps, and I treat the config as authoritative.

## Q3. PLDM, V-JEPA 2-AC, and the JEPA-WM "what drives success" study: what made dynamics and planning work

### Takeaway
PLDM shows latent-dynamics planning is the most robust approach to low-quality or random offline data in navigation, but it was not tested on pushing. Its loss sums prediction error over a multi-step rollout. V-JEPA 2-AC gets real-robot pick-and-place working from 62 hours of DROID video with a 300M block-causal predictor trained with teacher forcing plus a 2-step rollout loss. The Meta FAIR JEPA-WM study is the most direct ablation work. In simulation (including PushT) the best setup is a 6-layer ViT-S predictor with AdaLN + RoPE, proprioception, a 2-step rollout loss (better than 1-step and better than longer), training context W=3, planning context W=2, and CEM with L2 cost. Bigger predictors and encoders do not help in simulation. It reaches PushT 70.2% vs DINO-WM 66.0% under its own stricter protocol.

### Cited Findings
PLDM (Sobal et al., 2025)
- Environments are Two-Rooms (64x64 point-mass), Diverse PointMaze (64x64 RGB, action repeat 4) and Ant U-Maze; there are no pushing or manipulation tasks, which are listed as future work. — [PLDM, arXiv 2502.14819](https://arxiv.org/html/2502.14819)
- Loss: L_sim + α·L_var + β·L_cov + δ·L_time-sim + ω·L_IDM (VICReg-style variance/covariance, temporal smoothness, inverse dynamics), with the similarity term summed over all predicted timesteps up to horizon H. There is no EMA target. Predictors are a 2-layer GRU (Two-Rooms), conv (PointMaze) and a 3-layer MLP ensemble of 5 (Ant). — [PLDM](https://arxiv.org/html/2502.14819)
- Loss-term ablations (Two-Rooms / Diverse Maze success): full 98.0 / 98.7; no variance 13.4 / 11.4; no covariance 29.2 / 7.8; no time-similarity 71.0 / 95.6; no IDM 98.0 / 75.5. — [PLDM](https://arxiv.org/html/2502.14819)
- Planning: MPPI with replanning every step by default, plus an ensemble-variance uncertainty penalty in the cost. — [PLDM](https://arxiv.org/html/2502.14819)
- Data: PLDM, GCIQL and HILP are most robust when good data is replaced with random-policy data. PLDM and GCIQL reach ~80% with a few thousand transitions. PLDM generalizes best to unseen maze layouts, even when trained on 5 maps. — [PLDM](https://arxiv.org/html/2502.14819)
- On PushT, LeWM's re-implementation of PLDM (7-term loss, coefficients from a 256-config grid search on PushT) gets 78.0 +/- 5.0. PLDM's loss curves are "noisy and non-monotonic" next to LeWM's. — [LeWM paper](https://arxiv.org/html/2603.19312)

V-JEPA 2-AC (Meta, 2025)
- Frozen V-JEPA 2 ViT-g encoder (16x16x1408 features per frame). Predictor: a ~300M transformer, 24 layers, 16 heads, width 1024, block-causal attention in which each frame's patches attend to that step's action and end-effector-state tokens plus all earlier steps, with 3D-RoPE. — [V-JEPA 2, arXiv 2506.09985](https://arxiv.org/html/2506.09985)
- Data: about 62 hours of raw DROID video (Franka), 4-second clips at 4 fps, 256x256 (16 frames). Actions are 7-D end-effector deltas between frames. No rewards or success labels. — [V-JEPA 2](https://arxiv.org/html/2506.09985)
- Loss: teacher-forcing L1 over 15 steps plus a 2-step rollout L1 (backprop through one recurrent step), summed unweighted. — [V-JEPA 2](https://arxiv.org/html/2506.09985)
- Planning: CEM, energy = L1 distance between the predicted final representation and the goal-image embedding. 800 samples, 10 iterations, horizon 1, about 16 s per action, first action executed (MPC). Pick-and-place uses 3 image sub-goals. — [V-JEPA 2](https://arxiv.org/html/2506.09985)
- Real-robot success (two labs, 10 trials each), V-JEPA 2-AC vs Octo: reach 100 vs 100; grasp cup 65 vs 15; grasp box 25 vs 0; pick-and-place cup 80 vs 15; box 65 vs 10. Cosmos (7B video diffusion) got 0-20% at ~4 min per action. — [V-JEPA 2](https://arxiv.org/html/2506.09985)
- Stated limitations: sensitivity to camera position (the action frame must be inferred from a monocular view), error accumulation and search blow-up at long horizons, and image-only goals. Contact dynamics are not analyzed explicitly. — [V-JEPA 2](https://arxiv.org/html/2506.09985)

JEPA-WMs: "What Drives Success in Physical Planning with Joint-Embedding Predictive World Models?" (Terver, Yang, Ponce, Bardes, LeCun; Meta FAIR, Dec 2025, v3 May 2026)
- Final comparison (Table 2, success %): PushT DINO-WM 66.0 (4.7) vs JEPA-WM 70.2 (2.8); Wall 64.1 vs 78.8; Maze 81.6 vs 83.9; Metaworld Reach 44.8 vs 58.2; DROID 39.4 (DINO-WM) / 42.9 (V-JEPA-2-AC) / 48.2 (JEPA-WM). V-JEPA-2-AC was not evaluated on PushT, and its numbers come from a retrained model with a rollout-loss bug fix. — [JEPA-WMs, arXiv 2512.24497](https://arxiv.org/html/2512.24497v3), [code](https://github.com/facebookresearch/jepa-wms)
- Recommended simulated configuration (PushT, Maze, Wall, Metaworld): DINOv2 ViT-S encoder; ViT-S predictor, depth 6; AdaLN + RoPE action conditioning; proprioception; 2-step rollout loss; training context W=3; CEM with L2 cost. The real-robot configuration instead uses DINOv3 ViT-L, a 12-layer predictor, a 6-step rollout, W=5 and no proprio. — [JEPA-WMs](https://arxiv.org/html/2512.24497v3)
- Multi-step rollout loss: in simulation, success rises from 1-step (teacher forcing) to 2-step, then drops at more steps. On DROID the optimum is 6 steps. The theory: the H-step error is bounded by δ(Λ^H − 1)/(Λ − 1), where Λ is the predictor's Lipschitz constant, and residual transformer blocks often have Λ > 1. Rollout training may lower Λ at the cost of one-step accuracy, and the best number of steps depends on environment noise. — [JEPA-WMs](https://arxiv.org/html/2512.24497v3)
- Rollout implementation details that mattered: the best variant is 2-step with "last-gradient only" and a random initial context (prefix). "Equal-order" rollouts did worse because "what matters is to train the predictor to receive as input a mix of encoder outputs and predictor outputs". Backpropagating all gradients did not beat last-gradient only. TBPTT (detaching inputs before each predictor pass) improved both success and training time. — [JEPA-WMs App.](https://arxiv.org/html/2512.24497v3)
- Context: planning with W^p = 2 gives the highest success, and "the gap between W=1 and W=2 is large". The best training W is 3 in simulation and 5 on DROID. — [JEPA-WMs](https://arxiv.org/html/2512.24497v3)
- Proprioception: models with proprio are consistently better than vision-only models. Vision-only models on Metaworld reach the goal vicinity but oscillate. — [JEPA-WMs](https://arxiv.org/html/2512.24497v3)
- Predictor and encoder size: larger encoders and wider predictors do not improve simulated performance, and depth 6 is best for most simulated tasks (possibly 3 for Wall/Maze). On DROID, size correlates positively with success. AdaLN + RoPE is best on average, but only slightly. AdaLN-zero underperforms AdaLN on DROID, PushT and Maze. — [JEPA-WMs](https://arxiv.org/html/2512.24497v3)
- Planner: CEM-L2 is best overall, and L2 beat L1 for every planner. Gradient planners (GD, Adam) do very poorly on the 2-D tasks (Wall, PushT, Maze) because they get stuck in local minima. — [JEPA-WMs](https://arxiv.org/html/2512.24497v3)
- Data scaling (2/10/50/100%): performance rises with data for all methods, most on DROID and Wall and "less clear on Push-T and Metaworld" (figure only). — [JEPA-WMs](https://arxiv.org/html/2512.24497v3)
- PushT planning settings (Table 10): N=300, H=6, m=6 actions executed, top-k 10, J=30 iterations, W^p=2, frameskip 5, max 30 env steps (one plan per episode). Goals come from the validation split, reachable in H steps. Dataset: 18,500 PushT trajectories of length 100-300. — [JEPA-WMs App.](https://arxiv.org/html/2512.24497v3)

### Inferences
- For a simulated pushing task, the best-supported rollout-loss setting is teacher forcing plus a 2-step rollout with a random ground-truth prefix and TBPTT/last-gradient. JEPA-WMs found longer rollouts reduce success in simulation, and LeWM uses no rollout at all. Our 8-step open-loop rollout is far longer than either. It may lower one-step accuracy around contacts, as JEPA-WMs' Remark 1 predicts. An ablation of 8 vs 2 vs 0 rollout steps is directly motivated.
- The JEPA-WM PushT protocol (one 30-step open-loop plan, top-10 elites) gives 66-70% for strong models, against DINO-WM's 90% under MPC with goals 25 steps away. PushT success rates are highly protocol-dependent.
- PLDM's success on random data is a navigation result, where every transition is informative about dynamics. In pushing, random-walk data rarely makes contact. The paper offers no evidence that its data-quality robustness carries over to contact-rich tasks. This is my inference, not tested in PLDM.
- V-JEPA 2-AC's working recipe on real manipulation is large data (62 h, roughly 0.9M frames at 4 fps), a large predictor, block-causal attention over history, teacher forcing plus a short 2-step rollout, and short-horizon (horizon 1) MPC with image sub-goals. Nothing there indicates long open-loop rollout losses are the key ingredient.

### Gaps
- Exact per-variant PushT success numbers for the JEPA-WM ablations (rollout steps, W, proprio, depth, data fraction) are only in figures, not in the text I extracted.
- V-JEPA 2 does not report a rollout-step or history ablation in its main text. The camera analysis in Appendix B.4 was not read.
- The PLDM paper's Appendix G DreamerV3/TD-MPC2 comparison numbers were not in the extracted text.

## Q4. Other latent-planning world models evaluated on PushT or block pushing, and their numbers

### Takeaway
A wave of 2026 follow-ups to LeWM report PushT numbers. They mostly add training-time grounding signals to an otherwise LeWM-like model: inverse dynamics (LeWM+I), physical-state supervision plus temporal context (G-JEPA), or learned action proposals (G-JEPA AD, Fast-JEPA). The largest gains appear at long goal distances or hard tasks, not on the 25-step protocol, where everything saturates near 90-98%. Reconstruction-based baselines (DreamerV3 0.30, IRIS 0.32, TD-MPC2 0.00 in DINO-WM's PushT table) do much worse than latent-JEPA planners.

### Cited Findings
- DINO-WM Table 1 PushT baselines: IRIS 0.32, DreamerV3 0.30, TD-MPC2 0.00 (DINO-WM 0.90). On PushObj: IRIS 0.14, DreamerV3 0.18. — [DINO-WM](https://arxiv.org/html/2411.04983)
- LeWM+I ("Keeping JEPA World Models Plannable When Little of the Frame Moves", arXiv 2610.03137):
  - SLIM is a 2-D pymunk pushing benchmark with small objects. Under random actions only 0.37% of the frame changes per predictor step, vs 0.64% on PushT, and objects move 24-50x less than PushT's T-block. Data: 10,000 x 200-step episodes from a scripted primitive policy. — [arXiv 2610.03137](https://arxiv.org/html/2610.03137v1)
  - Plain LeWM on SLIM: 3 of 1,200 trials succeed. Probes show the latent ignores the action: action sensitivity 3.8e-4, 10-step rollout error 12.4x the true per-step motion, pusher-position probe error 163 px (vs 165 px for a constant predictor), CEM cost coefficient of variation 0.001 (a flat landscape). — [arXiv 2610.03137](https://arxiv.org/html/2610.03137v1)
  - Fix: add an inverse-dynamics head (3-layer MLP, width 256), with an encoder-side term ||h(z_t, z_{t+1}) − a_t||² and a predictor-side term ||h(sg(z_t), ẑ_{t+1}) − a_t||², weights (SIGReg 0.09, 1, 1). SLIM success rises to 0.348 +/- 0.016. Encoder-IDM alone gives 0.290, predictor-IDM alone 0.010, a shuffled-action null 0.003, frozen DINOv2-S 0.048, frozen DINOv2-S + IDM 0.163. — [arXiv 2610.03137](https://arxiv.org/html/2610.03137v1)
  - PushT: at the trained horizon (Ha=5), plain 0.98 vs LeWM+I 0.97. At Ha=10, plain 0.53 +/- 0.03 vs LeWM+I 0.83 +/- 0.01. — [arXiv 2610.03137](https://arxiv.org/html/2610.03137v1)
  - Planning protocol: a path cost (mean over the rollout) beats a terminal cost, 0.335 vs 0.083 on a PushT checkpoint. Replanning cadence matters: PushT run under SLIM's cadence "collapses both models". An action-sensitivity (spread/motion) below ~0.05 always failed to plan. — [arXiv 2610.03137](https://arxiv.org/html/2610.03137v1)
- G-JEPA ("Exploring Temporal Context-Aware Physical Grounding in Latent World Models", arXiv 2610.28483): adds a causal S4D temporal module used only in training, supervision of physical state (PushT: agent position and velocity, block position and angle), photometric-invariance loss, and optionally a diffusion action prior (AD) that fills 20% of CEM candidates.
  - PushT-Ada: G-JEPA w/o AD 79.43 +/- 1.58, G-JEPA 87.11 +/- 1.17, LeWM 56.90 +/- 2.35, TD-JEPA 56.64 +/- 2.71, PSG-JEPA 17.32 +/- 1.58. — [arXiv 2610.28483](https://arxiv.org/html/2610.28483)
  - Goal distance 25/50/75: G-JEPA w/o AD 96.09 / 80.34 / 47.53 vs LeWM 90.49 / 52.99 / 16.54. — [arXiv 2610.28483](https://arxiv.org/html/2610.28483)
  - Ablations on PushT-Ada: no grounding 56.90; state only 55.73; state + invariance 79.43; full 79.43. Temporal module: S4D 78.12, GRU 69.14, causal transformer 61.46, attention-only 53.39. — [arXiv 2610.28483](https://arxiv.org/html/2610.28483)
- Fast-JEPA (arXiv 2611.25927): an anchored multi-horizon predictor (predicts horizons k=1..5 directly from the observed latent, no recursive rollout) plus a learned action generator refined by CEM (N=64, I=2, K=8). PushT: LeWM 90.67, Fast-LeWM 90.33, Fast-LeWM+SC 92.00, Fast-JEPA 90.00, at 10.88 ms vs 1,729.94 ms planning latency (averaged over tasks). Without the proposal generator, Fast-JEPA drops to 45.00 on PushT. — [Fast-JEPA](https://arxiv.org/html/2611.25927)
- D-JEPA (arXiv 2609.24749): a search snippet reports 87.89% PushT success on a 256-start cohort. I did not open the paper, so the protocol is unverified. — [D-JEPA](https://arxiv.org/html/2609.24749v1)
- C-JEPA (Causal-JEPA, ICML 2026, same group as LeWM): object-level latent masking. It claims "comparable performance" on agent control while using 1% of the latent input features of patch-based world models. The abstract gives no PushT numbers. — [C-JEPA, arXiv 2602.11389](https://arxiv.org/abs/2602.11389)
- Same-weights LeWM TwoRoom success has been reported as 8%, 14%, 84%, 87%, 94% and 97.4% across papers, depending on goal construction and budget. — [omelas-tech/wm-eval](https://github.com/omelas-tech/wm-eval)

### Inferences
- The two follow-ups that most improve long-goal PushT (LeWM+I at 2x horizon: 0.53 -> 0.83; G-JEPA at 75-step goals: 17 -> 48-55%) both add a training signal that forces the latent and predictor to carry action-relevant, physically grounded state (inverse dynamics, or explicit agent/block state targets). This matches our diagnosis that the latent cost is fine but learned contact dynamics are weak.
- Two cheap additions are indicated: an encoder-side inverse-dynamics loss, and, since our sim gives us block and pusher state for free, an auxiliary state-regression head on z and ẑ. Note that in SLIM the predictor-side IDM alone did nothing (0.010), and in G-JEPA state supervision alone did nothing (55.73) without the invariance term. The gains are not guaranteed to transfer.
- The LeWM+I diagnostics (action sensitivity of rollouts, rollout error vs true motion, block-position probe on predicted latents, CEM cost coefficient of variation) can be run on our model now, to check whether our predictor under-reacts to pushes.
- Cross-paper PushT numbers are not comparable. Each paper's protocol (goal offset, budget, success threshold, replanning cadence, train vs val starts) moves LeWM between ~57% and 98%.

### Gaps
- No Navigation World Models, TD-JEPA (as a primary source) or Dreamer-style PushT numbers beyond those quoted inside DINO-WM and G-JEPA. I did not fetch the TD-JEPA or PSG-JEPA papers themselves.
- D-JEPA, Slot-JEPA and FF-JEPA turned up in search but were not read.
- I found no paper that isolates a kinematic (no-momentum) block vs a dynamic one, or that studies contact transitions specifically, e.g. success split by whether the plan requires making or breaking contact.

## Q5. What these works say about frame history/velocity, multi-step rollout losses, and data quantity/quality for pushing

### Takeaway
Every PushT success in this literature uses a short context of 2-3 frames with frameskip 5. History helps strongly on dynamic PushT (DINO-WM: 0.76 -> 0.92 at h=3 with a causal mask; JEPA-WMs: a large W=1 -> W=2 gap), but the evidence comes from a block with momentum and agent-velocity proprio. Multi-step rollout losses help only briefly: LeWM uses none, and JEPA-WMs finds 2 steps best in simulation and more steps worse. Data quantity is the strongest quantified lever (DINO-WM PushT: 200 trajectories 0.08, 1k 0.48, 18.5k 0.92). The successful PushT models all train on millions of frames of noisy-expert, contact-dense data.

### Cited Findings
History and velocity
- DINO-WM PushT: h=1 0.76; h=2 0.88 and h=3 0.92 with a frame-causal mask; without the mask, h=2 0.36 and h=3 0.08. — [DINO-WM Table 6](https://arxiv.org/html/2411.04983)
- DINO-WM's PushT proprio includes agent velocity (`with_velocity: True`). — [datasets/pusht_dset.py](https://github.com/gaoyuezhou/dino_wm/blob/main/datasets/pusht_dset.py)
- JEPA-WMs: planning context W^p=2 is best, with a large W=1 vs W=2 gap. Training W=3 is best in simulation and W=5 on DROID. — [JEPA-WMs](https://arxiv.org/html/2512.24497v3)
- LeWM: history 3 for PushT/Cube and 1 for TwoRoom. All context positions are supervised, which matters because planning starts from one frame. — [LeWM paper](https://arxiv.org/html/2603.19312), [issue #107](https://github.com/lucas-maes/le-wm/issues/107)
- On TwoRoom with replanning every 25 steps, 3 vs 1 planner context frames made no measurable difference (199/200 same outcomes). — [wm-eval](https://github.com/omelas-tech/wm-eval)
- G-JEPA's temporal module helps PushT-Ada (S4D 78.12 vs attention-only 53.39), but the paper does not isolate velocity or history length. — [G-JEPA](https://arxiv.org/html/2610.28483)

Multi-step rollout losses
- LeWM: pure one-step teacher forcing (`num_preds: 1`), PushT 96%. — [train.py](https://github.com/lucas-maes/le-wm/blob/main/train.py)
- DINO-WM: teacher forcing only (`num_pred: 1`), PushT 0.90-0.92. — [conf/train.yaml](https://github.com/gaoyuezhou/dino_wm/blob/main/conf/train.yaml)
- JEPA-WMs: 2-step rollout is best in simulation, longer is worse; 6-step is best on real DROID video. Last-gradient-only, random prefix and TBPTT were the best variants. — [JEPA-WMs](https://arxiv.org/html/2512.24497v3)
- V-JEPA 2-AC: teacher forcing over 15 steps + a 2-step rollout L1. — [V-JEPA 2](https://arxiv.org/html/2506.09985)
- PLDM: similarity loss summed over the full multi-step rollout (navigation only). — [PLDM](https://arxiv.org/html/2502.14819)
- Fast-JEPA avoids recursive rollout entirely by training direct predictions at horizons 1..5 from the observed latent, and stays at ~90% on PushT. — [Fast-JEPA](https://arxiv.org/html/2611.25927)

Data quantity
- DINO-WM PushT data scaling: 200 / 1k / 5k / 10k / 18.5k trajectories -> 0.08 / 0.48 / 0.72 / 0.88 / 0.92. — [DINO-WM Table 5](https://arxiv.org/html/2411.04983)
- LeWM PushT: 20,000 expert episodes x ~196 steps. — [LeWM paper](https://arxiv.org/html/2603.19312)
- JEPA-WMs: performance rises with data fraction for all methods; the PushT trend is "less clear" (figure only). — [JEPA-WMs](https://arxiv.org/html/2512.24497v3)
- PLDM: PLDM and GCIQL reach ~80% with a few thousand transitions in Two-Rooms navigation. — [PLDM](https://arxiv.org/html/2502.14819)

Data quality
- PushT data in DINO-WM, LeWM and JEPA-WMs is noisy replays of expert (human-demonstration) trajectories, so most frames involve purposeful pushing. — [DINO-WM](https://arxiv.org/html/2411.04983), [LeWM](https://arxiv.org/html/2603.19312)
- PushObj (unseen shapes, trained on 20,000 random 100-step trajectories) gets only 0.34 even for DINO-WM. — [DINO-WM Table 3](https://arxiv.org/html/2411.04983)
- PLDM is robust to random or low-quality data in navigation. — [PLDM](https://arxiv.org/html/2502.14819)
- LeWM lists sufficient coverage as a requirement, and notes low diversity weakens SIGReg in simple environments. — [LeWM paper](https://arxiv.org/html/2603.19312)
- SLIM (scripted primitive policy, 10k x 200 steps, small objects that move rarely) breaks plain LeWM entirely (3/1,200). — [arXiv 2610.03137](https://arxiv.org/html/2610.03137v1)

Planner settings common to the PushT successes
- CEM ~300 samples, 10-30 elites, 30 iterations, horizon 5-6 blocks of 5 env steps, a terminal latent L2 cost (LeWM, DINO-WM), and the full plan executed before replanning. — [LeWM cem.yaml](https://github.com/lucas-maes/le-wm/blob/main/config/eval/solver/cem.yaml), [DINO-WM plan_pusht.yaml](https://github.com/gaoyuezhou/dino_wm/blob/main/conf/plan_pusht.yaml), [JEPA-WMs Table 10](https://arxiv.org/html/2512.24497v3)
- Gradient-based planning is clearly worse on PushT: DINO-WM GD 0.28 vs CEM 0.86 open-loop; LeWM SGD 26 vs CEM 96. — [DINO-WM](https://arxiv.org/html/2411.04983), [LeWM](https://arxiv.org/html/2603.19312)

### Inferences
Ranked by how well the literature supports each change, given our diagnosis that dynamics are weak and the cost is fine:
1. More and contact-denser data. Our 96k env steps are about 40x below LeWM/DINO-WM PushT, in the range where DINO-WM's own curve gives roughly 0.1-0.5 success. Our 30% random-walk share probably adds few contact transitions. Raising the dataset 10-40x and checking the fraction of transitions where the block moves is the best-evidenced first step.
2. Shorter rollout loss. Replace the 8-step open-loop rollout loss with teacher forcing plus a 2-step rollout (random prefix, TBPTT/last-gradient), per JEPA-WMs. LeWM and DINO-WM reach ~90%+ with teacher forcing alone.
3. Predictor regularization and form. LeWM's dropout 0 -> 0.1 was worth ~18 points, and bigger predictors were not better (both LeWM and JEPA-WMs). A small causal transformer with AdaLN over a 2-3 latent context is the architecture all PushT successes share. If we keep the MLP, add dropout before scaling it up.
4. Larger latent. LeWM degrades below ~184 dims, and SIGReg struggles with low intrinsic dimension. Try 64-192 dims, re-checking that the "true sim + latent cost" number stays high.
5. Action and physical grounding. Add an encoder-side inverse-dynamics loss (LeWM+I) and/or an auxiliary block/pusher state-regression head (G-JEPA style). Run the action-sensitivity diagnostic first; values below ~0.05 predicted planning failure in SLIM.
6. History. Probably lower priority for a kinematic, fully observed block. DINO-WM's history gain was on a block with momentum, and in TwoRoom it made no difference at 25-step replanning. If added, supervise all context lengths.
7. Evaluation hygiene. Report success binned by goal distance (the published ~90-96% is at 25-step goals; ~53% at 50 and ~17% at 75 for LeWM), use ≥200 episodes, and measure a random-policy floor on the same episodes.
8. Planner checks. Our running cost is already in line with LeWM+I's path-cost finding. Compare predicted vs actual cost of CEM-chosen plans, since a large optimism gap means the optimizer is exploiting model error (le-wm #107 saw 27x). This can matter more than CEM sample count, which matches our finding that more samples did not help.

### Gaps
- No paper ablates history on a kinematic, momentum-free pushing task, so whether history matters for our scene is untested in this literature.
- No paper reports a direct comparison of 8-step vs 2-step rollout losses on PushT with exact numbers in text; the JEPA-WMs curve is figure-only.
- No study I found reports success specifically on contact-initiating segments, or compares expert vs random data proportions for pushing at matched data size.
- Exact data-scaling numbers for LeWM (vs DINO-WM) on PushT were not found.
