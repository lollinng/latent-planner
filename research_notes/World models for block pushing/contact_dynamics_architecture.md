# Architecture choices for learned visual dynamics in pushing and contact

Scope: which representation and predictor choices make visual world models predict pushing and contact well, with the reported evidence, and which of them could fit a tiny browser model (ONNX, about 1-3 MB, about 100 ms per plan on CPU). Our baseline for comparison: a LeWM-style JEPA with a small CNN, 64x64 input, one 32-d global latent, an MLP predictor taking z_t plus a 4-step action chunk, 8-step rollout loss and SIGReg. It reaches 32-39% CEM success. With the true simulator and our latent cost it reaches 86%, and a linear probe localizes the block to about 4 px and the agent to about 3.6 px.

Protocol caveat for every PushT number below: papers use different PushT evaluation protocols, so their numbers are not comparable with each other. DINO-WM reports itself at 0.86 (CEM) and 0.90 (MPC) ([DINO-WM](https://arxiv.org/abs/2411.04983)). LeWM reports DINO-WM at 92.0 ± 1.6% ([LeWM](https://arxiv.org/html/2603.19312)). The JEPA-WMs study reports DINO-WM at 66.0 ± 4.7% under its own protocol (H=6, at most 30 steps) ([Terver et al.](https://arxiv.org/html/2512.24497v3)). PushT is also harder than our scene: it uses a T-shaped block that rotates, and the agent is moved by a PD controller.

## 1. Spatial vs global latents (patch, feature-map, slot, keypoint vs one vector)

### Takeaway
A frozen, pretrained global token does much worse than patch tokens on PushT (DINO-WM: CLS 0.44 vs patches 0.90). An end-to-end-trained global token can still match patch models (LeWM: one 192-d CLS token, 96% on PushT vs DINO-WM 92%), but only with a 192-d latent, a roughly 10M-parameter transformer predictor, 3 frames of history and 224x224 input. LeWM itself reports that performance drops below about 184 latent dims. Slot latents (4 slots of 128-d) reach 85% on PushT, and their planning success tracks how cleanly the slots separate objects (r = 0.96).

### Cited Findings
- **DINO-WM encoder ablation, Table 2** (planning success rate, frozen encoders, same ViT predictor). PushT: R3M 0.42, ResNet 0.20, DINO CLS 0.44, DINO patch 0.90. Wall: 0.34 / 0.12 / 0.58 / 0.96. Reach: 0.40 / 0.06 / 0.60 / 0.92. PointMaze is saturated (0.94-0.98 for all) — [DINO-WM](https://arxiv.org/abs/2411.04983)
- DINO-WM prediction quality on PushT (LPIPS, lower is better): R3M 0.045, ResNet 0.063, DINO CLS 0.039, DINO patch 0.007 (about 5.6x lower than CLS). SSIM: 0.956 / 0.950 / 0.973 / 0.985 — [DINO-WM](https://arxiv.org/abs/2411.04983)
- DINO-WM generalization (Table 3). PushObj: DINO CLS 0.18 vs patch 0.34. WallRandom: 0.64 vs 0.82 — [DINO-WM](https://arxiv.org/abs/2411.04983)
- DINO-WM latent size: 14x14x384 DINOv2 patch features (196 tokens) from 224 images (the appendix says 196x196, which conflicts with the 224 setting). Planning with CEM (100 samples x 10 iterations) takes 53 s on an A6000 — [DINO-WM](https://arxiv.org/abs/2411.04983)
- **LeWM** uses one global vector: the CLS token of a ViT-tiny (about 5M params, patch 14, 224x224 input), then a 1-layer MLP projector with BatchNorm, giving a 192-d latent. It encodes about 200x fewer tokens than DINO-WM and plans about 48x faster (under 1 s per plan) — [LeWM](https://arxiv.org/html/2603.19312)
- LeWM PushT success, three seeds: LeWM 96.0 ± 2.83, DINO-WM 92.0 ± 1.63, PLDM 78.0 ± 5.0. LeWM loses to PLDM and DINO-WM on Two-Room and slightly to DINO-WM on OGBench-Cube — [LeWM](https://arxiv.org/html/2603.19312)
- LeWM embedding-dimension sweep: "performance drops below roughly 184 dimensions and saturates above that threshold". The exact values are only in a figure (Fig. 15) — [LeWM](https://arxiv.org/html/2603.19312)
- LeWM PushT probing (Table 1), Pearson r for block location: linear 0.986 for LeWM, 0.997 for DINO-WM, 0.938 for PLDM; MLP probes reach 0.999 for LeWM. For block angle, linear r is 0.902 for LeWM vs 0.979 for DINO-WM. MSE units are not stated in the extracted text, so they cannot be compared with our 4 px — [LeWM](https://arxiv.org/html/2603.19312)
- LeWM stated limitation: low data diversity weakens SIGReg in simple, low-dimensional environments (the Two-Room case) — [LeWM](https://arxiv.org/html/2603.19312)
- JEPA-WMs study conclusion: "fine-grained image encoders" are among the factors that matter most. DINO encoders beat V-JEPA. Larger encoders do not help on simulated tasks — [Terver et al., What Drives Success in Physical Planning with JEPA-WMs](https://arxiv.org/html/2512.24497v3)
- **Slots on PushT** ("Better Slots, Better Worlds"): SlotContrast-WM (4 slots x 128-d on frozen DINOv3-S features, 1-layer 4-head transformer predictor, frameskip 5, CEM with 300 samples x 30 iterations) reaches 84.7 ± 1.9% without proprioception and 85.3 ± 3.4% with it. The VideoSAUR (C-JEPA-style) slot encoder reaches 74.7 ± 1.9% without proprioception. Planning success correlates with video FG-ARI at r = 0.96 and with mBO at r = 0.94 — [Better Slots, Better Worlds](https://arxiv.org/html/2608.12078)
- Same paper: all models fail when the block's shape changes, which changes the contact dynamics. LeWM's single CLS token collapses under appearance and frame-level shifts. Exact DINO-WM and LeWM numbers there are only in figures — [Better Slots, Better Worlds](https://arxiv.org/html/2608.12078)
- C-JEPA (object-level masking over slot latents): comparable control performance with "only 1% of the total latent input features required by patch-based world models". No PushT numbers in the abstract — [C-JEPA](https://arxiv.org/abs/2602.11389)

### Inferences
- The DINO-WM gap (0.44 vs 0.90) mostly says that a frozen, generic CLS token discards spatial detail. It does not prove that global vectors cannot work: LeWM's trained global token matches patches. The structural reason patches help is that position stays explicit (which token, which location), so the predictor does not have to decode coordinates from an entangled vector.
- Our latent is 32-d, about 6x below LeWM's reported knee of about 184 dims. Our scene has fewer degrees of freedom (agent x,y and block x,y, maybe an angle), so information capacity is not the issue in principle. But SIGReg pushes the 32 dims toward an isotropic Gaussian, and LeWM notes SIGReg is weaker in simple, low-dimensional scenes. Positions may end up encoded along a curved, nonlinear manifold, which would fit both our coarse linear read-out (about 4 px) and an MLP predictor that does not get better with width. Sweeping latent dim (64, 128, 192) is cheap, because the predictor's cost sits in its hidden layers.
- 4 px of error in a 64 px world is likely about the same size as the agent radius and the contact gap. If the latent cannot tell "touching" from "2-4 px away", the predictor cannot learn when contact starts, so it averages the outcomes of contact and no contact. That is exactly where a free-motion-accurate model fails at pushing. The true-simulator-plus-latent-cost result (86%) fits this reading: the cost tolerates 4 px, but contact onset does not.

### Gaps
- I found no paper that directly ablates "learned global vector vs learned 8x8 feature map" with the same small CNN on a pushing task. DINO-WM's comparison uses frozen pretrained encoders only.
- I did not retrieve exact LeWM latent-dim sweep values (they are in a figure only) or exact DINO-WM and LeWM numbers in the slot paper (also figures).
- SlotFormer and OCVT: not fetched in this pass, so no numbers. Their benchmarks (OBJ3D, CLEVRER, Physion, PHYRE) are passive video, not action-conditioned pushing.

## 2. Object-centric and structured world models for pushing (C-SWM, slots, keypoints, particles, graph networks)

### Takeaway
Structure helps most when it makes object positions explicit and models agent-object interaction pairwise. In C-SWM, removing the object factorization drops 10-step accuracy on a pushing grid world from 99.9% to 24.1%. In "Keypoints into the Future", a plain 2x500 MLP on keypoint coordinates pushes as accurately as one on ground-truth state (about 1.1-1.2 cm), while autoencoder and Transporter latents are about 2x worse in position and 3-4x worse in angle. Slot Attention can fail through binding errors: two cubes in one slot gave about 0% success.

### Cited Findings
- **C-SWM setup.** 2D Shapes and 3D Blocks are 5 objects on a 5x5 grid. Each action moves one object one cell, and the push has no effect if the target cell is blocked (an object-object interaction). The object extractor for 2D Shapes is tiny: one 10x10 conv with 16 maps, then a 1x1 conv to 5 maps. Each object latent is 2-d (4-d for Atari and physics). The transition is a GNN with one round of message passing on a fully connected graph, with the one-hot action given to the acted-on object, predicting a residual update — [C-SWM, Kipf et al. 2020](https://arxiv.org/abs/1911.12247)
- **C-SWM ablations**, 2D Shapes, H@1 at 1/5/10 steps:
  - full C-SWM: 100 / 100 / 99.9
  - per-object MLP instead of the latent GNN: 99.9 / 97.4 / 89.7
  - no factored states: 54.5 / 34.4 / 24.1
  - pixel loss instead of contrastive: 49.9 / 6.5 / 1.4
  - autoencoder world model: 98.7 / 36.1 / 6.5
  - VAE world model: 94.2 / 14.1 / 1.4

  The 3D Blocks pattern is similar (GNN removed: 86.0 at 10 steps). Gaps show up mainly in multi-step rollouts — [C-SWM](https://arxiv.org/abs/1911.12247)
- C-SWM on Atari: K=1 sometimes beats K=5 (Pong 10-step H@1: K=1 11.5 vs K=5 5.3), and the paper reports that slots are less interpretable and results vary between runs. On 3-body physics (no actions), C-SWM has only a slight edge over the autoencoder at 10 steps (75.5 vs 67.9) — [C-SWM](https://arxiv.org/abs/1911.12247)
- **Keypoints into the Future** (Manuelli, Li, Florence, Tedrake; simulated box pushing). The dynamics model is an MLP with 2 hidden layers of 500 units. Its state is keypoint locations plus robot proprioception, with history 1, planned with MPPI (1000 samples, H=10, 3 iterations). Closed-loop pose error, top-down camera, cm / degrees:
  - ground-truth 3D points: 1.14 / 4.01
  - keypoint variants: 1.12-1.19 / 3.88-4.43
  - Transporter 2D: 2.08 / 13.59
  - Transporter 3D: 2.01 / 15.95
  - autoencoder: 2.18 / 14.79

  On mugs (category-level generalization): keypoints 1.0-1.4 cm vs autoencoder 9.05 cm / 56.84° — [Keypoints into the Future](https://arxiv.org/abs/2009.05085)
- Keypoints into the Future, hardware: about 10 minutes of random pushing data, success (within 3 cm and 30°) of 65-89% across four trajectories. The paper credits the physically grounded, low-dimensional keypoint state, correspondence pretraining, and weighting or selecting reliable, well-spread keypoints (the occlusion task: DS 3.66 cm vs WSDS 1.28 cm) — [Keypoints into the Future](https://arxiv.org/abs/2009.05085)
- **Entity-Centric RL** (Deep Latent Particles with an Entity Interaction Transformer), success from images: 2 cubes 0.968, 3 cubes 0.919, adjacent goals 0.710 (the state-based SMORL baseline: 0.980 / 0.932 / 0.716). With Slot Attention, 1 cube matched DLP, but 2 cubes gave about 0% because both cubes were often put in one slot. The unstructured β-VAE single-vector baseline "fails to learn at all" from images with more than one object. Note this is model-free RL, not a learned dynamics model — [Haramati, Daniel, Tamar, ICLR 2024](https://arxiv.org/html/2404.01220v1)
- Slot quality and planning success on PushT correlate at r = 0.96 (FG-ARI), with gains saturating at high slot quality — [Better Slots, Better Worlds](https://arxiv.org/html/2608.12078)
- **Hybrid analytical models** for planar pushing: a CNN maps raw depth or point clouds to the inputs of an analytical push model. The abstract claims a "significantly" reduced training-data need and better generalization to novel interactions than a pure neural network. No numbers are in the abstract — [Kloss et al.](https://arxiv.org/abs/1710.04102)

### Inferences
- The common thread: a dynamics MLP is good enough once the state is a precise, low-dimensional list of positions (keypoints at ground-truth level). Errors come from the representation, not the predictor's capacity. This matches our finding that a 512x3 or 1024x3 MLP barely helps.
- Our scene has exactly two entities (agent and block), probably of distinct colors and shapes. That is the easy case for slots or keypoints, so the binding failure seen with 2+ identical cubes is less of a risk. A 2-entity latent (for example 2 x 8-d) with a pairwise "edge" MLP (agent, block, action → block update) is a direct C-SWM analogue and very small.
- Relative coordinates (block minus agent) fed to the interaction function are a standard inductive bias in interaction-network models. They should make contact onset a simple threshold on distance. This is my inference, not a sourced result.

### Gaps
- Interaction Networks, Propagation Networks and DPI-Net on pushing were not fetched this pass, so I have no numbers.
- Transporter's own paper (Kulkarni et al. 2019) was not fetched. Its numbers above come only from its use as a baseline in Keypoints into the Future.
- I found no head-to-head of slots vs keypoints vs a global vector inside one JEPA-style action-conditioned model on our kind of 2-D pushing scene.

## 3. Predictor design for contact (transformer vs MLP, action conditioning, deltas, history, chunk length, rollout loss)

### Takeaway
Among the levers the papers measure, frame history (going from 1 to 2-3 frames) and proprioception input have the clearest effects. AdaLN action conditioning helps only slightly. Rollout-loss length has an optimum of about 2 steps in simulation, and longer is worse. Predictor size has a sweet spot (in LeWM, ViT-S beats both Tiny and Base), and predictor dropout 0.1 lifts LeWM's PushT result from 78% to 96%. All the strong PushT models use frameskip 5 (5 env steps per prediction), so a 4-step chunk is not unusual. No paper isolates the chunk length's effect on contact.

### Cited Findings
- **History, DINO-WM Table 6** (PushT success). History h=1: 0.76. h=2: 0.36 without a causal mask, 0.88 with it. h=3: 0.08 without the mask, 0.92 with it — [DINO-WM](https://arxiv.org/abs/2411.04983)
- JEPA-WMs: "a big gap between W=1 and W=2" in context length. The best W is 3 for simulated tasks and 5 for DROID — [Terver et al.](https://arxiv.org/html/2512.24497v3)
- **Proprioception**: models with proprioception are "consistently better" than vision-only models across environments (numbers are in a figure) — [Terver et al.](https://arxiv.org/html/2512.24497v3). On PushT slots, adding proprioception gives 84.7 → 85.3% (nms=0) and 72.0 → 82.7% (nms=1) for SlotContrast-WM, and 74.7 → 85.3% for VideoSAUR with the full C-JEPA recipe — [Better Slots, Better Worlds](https://arxiv.org/html/2608.12078). In Keypoints into the Future, the dynamics state is keypoints plus robot proprioception — [Keypoints into the Future](https://arxiv.org/abs/2009.05085)
- **Action conditioning**: the candidates compared were feature-concatenation with sincos, sequence (token) conditioning with RoPE, feature-concatenation with RoPE, AdaLN with RoPE, and AdaLN-zero with RoPE. AdaLN+RoPE has the best average, "but the margin is slight". Feature-concatenation with sincos is best on Metaworld. AdaLN-zero is worse than AdaLN on DROID, Push-T and Maze — [Terver et al.](https://arxiv.org/html/2512.24497v3). DINO-WM simply concatenates an MLP action embedding (dim 10) to every patch token, with no ablation — [DINO-WM](https://arxiv.org/abs/2411.04983). LeWM uses zero-initialized AdaLN in a 6-layer, 16-head transformer predictor (about 10M params) — [LeWM](https://arxiv.org/html/2603.19312). Visual Foresight concatenates actions into all conv layers, not only the bottleneck — [Visual Foresight](https://arxiv.org/abs/1812.00568)
- **Predictor size**, LeWM PushT: Tiny 80.67 ± 6.54, Small (ViT-S) 96.0 ± 2.83, Base 86.7 ± 3.06. Dropout p=0.0: 78 ± 6.54. p=0.1: 96.0. p=0.2: 85.33. p=0.5: 66.67 — [LeWM](https://arxiv.org/html/2603.19312). JEPA-WMs: deeper predictors do not help on simulated tasks; the recommended depth is 6 for simulation and 12 for DROID — [Terver et al.](https://arxiv.org/html/2512.24497v3)
- **Rollout loss**: performance rises from pure teacher forcing to a 2-step rollout loss, then drops on simulated environments. On DROID the optimum moves to 6 steps. Candidates were 1, 2, 3 and 6 predictor steps, each spanning frameskip 5 — [Terver et al.](https://arxiv.org/html/2512.24497v3)
- **Decoder or reconstruction loss hurts planning**: DINO-WM PushT goes 0.92 → 0.80 when the decoder loss is backpropagated into the predictor — [DINO-WM](https://arxiv.org/abs/2411.04983). LeWM PushT goes 96.0 → 86.0 with a decoder loss — [LeWM](https://arxiv.org/html/2603.19312). In C-SWM, a pixel loss instead of the contrastive loss gives 10-step H@1 of 1.4 vs 99.9 — [C-SWM](https://arxiv.org/abs/1911.12247)
- **Residual (delta) prediction**: the C-SWM transition GNN predicts a residual latent update — [C-SWM](https://arxiv.org/abs/1911.12247). Visual Foresight predicts 2-D flow that warps earlier frames, rather than generating pixels, arguing that this reuses object appearance and gives motion directly — [Visual Foresight](https://arxiv.org/abs/1812.00568)
- **Frameskip and chunk settings**:
  - DINO-WM uses frameskip 5 for PushT, PushObj, Reacher, PointMaze and Wall, and 1 for Rope and Granular. There is no frameskip ablation — [DINO-WM](https://arxiv.org/abs/2411.04983)
  - LeWM: frameskip 5, history 3 on PushT — [LeWM](https://arxiv.org/html/2603.19312)
  - JEPA-WMs PushT: f=5, H=6 planning steps, 6 actions executed per replan, at most 30 steps — [Terver et al.](https://arxiv.org/html/2512.24497v3)
  - SlotContrast-WM: frameskip 5, horizon 5 (25 env steps) — [Better Slots, Better Worlds](https://arxiv.org/html/2608.12078)
- **Planner**: CEM beats gradient methods on PushT. LeWM: CEM 96.0, Adam 84, RMSProp 67.3, SGD 26. JEPA-WMs PushT: CEM-L2 70.2, NG-L2 48.0, Adam-L2 9.0, GD-L2 9.1. DINO-WM: MPC 0.90, CEM 0.86, GD 0.28 — [LeWM](https://arxiv.org/html/2603.19312), [Terver et al.](https://arxiv.org/html/2512.24497v3), [DINO-WM](https://arxiv.org/abs/2411.04983)
- **Data**: DINO-WM PushT success rises with the number of training trajectories: 200 → 0.08, 1000 → 0.48, 5000 → 0.72, 10000 → 0.88, 18500 → 0.92 — [DINO-WM](https://arxiv.org/abs/2411.04983)

### Inferences
- Our predictor sees only z_t, which matches DINO-WM's h=1 setting (0.76, vs 0.92 at h=3). Our scene is kinematic with no momentum, so in principle the current state alone determines the next one, and history should matter less than in PushT (which has PD control and block inertia). But history can still help when the single-frame latent localizes poorly: two frames give the predictor a second, noisy look at positions and the approach direction. It is cheap to test (input z_{t-4}, z_t).
- Proprioception is the cheapest, best-supported lever for us. The browser demo knows the agent's exact position. Feeding it as input, or replacing the agent part of the latent with exact coordinates, removes 3.6 px of agent error from the contact geometry.
- An 8-step rollout loss (32 env steps) is well beyond the JEPA-WMs simulation optimum (2 predictor steps, 10 env steps). A long rollout loss can push the predictor toward smooth, averaged predictions that blur contact onset. Worth sweeping 1, 2, 4 and 8.
- Action chunk: with 4 env steps per prediction, contact can begin at any of the 4 substeps. That makes the z_t → z_{t+4} map piecewise and harder for an MLP. The reference models use 5-step chunks and still reach 85-96%, but with transformers, 192-d+ latents and history. A test with chunks of 1-2 steps (more predictor calls per plan) would isolate whether chunking is part of our gap. Feeding the 4 substep actions as a sequence (rather than flattened) is a possible intermediate step. Both are inferences, not sourced.
- AdaLN or FiLM conditioning in an MLP costs almost nothing (a small action MLP produces per-layer scale and shift). The evidence says the gain is small but positive on PushT relative to AdaLN-zero; concatenation is close behind.
- Dropout 0.1 in the predictor (78 → 96 in LeWM) is a free, cheap change for a small model.

### Gaps
- No source found that ablates action-chunk or frameskip length specifically for contact accuracy.
- No source found on mixture or multimodal outputs to handle contact discontinuities in JEPA-style latent predictors. Visual Foresight's stochastic aspects were not covered in what I fetched.
- No direct MLP vs transformer comparison on the same latent for pushing. LeWM's "Tiny" predictor is still a transformer.

## 4. Classic visual pushing models (Visual Foresight / SNA, Learning to Poke, MIT Push, deep dynamics models)

### Takeaway
Classic pixel-space pushing models worked at 64x64 by predicting motion (flow) rather than appearance, and by tracking where specific "designated pixels" go. Their main failure mode was drift in the predicted object position over longer pushes, which re-localizing the object at every step largely fixed (20% → 66% on long pushes). Learning to Poke found a feature-space forward model useful mainly as a regularizer, and reported that pose accuracy for poking was "not satisfactory" even with 100K pokes.

### Cited Findings
- **Visual Foresight / SNA**: a recurrent model of stacked conv-LSTMs that predicts 2-D flow fields to warp previous frames. Temporal skip connections transform pixels from frames up to two steps back and from the first frame, so objects occluded by the arm reappear (DNA, which uses only the previous frame, "overwrites" occluded objects). It uses 64x64 images, with actions concatenated into every conv layer — [Visual Foresight, Ebert et al. 2018](https://arxiv.org/abs/1812.00568)
- Visual Foresight, multi-object pushing (Table 1): improvement in moved-object distance DNA 0.83 ± 0.25 px vs SNA 10.6 ± 0.82 px — [Visual Foresight](https://arxiv.org/abs/1812.00568)
- Visual Foresight, long-distance pushing (20 tasks, 30 cm, success within 15 px ≈ 7.5 cm): predictor propagation 20%, OpenCV tracking 45%, registration network 66%. Short-distance (15 cm): 83% for all. Open-loop comparison: Visual MPC 75% vs calibrated-camera baseline 18.75% — [Visual Foresight](https://arxiv.org/abs/1812.00568)
- **Learning to Poke by Poking**: AlexNet-like conv features. The forward model predicts the next image's features (L1 loss), not pixels. More than 400 hours of robot time collected over 100K pokes on 16 objects. In simulation, joint inverse+forward training beat inverse-only at 10K and 20K examples, with parity at 100K. "The forward model in turn regularizes the feature space for the inverse model". Pose accuracy "is not satisfactory". Contact outcomes are hard because "real world objects have complex geometric and material properties" — [Agrawal et al. 2016](https://arxiv.org/abs/1606.07419)
- **Hybrid learned + analytical planar-push models** need less data and generalize better to novel interactions than pure neural models (abstract claim, no numbers retrieved) — [Kloss et al.](https://arxiv.org/abs/1710.04102)

### Inferences
- Visual Foresight's jump from 20% to 66% when the object is re-localized each step matches our setup's structure. MPC with re-encoding at every replan limits drift, so single-step contact accuracy matters more than long-horizon accuracy. Fewer, shorter replans also help.
- Flow and designated-pixel prediction amounts to an explicitly spatial, equivariant dynamics model. That is the same structural point as patches and keypoints: keep "where" explicit.

### Gaps
- No numbers retrieved for MIT Push dataset learned models (Bauza & Rodriguez GP models, Ajay et al.) or for Kloss et al.'s errors.
- "Deep Dynamics Models for Learning Dexterous Manipulation" (PDDM) was not covered. It is state-based, so less relevant.
- Learning to Poke's position and angle error numbers are only in a figure.

## 5. Resolution, encoder capacity and latent precision for object localization

### Takeaway
No source directly tests "8x8 feature map vs global vector" for localization in a tiny CNN world model. The closest evidence: CNNs are poor at turning pixel positions into coordinates unless given coordinate channels (CoordConv: 150x faster, 10-100x fewer parameters, perfect generalization on that task). Spatial-softmax feature points were designed to track task-relevant object positions for manipulation. LeWM's ResNet-18 encoder is almost as good as its ViT (94 vs 96%), so encoder type matters less than keeping position explicit. Bigger encoders do not help on simulated tasks.

### Cited Findings
- **CoordConv**: convolutional nets "fail spectacularly" at mapping between (x, y) coordinates and one-hot pixel space. Adding x and y coordinate channels solves this with perfect generalization, 150x faster and with 10-100x fewer parameters. It also reports benefits for Atari agents — [Liu et al. 2018](https://arxiv.org/abs/1807.03247)
- **Deep spatial autoencoders** learn "a set of feature points that describe the environment", capturing object positions. The controller "learns to track task-relevant objects and manipulate their configuration with the robot's arm" — [Finn et al. 2016](https://arxiv.org/abs/1509.06113)
- Encoder type: LeWM with a ViT encoder gets 96.0 ± 2.83 on PushT, and with ResNet-18 94.0 ± 3.27 — [LeWM](https://arxiv.org/html/2603.19312). JEPA-WMs: larger encoders do not help on simulated tasks (DINOv2-S is recommended for simulation) — [Terver et al.](https://arxiv.org/html/2512.24497v3)
- Latent size: LeWM performance drops below about 184 dims and saturates above — [LeWM](https://arxiv.org/html/2603.19312). DINO patch (196 x 384) vs CLS (384): PushT 0.90 vs 0.44 — [DINO-WM](https://arxiv.org/abs/2411.04983). Slots: 4 x 128 on PushT reaches 85% — [Better Slots, Better Worlds](https://arxiv.org/html/2608.12078). C-JEPA: object slots use 1% of the latent features of patch models, with comparable performance — [C-JEPA](https://arxiv.org/abs/2602.11389)
- Pixel-space pushing worked at 64x64 resolution with flow-based prediction — [Visual Foresight](https://arxiv.org/abs/1812.00568). C-SWM's 2-D object latents (2-d per object) give near-perfect 10-step prediction on its 5x5 grid pushing world — [C-SWM](https://arxiv.org/abs/1911.12247)

### Inferences
- The evidence points to structure, not size: slots at 1% of patch features, ResNet roughly equal to ViT, and bigger encoders not helping. Our 4 px error most likely comes from how a global-pooled CNN encodes position (a translation-invariant net plus a dense layer has to learn coordinates indirectly), not from 64x64 resolution. At 64x64 a 1 px object shift is visible, so sub-pixel localization is physically possible.
- Background, not verified in this session: spatial softmax (Levine/Finn) computes a softmax over each conv channel's H x W map and outputs the expected (x, y) per channel. That is exactly coordinate regression. A 32-d latent made of 16 spatial-softmax keypoints keeps our latent size but makes every dimension a precise position.
- An 8x8 feature map latent should localize at about 8 px per cell, and finer through channel interpolation. It also multiplies predictor cost about 64x if channel counts stay fixed (see section 6).
- A training-time auxiliary probe loss (predict true agent and block positions from z, using the simulator's state) would directly force precise localization. It is privileged information that costs nothing at inference. This is my inference: no source tested it in a JEPA world model. The evidence that a decoder loss hurts planning (DINO-WM, LeWM) applies to pixel reconstruction, not to a low-dimensional position probe, but it is a caution that auxiliary losses can trade off against planning quality.

### Gaps
- No direct experiment found on latent resolution (feature-map size) vs object localization error in a small CNN world model.
- No paper found measuring a world model's linear-probe localization error in pixels against its pushing success.

## 6. Fit for a tiny browser model (ONNX ~1-3 MB, ~100 ms per plan on CPU)

### Takeaway
The best-supported changes that also fit the budget are cheap ones that make positions explicit: give the predictor the agent's exact position, use a spatial-softmax keypoint or 2-slot latent with a pairwise interaction MLP, and add CoordConv channels. Patch or feature-map latents with a transformer predictor (DINO-WM, LeWM, slot models) carry the strongest evidence but are 10-1000x over budget as published.

### Cited Findings (sizes and costs of the reference models)
- DINO-WM: frozen DINOv2 encoder with 196 x 384 patch tokens. CEM planning takes 53 s on an A6000 GPU — [DINO-WM](https://arxiv.org/abs/2411.04983)
- LeWM: about 15M parameters (about 5M ViT-tiny encoder, about 10M predictor), under 1 s per plan, about 48x faster than DINO-WM — [LeWM](https://arxiv.org/html/2603.19312)
- SlotContrast-WM: frozen DINOv3-S features (256 tokens x 384), 4 slots x 128, 1-layer transformer predictor, CEM with 300 samples x 30 iterations — [Better Slots, Better Worlds](https://arxiv.org/html/2608.12078)
- Keypoints into the Future: an MLP with 2 x 500 hidden units on keypoints plus proprioception, pushing about as accurately as ground-truth state — [Keypoints into the Future](https://arxiv.org/abs/2009.05085)
- C-SWM 2-D Shapes: a one-layer 10x10 conv extractor, 2-d object latents, and a one-round message-passing GNN — [C-SWM](https://arxiv.org/abs/1911.12247)

### Inferences (my assessment; parameter counts are my estimates)
Budget: 1-3 MB is about 0.25-0.75M parameters in fp32 or about 1-3M in int8. Per-plan cost is about (CEM samples x iterations x horizon) x (cost of one predictor call), so the predictor's cost per call dominates.

1. **Proprioception or exact agent position as predictor input.** Zero extra size, negligible cost. Evidence: JEPA-WMs (consistently better) and slot PushT results (72 → 83%). It directly removes agent localization error from contact geometry. Highest priority.
2. **Spatial-softmax keypoint encoder** (for example 16 keypoints → a 32-d (x, y) latent) plus **CoordConv channels**. About the same size and cost as the current CNN. Evidence: Keypoints into the Future (an MLP on keypoints matches ground-truth state), Deep Spatial Autoencoders, CoordConv. Caveat: SIGReg on keypoint coordinates may conflict with their bounded, structured nature. An option is to apply SIGReg only to a projected head, or to rely on the prediction loss plus a probe loss. This needs testing.
3. **Factored 2-entity latent with a pairwise interaction MLP** (C-SWM-style: per-entity encoder, edge MLP on (z_agent, z_block, relative position, action), node update predicts a residual). With hidden width 128-256 this is about 50-150k parameters, and per-call cost is similar to or below the current MLP. Evidence: C-SWM factored vs not (99.9 vs 24.1 at 10 steps) and GNN vs per-object MLP (99.9 vs 89.7). Risk: unsupervised slot binding. With two visually distinct entities, a fixed 2-channel extractor (as in C-SWM) should bind reliably.
4. **History of 2 latents, FiLM/AdaLN action conditioning, dropout 0.1, shorter rollout loss (2-4 steps).** Nearly free in size; history at most doubles the predictor's input. Evidence: DINO-WM (0.76 → 0.92 with h=3), JEPA-WMs (W1 → W2 gap; AdaLN slight gain; best rollout length 2), LeWM (dropout 78 → 96).
5. **Larger latent (64-192 d).** Small size increase (only the first and last layers grow). Evidence: LeWM knee at about 184 dims, though for a harder scene.
6. **Feature-map latent (for example 8x8xC) with a conv or tiny-transformer predictor.** The strongest structural evidence (DINO-WM patch vs CLS) but the most expensive: at C=16 the latent is 1024-d, and a 3-layer 3x3 conv predictor at 16-32 channels costs about 1-5M MACs per call vs about 0.3-1M for our MLP. CEM rollouts would be 3-10x slower, likely over 100 ms unless the sample count or horizon is cut. Feasible as a stretch option at 4x4 or with fewer CEM samples.
7. **Frozen DINO, ViT encoders, or a LeWM-size transformer predictor.** Not feasible at 1-3 MB (DINOv2-S alone is about 21M parameters, and LeWM is about 15M).

Suggested test order: 1, then 4 (cheap ablations), then 2 or 3 (structural changes). Re-check the linear-probe error after each change: a change that brings block and agent localization to about 1 px should show whether representation precision is the cause of the contact errors.

### Gaps
- No source reports browser or ONNX latencies for any of these architectures. All per-call cost figures above are my estimates, not measurements.
- No published tiny (<1M parameter) JEPA world model with pushing results was found to anchor feasibility directly.
