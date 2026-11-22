# LOG

## 2026-11-22
- 05:22 Started. Machine check: WSL2 laptop, RTX 3050 Laptop 4 GB, 10.7 GB RAM, 16 threads. No reverse proxy. -> research/SERVER.md
- 05:35 Licenses: le-wm MIT, stable-worldmodel MIT, stable-pretraining MIT, eb_jepa Apache-2.0, dino_wm MIT, PLDM MIT,
  vjepa2 MIT. **lejepa is CC BY-NC 4.0** and **keon/jepa has no license** -> read only, don't copy code.
- 05:50 Read LeWM source (jepa.py, module.py, train.py, configs). Paper config: ViT-tiny/14 at 224 px, embed 192,
  history 3, 6-layer AdaLN predictor, lambda=0.09, SIGReg 17 knots on [0,3], 1024 projections, AdamW 5e-5, bf16, CEM
  300 samples / 30 iters / top-30, horizon 5 x frameskip 5.
- 06:05 Decision: own 2-D env for the demo (exact JS port possible), LeWM-style objective, small CNN+MLP model. -> PLAN.md
