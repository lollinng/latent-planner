# Server / machine check (2026-11-22)

This "server" is a **Windows laptop running Ubuntu under WSL2**, not a cloud box.

| Item | Value |
|---|---|
| OS | Ubuntu 26.04.1 LTS on WSL2 (kernel 6.18.33.2-microsoft-standard-WSL2) |
| CPU | AMD Ryzen 7 4800H, 16 threads |
| RAM | 10.7 GiB visible to WSL (about 6.5 GiB free with current services), 8 GiB swap |
| GPU | NVIDIA GeForce RTX 3050 Laptop GPU, **4 GB VRAM**, compute capability 8.6, driver 566.07, CUDA 12.7 |
| Measured GPU throughput | ~3.3 TFLOPS fp32, ~5.8 TFLOPS fp16 (4096² matmul, torch 2.14.1+cu126) |
| Disk | 1 TB, 828 GB free |
| Python | system 3.14.4; uv-managed 3.10 to 3.13; project venv uses **3.12** (`.venv`, created with uv) |
| Node | v22.22.1, npm 9.x/npx 9.2.0 |
| Browser | `google-chrome` installed (headless testing) |
| Docker | installed (not used by this project) |
| Reverse proxy | **none** (no nginx/caddy) |
| gh | logged in; the personal account `lollinng` owns this project |

## What this means for the project
- **GPU is usable** but small (4 GB). Training LeWM at its paper settings (ViT-tiny at 224 px, batch 128, ~15M params)
  would need small batches and many hours here. A small model on 64x64 frames trains in minutes. This is why the
  live demo uses our own 2-D environment and a ~1M-param model (see `PLAN.md`).
- **No reverse proxy and no public server.** It is a laptop behind NAT, so self-hosting the demo is a bad idea.
  The personal site `prathameshjadhav.in` is GitHub Pages from `lollinng/lollinng.github.io` (CNAME file present).
  A GitHub Pages project site in the `lollinng` account would be served at
  `https://prathameshjadhav.in/latent-planner/` with **no DNS change**. Needs a public repo (ask first).
- Nothing on this machine was modified apart from creating `~/personal/latent-planner/`.
