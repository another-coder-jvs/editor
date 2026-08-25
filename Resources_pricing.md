
────────────────────────────────────────────────────────────────────────────────

🏗️ Architecture Overview

┌───────────┬───────────────────────────────┬───────────────────────────────┐
│ Component │ What it runs                  │ Where it goes                 │
├───────────┼───────────────────────────────┼───────────────────────────────┤
│ Frontend  │ React + Vite (static build)   │ Static host (free tier works) │
│ Backend   │ FastAPI + PyTorch + AI models │ GPU cloud server              │
└───────────┴───────────────────────────────┴───────────────────────────────┘

────────────────────────────────────────────────────────────────────────────────

📦 Required Resources

Backend (the expensive part)

┌──────────────┬──────────────────────────────────────────────┬─────────────────────┐
│ Resource     │ Minimum                                      │ Recommended         │
├──────────────┼──────────────────────────────────────────────┼─────────────────────┤
│ GPU VRAM     │ 24 GB (RTX 3090/4090)                        │ 48 GB (A6000/A5000) │
│ System RAM   │ 32 GB                                        │ 64 GB               │
│ Disk Storage │ 50 GB (model weights ~17 GB + temp/projects) │ 100 GB              │
│ CPU          │ 8 cores                                      │ 16+ cores           │
│ OS           │ Linux (Ubuntu 22.04+ with CUDA 11.8+)        │ Same                │
└──────────────┴──────────────────────────────────────────────┴─────────────────────┘

Why 24GB VRAM minimum? Your SDXL inpainting model alone needs ~6 GB, plus Grounding DINO, SAM2, and Real-ESRGAN loaded simultaneously. FLUX.1-dev needs even more.

Frontend (cheap/free)

┌───────────┬───────────────────────┐
│ Resource  │ Minimum               │
├───────────┼───────────────────────┤
│ Storage   │ ~10 MB (static build) │
│ Bandwidth │ Depends on users      │
│ Runtime   │ None (static files)   │
└───────────┴───────────────────────┘

────────────────────────────────────────────────────────────────────────────────

💰 Deployment Options & Pricing

Option 1: RunPod (Recommended — best price/performance)

┌─────────────────┬──────────┬───────┬──────────┬──────────────────────┐
│ Tier            │ GPU      │ VRAM  │ Price    │ Monthly (8h/day use) │
├─────────────────┼──────────┼───────┼──────────┼──────────────────────┤
│ Community Cloud │ RTX 3090 │ 24 GB │ $0.22/hr │ ~$53/mo              │
│ Community Cloud │ RTX 4090 │ 24 GB │ $0.34/hr │ ~$82/mo              │
│ Secure Cloud    │ RTX 4090 │ 24 GB │ $0.69/hr │ ~$166/mo             │
│ Community Cloud │ A5000    │ 24 GB │ $0.26/hr │ ~$62/mo              │
└─────────────────┴──────────┴───────┴──────────┴──────────────────────┘

Storage: $0.07/GB/month (~$3.50 for 50 GB)

Frontend: Free via Cloudflare Pages or Vercel

Total estimated: $55–$170/month (depending on GPU tier and usage hours)

────────────────────────────────────────────────────────────────────────────────

Option 2: Vast.ai (Cheapest — marketplace model)

┌───────────┬──────────┬────────────────┬──────────────────┐
│ Tier      │ GPU      │ Price          │ Monthly (8h/day) │
├───────────┼──────────┼────────────────┼──────────────────┤
│ On-demand │ RTX 3090 │ $0.10–$0.20/hr │ ~$25–$48/mo      │
│ On-demand │ RTX 4090 │ $0.20–$0.40/hr │ ~$48–$96/mo      │
└───────────┴──────────┴────────────────┴──────────────────┘

Total estimated: $25–$100/month

⚠️ Vast.ai is community-sourced, so reliability varies.

────────────────────────────────────────────────────────────────────────────────

Option 3: Lambda Labs

┌───────────────────┬──────────┬──────────────────┐
│ GPU               │ Price    │ Monthly (8h/day) │
├───────────────────┼──────────┼──────────────────┤
│ RTX A6000 (48 GB) │ $1.10/hr │ ~$264/mo         │
│ A100 (40 GB)      │ $1.29/hr │ ~$310/mo         │
└───────────────────┴──────────┴──────────────────┘

Total estimated: $270–$315/month

────────────────────────────────────────────────────────────────────────────────

Option 4: Google Colab Pro (Budget / Personal Use)

┌────────────┬────────┬─────────────────────────────┐
│ Plan       │ Price  │ GPU                         │
├────────────┼────────┼─────────────────────────────┤
│ Colab Free │ $0     │ T4 (15 GB VRAM) — too small │
│ Colab Pro  │ $10/mo │ T4/V100 — marginal          │
│ Colab Pro+ │ $50/mo │ A100 (40 GB) — works        │
└────────────┴────────┴─────────────────────────────┘

⚠️ Colab has session timeouts (12h max), not ideal for production.

────────────────────────────────────────────────────────────────────────────────

Frontend Hosting (Free)

┌──────────────────┬───────┬────────────────────────────────────┐
│ Platform         │ Price │ Notes                              │
├──────────────────┼───────┼────────────────────────────────────┤
│ Cloudflare Pages │ $0    │ Unlimited bandwidth, 500 builds/mo │
│ Vercel (Hobby)   │ $0    │ 100 GB bandwidth, non-commercial   │
│ Netlify          │ $0    │ 100 GB bandwidth                   │
│ GitHub Pages     │ $0    │ Static only, 1 GB storage          │
└──────────────────┴───────┴────────────────────────────────────┘

────────────────────────────────────────────────────────────────────────────────

🏆 Recommended Setup (Minimum Cost)

┌───────────────────┬───────────────────────────────┬─────────────────────────────┐
│ Component         │ Platform                      │ Cost                        │
├───────────────────┼───────────────────────────────┼─────────────────────────────┤
│ Backend           │ RunPod Community (RTX 3090)   │ $0.22/hr ($53/mo at 8h/day) │
│ Storage           │ RunPod Network Volume (50 GB) │ ~$3.50/mo                   │
│ Frontend          │ Cloudflare Pages              │ $0                          │
│ Domain (optional) │ Cloudflare Registrar          │ ~$10/year                   │
│ Total             │                               │ ~$57/month                  │
└───────────────────┴───────────────────────────────┴─────────────────────────────┘

If you use it only a few hours/day:

┌──────────────────┬──────────────┬──────────┐
│ Usage            │ Backend Cost │ Total    │
├──────────────────┼──────────────┼──────────┤
│ 2h/day           │ ~$13/mo      │ ~$17/mo  │
│ 4h/day           │ ~$27/mo      │ ~$31/mo  │
│ 8h/day           │ ~$53/mo      │ ~$57/mo  │
│ 24/7 (always on) │ ~$160/mo     │ ~$164/mo │
└──────────────────┴──────────────┴──────────┘

────────────────────────────────────────────────────────────────────────────────

⚡ Quick Start on RunPod

1. Deploy a GPU Pod with RTX 3090 (24 GB), CUDA 11.8, 32+ GB RAM, 50 GB disk
2. Clone repo →  pip install -r requirements.txt  → download models
3. Run  uvicorn main:app --host 0.0.0.0 --port 8000 
4. Build frontend:  cd frontend && npm run build  → serve via nginx or deploy to Cloudflare Pages
5. Point frontend  config.ts   baseUrl  to your RunPod pod's public IP
