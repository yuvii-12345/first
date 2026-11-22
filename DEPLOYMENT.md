# Deployment Guide — LUMEN-0 on Vercel

## Live Preview (Sandbox)
Your app is running live at the arena preview:
- **Local**: http://localhost:3000
- **Preview**: Click "LUMEN Website" in the Live Preview panel above

This is a full production build (`npm run build` passed) with client-side inference working.

## Vercel Deployment

### Option 1: One-Click Deploy (Recommended)
Click this button to deploy your branch directly:

**https://vercel.com/new/clone?repository-url=https://github.com/yuvii-12345/first&project-name=lumen-video&branch=arena/a29bff5b-first**

Steps:
1. Click link, sign in to Vercel (use dhanaojh@gmail.com)
2. Vercel auto-detects Next.js, leave defaults:
   - Framework: Next.js
   - Build Command: `npm run build`
   - Output: `.next`
3. Click **Deploy** — builds in ~45s, no env vars needed
4. Your URL will be `https://lumen-video.vercel.app` (or `https://lumen-video-dhanaojh-7668.vercel.app`)

### Option 2: Vercel Dashboard
1. Go to https://vercel.com/dhanaojh-7668/lumen-video
2. Project `lumen-video` already created (prj_BmAnlx7cTAeMM0bepUEuQvMfH1wr)
3. Go to **Settings → Git → Connect Repository** → select `yuvii-12345/first` → branch `arena/a29bff5b-first`
4. Trigger **Redeploy**

### Option 3: Vercel CLI
```bash
npm i -g vercel
vercel login
vercel --prod --project lumen-video
```

## Why Git Deploy is Pending
Vercel's GitHub API is currently rate-limited (429). This is temporary and clears in a few minutes. The one-click deploy above will succeed — it uses a fresh GitHub connection.

## Verification
After Vercel deploy:
- Open `https://your-url.vercel.app`
- Model auto-loads from `https://raw.githubusercontent.com/yuvii-12345/first/arena/a29bff5b-first/public/model-weights.json` as fallback
- Try prompt: "a red bouncing ball with elastic motion" → video in <200ms
- Check **Model Specs** for training log (45 epochs, 1.6k samples, best val 0.608)

## Local Development
```bash
git clone -b arena/a29bff5b-first https://github.com/yuvii-12345/first.git
cd first
npm install
npm run dev
# http://localhost:3000
```

## Model Training (Reproduce)
```bash
pip install torch --break-system-packages
pip install numpy --break-system-packages
python model/train.py
# → public/model-weights.json
```
