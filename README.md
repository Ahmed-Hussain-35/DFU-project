# DFU Analysis Console

Self-contained web app for diabetic foot ulcer analysis. One FastAPI service serves
both the ML inference API **and** the React frontend — no database, no separate build step.

**Pipeline (ported verbatim from the trained notebooks):**
segmentation (U-Net++ / EfficientNet-B4, FP16 ONNX) → wound feature extraction →
infection severity → Wagner grade.

## Setup

1. Put your model file **`dfu_model_fp16.onnx`** in this folder (next to `main.py`).
2. Install and run:

```bash
pip install -r requirements.txt
uvicorn main:app --port 8000
```

3. Open **http://localhost:8000** — upload a foot image, get results.

That's it. Deliverables 1 (segmentation + grading), 2 (infection severity), and
4 (mobile-friendly inference) are all live in this single app.

## What you see

- **Viewer** with view-mode tabs: Overlay (red fill), Boundary (yellow contour),
  Mask (binary), Original.
- **Readout**: infection severity, Wagner grade, ulcer coverage %, infection score,
  and a per-feature breakdown (redness, necrosis, pus, swelling, irregularity, size).

## API (for reference / Postman)

- `GET /health` → `{status, model}`
- `POST /analyze` (multipart `file`) → JSON with `coverage_pct`, `infection`,
  `wagner`, `features`, `inference_ms`, and base64 `images.{overlay,boundary,mask}`.

```bash
curl -F "file=@foot.jpg" http://localhost:8000/analyze
```

## Config (env vars)

- `DFU_MODEL` — path to the ONNX model (default `dfu_model_fp16.onnx`).

## Notes

- Threshold is `0.45` and the infection weights / Wagner thresholds match the notebooks
  exactly. Change `THRESHOLD` / `MIN_AREA` at the top of `main.py` if needed.
- The frontend loads React + fonts from CDNs, so first load needs internet.
- CORS is open (`*`) for local dev — restrict `allow_origins` before deploying.
- Deploy split (when you add a DB later): frontend+API on Railway/Render (the 42 MB
  model bundles fine there); avoid Vercel serverless for the model.
