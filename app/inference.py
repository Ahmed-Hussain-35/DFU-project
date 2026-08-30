"""
Real DFU segmentation + infection pipeline, wired for the clinical platform.
Auto-detects .pth (torch) or .onnx (onnxruntime) via DFU_MODEL env var.
Run uvicorn from the project root so the relative model path resolves correctly.
"""
import os, base64
import numpy as np
import cv2

MODEL_PATH = os.getenv("DFU_MODEL", "dfu_model_fp16.onnx")
IMG_SIZE   = 384
THRESHOLD  = 0.45
MIN_AREA   = 100
ENCODER    = "efficientnet-b4"
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

_BACKEND = "torch" if MODEL_PATH.lower().endswith((".pth", ".pt")) else "onnx"

if _BACKEND == "torch":
    import torch
    import segmentation_models_pytorch as smp
    _device = "cuda" if torch.cuda.is_available() else "cpu"
    _model = smp.UnetPlusPlus(
        encoder_name=ENCODER, encoder_weights=None,
        in_channels=3, classes=1, activation=None,
        decoder_attention_type="scse",
    ).to(_device)
    _ckpt = torch.load(MODEL_PATH, map_location=_device)
    _model.load_state_dict(_ckpt["model"] if "model" in _ckpt else _ckpt)
    _model.eval()

    def _infer(x_nchw):
        with torch.no_grad():
            t = torch.from_numpy(x_nchw).to(_device)
            return torch.sigmoid(_model(t)).cpu().numpy()
else:
    import onnxruntime as ort
    _sess = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
    _in = _sess.get_inputs()[0].name

    def _infer(x_nchw):
        out = _sess.run(None, {_in: x_nchw})[0]
        return 1.0 / (1.0 + np.exp(-out))


def check_image_quality(image):
    warnings = []
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
    if blur_score < 60:
        warnings.append(f"Image appears blurry (sharpness {blur_score:.0f})")
    brightness = gray.mean()
    if brightness < 40:
        warnings.append(f"Image is too dark (brightness {brightness:.0f}/255)")
    elif brightness > 220:
        warnings.append(f"Image is overexposed/glare (brightness {brightness:.0f}/255)")
    h, w = image.shape[:2]
    if min(h, w) < 200:
        warnings.append(f"Resolution too low ({w}x{h}) — retake closer, higher-res")
    return (len(warnings) == 0), warnings


def segment_ulcer(image):
    orig_h, orig_w = image.shape[:2]
    resized = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    x = resized.astype(np.float32) / 255.0
    x = (x - MEAN) / STD
    x = np.transpose(x, (2, 0, 1))[None].astype(np.float32)
    probs = _infer(x)
    mask = (probs[0, 0] > THRESHOLD).astype(np.uint8)
    mask = _postprocess(mask)
    return cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)


def _postprocess(mask):
    k1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k1)
    k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k2)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    cleaned = np.zeros_like(mask)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= MIN_AREA:
            cleaned[labels == i] = 1
    return cleaned


def analyze_features(image, mask):
    if mask.sum() == 0:
        return {k: 0.0 for k in ["Redness", "Necrosis", "Pus", "Swelling", "Irregularity", "Size"]}
    pix = image[mask == 1]
    R, G, B = pix[:, 0].astype(float), pix[:, 1].astype(float), pix[:, 2].astype(float)
    redness = float(np.mean(np.clip(R - (G + B) / 2, 0, 255)) / 255 * 100)
    brightness = np.mean(pix, axis=1)
    necrosis = float((brightness < 60).sum() / len(pix) * 100)
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)[mask == 1]
    yellow = ((hsv[:, 0] >= 15) & (hsv[:, 0] <= 45) & (hsv[:, 1] > 50) & (hsv[:, 2] > 60))
    pus = float(yellow.sum() / len(hsv) * 100)
    coverage = (mask.sum() / (image.shape[0] * image.shape[1])) * 100
    size = 10 if coverage < 1 else 30 if coverage < 5 else 60 if coverage < 10 else 80 if coverage < 15 else 100
    irregular = float(np.clip(np.std(pix) / 80 * 100, 0, 100))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    surrounding = cv2.dilate(mask, kernel) - mask
    if surrounding.sum() > 0:
        spix = image[surrounding == 1]
        sR, sG, sB = spix[:, 0].astype(float), spix[:, 1].astype(float), spix[:, 2].astype(float)
        swelling = float(np.mean(np.clip(sR - (sG + sB) / 2, 0, 255)) / 255 * 100)
    else:
        swelling = 0.0
    return {"Redness": redness, "Necrosis": necrosis, "Pus": pus,
            "Swelling": swelling, "Irregularity": irregular, "Size": size}


def compute_infection(features, coverage):
    if coverage < 1.0:
        for k in features:
            features[k] = 0.0
        return {"score": 0.0, "level": "NONE / MINIMAL", "color": "#27ae60",
                "override": "Coverage too small to assess infection"}
    r, n, p, s, i, sz = (features["Redness"], features["Necrosis"], features["Pus"],
                         features["Swelling"], features["Irregularity"], features["Size"])
    score = (min(r * 2.5, 100) * 0.18 + min(n * 2.5, 100) * 0.22 + min(p * 4.0, 100) * 0.18 +
             min(s * 2.5, 100) * 0.15 + min(i * 1.8, 100) * 0.12 + sz * 0.15)
    floor, reason = 0, None
    if n > 30:                                     floor, reason = 75, f"Necrosis {n:.1f}%"
    elif n > 20:                                   floor, reason = 60, f"Necrosis {n:.1f}%"
    if p > 15 and 70 > floor:                      floor, reason = 70, f"Pus {p:.1f}%"
    if sz >= 80 and (r > 15 or i > 45) and 60 > floor:
        floor, reason = 60, f"Large wound (size {sz:.0f})"
    override = f"{reason} raised score to {floor:.0f}" if floor > score else None
    score = float(np.clip(max(score, floor), 0, 100))
    if score < 20:   level, color = "NONE / MINIMAL", "#27ae60"
    elif score < 40: level, color = "MILD INFECTION", "#f39c12"
    elif score < 65: level, color = "MODERATE INFECTION", "#e67e22"
    else:            level, color = "SEVERE \u2014 urgent clinician review suggested", "#c0392b"
    return {"score": round(score, 2), "level": level, "color": color, "override": override}


def _b64_png(rgb):
    ok, buf = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    return "data:image/png;base64," + base64.b64encode(buf).decode()


def build_overlays(image, mask):
    overlay = image.astype(np.float32) / 255.0
    red = overlay.copy(); red[mask == 1] = [1, 0, 0]
    blended = ((cv2.addWeighted(overlay, 0.6, red, 0.4, 0)) * 255).astype(np.uint8)
    boundary = image.copy()
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(boundary, contours, -1, (255, 255, 0), 3)
    mask_rgb = cv2.cvtColor((mask * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
    return _b64_png(blended), _b64_png(boundary), _b64_png(mask_rgb)


def get_severity(coverage_pct):
    if coverage_pct == 0: return "NO ULCER DETECTED"
    if coverage_pct < 1:  return "MILD"
    if coverage_pct < 5:  return "MODERATE"
    if coverage_pct < 15: return "SEVERE"
    return "CRITICAL"


def analyze(image):
    quality_ok, quality_warnings = check_image_quality(image)
    mask = segment_ulcer(image)
    feats = analyze_features(image, mask)
    coverage = float(mask.sum() / (image.shape[0] * image.shape[1]) * 100)
    infection = compute_infection(feats, coverage)
    severity = get_severity(coverage)
    overlay, boundary, mask_png = build_overlays(image, mask)
    return {
        "coverage_pct": round(coverage, 3),
        "infection": infection,
        "severity_label": severity,
        "features": {k: round(v, 2) for k, v in feats.items()},
        "image_quality": {"ok": quality_ok, "warnings": quality_warnings},
        "images": {"overlay": overlay, "boundary": boundary, "mask": mask_png},
    }
