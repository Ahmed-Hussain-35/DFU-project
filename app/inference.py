
"""
Real DFU segmentation + infection pipeline for Phase 1.
Includes 30-day erosion (healing) & dilation (worsening) simulation
with Green/Pink visual overlays.
"""
import os, base64, warnings
import numpy as np
import cv2

MODEL_PATH = "ml model/best_model.pth"
IMG_SIZE   = 384
THRESHOLD  = 0.45
MIN_AREA   = 100
ENCODER    = "efficientnet-b4"
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

NONE_THR = 20
MILD_THR = 40
MOD_THR  = 65

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


# =====================================================================
# 1. QUALITY CHECK & PREPROCESSING
# =====================================================================
def check_image_quality(image):
    warnings_list = []
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
    if blur_score < 60:
        warnings_list.append(f"Image appears blurry (sharpness {blur_score:.0f})")
    brightness = gray.mean()
    if brightness < 40:
        warnings_list.append(f"Image is too dark (brightness {brightness:.0f}/255)")
    elif brightness > 220:
        warnings_list.append(f"Image is overexposed/glare (brightness {brightness:.0f}/255)")
    h, w = image.shape[:2]
    if min(h, w) < 200:
        warnings_list.append(f"Resolution too low ({w}x{h}) — retake closer, higher-res")
    return (len(warnings_list) == 0), warnings_list


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


# =====================================================================
# 2. FEATURE ANALYSIS & SEVERITY COMPUTATION
# =====================================================================
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
    if coverage < 0.05:
        return {"score": 0.0, "level": "NONE / MINIMAL", "color": "#27ae60",
                "override": "Coverage too small to assess infection"}
    r, n, p, s, i, sz = (features["Redness"], features["Necrosis"], features["Pus"],
                         features["Swelling"], features["Irregularity"], features["Size"])
    
    redness_b   = min(r * 2.5, 100)
    necrosis_b  = min(n * 2.5, 100)
    pus_b       = min(p * 4.0, 100)
    swelling_b  = min(s * 2.5, 100)
    irregular_b = min(i * 1.8, 100)
    
    score = (redness_b * 0.15 + necrosis_b * 0.30 + pus_b * 0.20 +
             swelling_b * 0.15 + irregular_b * 0.10 + sz * 0.10)
    
    floor, reason = 0, None
    if n > 30:
        floor, reason = 75, f"Necrosis {n:.1f}% → Forced SEVERE"
    elif n > 20:
        floor, reason = 60, f"Necrosis {n:.1f}% → Forced MODERATE"
    if p > 15 and 70 > floor:
        floor, reason = 70, f"Pus {p:.1f}% → Forced MODERATE-SEVERE"
    if sz >= 80 and r > 20 and 65 > floor:
        floor, reason = 65, f"Large+Red → Forced MODERATE"
        
    override = reason if floor > score else None
    score = float(np.clip(max(score, floor), 0, 100))
    
    if score < NONE_THR:
        level, color = "🟢 NONE / MINIMAL", "#27ae60"
    elif score < MILD_THR:
        level, color = "🟡 MILD INFECTION", "#f39c12"
    elif score < MOD_THR:
        level, color = "🟠 MODERATE INFECTION", "#e67e22"
    else:
        level, color = "🔴 SEVERE INFECTION", "#c0392b"
        
    return {"score": round(score, 2), "level": level, "color": color, "override": override}


# =====================================================================
# 3. WAGNER CLASSIFICATION & CLINICAL LOGIC
# =====================================================================
def compute_wagner_grade(features, mask, image_shape, infection_score):
    if mask.sum() == 0:
        return {
            "grade": 0, "title": "Grade 0 — Pre-ulcer / At Risk",
            "description": "No active ulcer detected. Preventive care recommended.",
            "color": "#2ecc71", "emoji": "🟢"
        }
    
    necrosis  = features["Necrosis"]
    pus       = features["Pus"]
    coverage  = (mask.sum() / (image_shape[0] * image_shape[1])) * 100
    
    if necrosis > 50 and coverage > 15:
        return {
            "grade": 5, "title": "Grade 5 — Extensive Gangrene",
            "description": "Critical: extensive necrosis. Possible major amputation.",
            "color": "#111111", "emoji": "⚫"
        }
    elif necrosis > 30:
        return {
            "grade": 4, "title": "Grade 4 — Localized Gangrene",
            "description": "Severe: localized necrosis. Surgical debridement needed.",
            "color": "#c0392b", "emoji": "🔴"
        }
    elif pus > 15 or infection_score > 65:
        return {
            "grade": 3, "title": "Grade 3 — Deep Ulcer + Abscess",
            "description": "Deep infection with abscess. Aggressive treatment needed.",
            "color": "#d35400", "emoji": "🟠"
        }
    elif coverage > 5 or infection_score > 40:
        return {
            "grade": 2, "title": "Grade 2 — Deep Ulcer",
            "description": "Ulcer extends to deeper tissues. Medical attention needed.",
            "color": "#e67e22", "emoji": "🟠"
        }
    elif coverage > 0:
        return {
            "grade": 1, "title": "Grade 1 — Superficial Ulcer",
            "description": "Superficial ulcer. Standard wound care.",
            "color": "#f1c40f", "emoji": "🟡"
        }
    else:
        return {
            "grade": 0, "title": "Grade 0 — Pre-ulcer",
            "description": "No active ulcer. Maintain preventive care.",
            "color": "#2ecc71", "emoji": "🟢"
        }


def get_clinical_decision(infection_result, wagner):
    grade = wagner["grade"]
    
    if grade >= 4 or "SEVERE" in infection_result["level"]:
        return {
            "critical": True,
            "level": "CRITICAL — EMERGENCY",
            "color": "#c0392b",
            "icon": "🚨",
            "advice": "EMERGENCY: Immediate hospital care required. Risk of amputation/sepsis.",
            "block_sim": False,
        }
    if grade == 3 or "MODERATE" in infection_result["level"]:
        return {
            "critical": False,
            "level": "HIGH PRIORITY",
            "color": "#e67e22",
            "icon": "⚠️",
            "advice": "URGENT: See doctor TODAY. Antibiotics likely needed.",
            "block_sim": False,
        }
    if grade == 2 or "MILD" in infection_result["level"]:
        return {
            "critical": False,
            "level": "MODERATE",
            "color": "#f39c12",
            "icon": "⚠",
            "advice": "See doctor within 48 hours. Monitor closely.",
            "block_sim": False,
        }
    return {
        "critical": False,
        "level": "ROUTINE",
        "color": "#27ae60",
        "icon": "✓",
        "advice": "Continue current wound care. Monitor weekly.",
        "block_sim": False,
    }


def get_progression_alert(change_pct, decision):
    if change_pct <= -50:
        return {"level": "EXCELLENT HEALING", "color": "#27ae60",
                "advice": "Excellent recovery! Continue current treatment.",
                "urgency": "None", "icon": "✅"}
    elif change_pct <= -20:
        return {"level": "GOOD HEALING", "color": "#27ae60",
                "advice": "Good healing progress. Continue care.",
                "urgency": "None", "icon": "✅"}
    elif change_pct <= -5:
        return {"level": "SLOW HEALING", "color": "#f1c40f",
                "advice": "Slow but positive progress. Monitor weekly.",
                "urgency": "Low", "icon": "⏳"}
    elif change_pct <= 5:
        return {"level": "STABLE", "color": "#f39c12",
                "advice": "Condition stable. Monitor closely.",
                "urgency": "Low", "icon": "⚠"}
    elif change_pct <= 20:
        return {"level": "SLOW WORSENING", "color": "#e67e22",
                "advice": "Worsening detected. See doctor within 48h.",
                "urgency": "Moderate", "icon": "⚠️"}
    elif change_pct <= 50:
        return {"level": "ACTIVE WORSENING", "color": "#c0392b",
                "advice": "URGENT: Deteriorating. See doctor TODAY.",
                "urgency": "High", "icon": "🚨"}
    else:
        return {"level": "RAPID DETERIORATION", "color": "#7f1d1d",
                "advice": "HIGH RISK: Seek urgent care within 24-48h.",
                "urgency": "Critical", "icon": "🆘"}


# =====================================================================
# 4. VISUALIZATION & COLAB OVERLAYS
# =====================================================================
def _b64_png(rgb):
    ok, buf = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    return "data:image/png;base64," + base64.b64encode(buf).decode()


def build_overlays(image, mask):
    overlay = image.astype(np.float32) / 255.0
    red = overlay.copy()
    red[mask == 1] = [1, 0, 0]
    blended = ((cv2.addWeighted(overlay, 0.6, red, 0.4, 0)) * 255).astype(np.uint8)
    
    boundary = image.copy()
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(boundary, contours, -1, (255, 255, 0), 3)
    
    mask_rgb = cv2.cvtColor((mask * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
    return _b64_png(blended), _b64_png(boundary), _b64_png(mask_rgb)


def build_healing_overlay(image, original_mask, current_mask):
    """Green for active wound + Pink/Salmon for healed region (Colab match)."""
    base = image.astype(np.float32) / 255.0
    overlay = base.copy()
    healed_region = (original_mask == 1) & (current_mask == 0)
    overlay[healed_region] = [1.0, 0.85, 0.75] # Pink/Light salmon
    if current_mask.sum() > 0:
        overlay[current_mask == 1] = [0.0, 1.0, 0.0] # Green for remaining ulcer
    blended = ((cv2.addWeighted(base, 0.5, overlay, 0.5, 0)) * 255).astype(np.uint8)
    return _b64_png(blended)


def build_worsening_overlay(image, current_mask):
    """Red for expanding wound."""
    base = image.astype(np.float32) / 255.0
    overlay = base.copy()
    if current_mask.sum() > 0:
        overlay[current_mask == 1] = [1.0, 0.0, 0.0] # Red
    blended = ((cv2.addWeighted(base, 0.5, overlay, 0.5, 0)) * 255).astype(np.uint8)
    return _b64_png(blended)


def get_severity(coverage_pct):
    if coverage_pct == 0: return "NO ULCER DETECTED"
    if coverage_pct < 1:  return "MILD"
    if coverage_pct < 5:  return "MODERATE"
    if coverage_pct < 15: return "SEVERE"
    return "CRITICAL"


# =====================================================================
# 5. SINGLE-VISIT CLINICAL SIMULATION TIMELINE
# =====================================================================
def simulate_progression(image):
    mask = segment_ulcer(image)
    feats = analyze_features(image, mask)
    coverage = float(mask.sum() / (image.shape[0] * image.shape[1]) * 100)
    infection = compute_infection(feats, coverage)
    wagner = compute_wagner_grade(feats, mask, image.shape, infection["score"])
    decision = get_clinical_decision(infection, wagner)

    payload = {
        "critical": decision.get("critical", False),
        "clinical_decision": decision,
        "wagner": wagner,
        "infection": infection,
        "coverage_pct": round(coverage, 3),
        "features": {k: round(v, 2) for k, v in feats.items()},
        "healing_timeline": [],
        "worsening_timeline": [],
        "healing_alert": None,
        "worsening_alert": None,
    }

    days = [0, 7, 14, 21, 30]
    total_pixels = image.shape[0] * image.shape[1]
    init_cov = coverage

    # A. Healing simulation (Erosion - Green/Pink)
    for day in days:
        if day == 0:
            sim_mask = mask.copy()
        else:
            k = max(3, int(day * 0.5))
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            sim_mask = cv2.erode(mask, kernel, iterations=1)
        
        sim_cov = float(sim_mask.sum() / total_pixels * 100)
        change_pct = ((sim_cov - init_cov) / init_cov * 100) if init_cov > 0 else 0
        overlay_b64 = build_healing_overlay(image, mask, sim_mask)
        
        payload["healing_timeline"].append({
            "day": day,
            "coverage_pct": round(sim_cov, 3),
            "change_pct": round(change_pct, 2),
            "overlay": overlay_b64
        })

    # B. Worsening simulation (Dilation - Red)
    for day in days:
        if day == 0:
            sim_mask = mask.copy()
        else:
            k = max(3, int(day * 0.7))
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            sim_mask = cv2.dilate(mask, kernel, iterations=1)
            
        sim_cov = float(sim_mask.sum() / total_pixels * 100)
        change_pct = ((sim_cov - init_cov) / init_cov * 100) if init_cov > 0 else 0
        overlay_b64 = build_worsening_overlay(image, sim_mask)
        
        payload["worsening_timeline"].append({
            "day": day,
            "coverage_pct": round(sim_cov, 3),
            "change_pct": round(change_pct, 2),
            "overlay": overlay_b64
        })

    final_heal_change = payload["healing_timeline"][-1]["change_pct"]
    final_worse_change = payload["worsening_timeline"][-1]["change_pct"]
    
    payload["healing_alert"] = get_progression_alert(final_heal_change, decision)
    payload["worsening_alert"] = get_progression_alert(final_worse_change, decision)
    
    return payload


# =====================================================================
# 6. MAIN SYSTEM ANALYSIS HOOK
# =====================================================================
def analyze(image):
    quality_ok, quality_warnings = check_image_quality(image)
    mask = segment_ulcer(image)
    feats = analyze_features(image, mask)
    coverage = float(mask.sum() / (image.shape[0] * image.shape[1]) * 100)
    infection = compute_infection(feats, coverage)
    wagner = compute_wagner_grade(feats, mask, image.shape, infection["score"])
    severity = get_severity(coverage)
    overlay, boundary, mask_png = build_overlays(image, mask)
    
    return {
        "coverage_pct": round(coverage, 3),
        "infection": infection,
        "wagner": wagner,
        "severity_label": severity,
        "features": {k: round(v, 2) for k, v in feats.items()},
        "image_quality": {"ok": quality_ok, "warnings": quality_warnings},
        "images": {"overlay": overlay, "boundary": boundary, "mask": mask_png},
    }



