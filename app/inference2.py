"""
Phase 2 pipeline — 3 ML models:
  1. Ulcer segmentation (UnetPlusPlus, efficientnet-b4)
  2. Tissue segmentation (UnetPlusPlus, 4 classes: bg, granulation, slough, necrosis)
  3. Wagner grade classifier (timm efficientnet_b4, 4 classes: G1-G4)

Two-Visit Progression Engine updated to exactly match Colab logic.
"""
import os, base64
import numpy as np
import cv2

ULCER_MODEL = "ml model/best_model1.pth"
TISSUE_MODEL = "ml model/best_tissue_model.pth"
WAGNER_MODEL = "ml model/best_wagner_model.pth"

ULCER_SIZE   = 512
ULCER_THR    = 0.50
TISSUE_SIZE  = 512
WAGNER_SIZE  = 384
NUM_TISSUE   = 4
NUM_WAGNER   = 4
ENCODER      = "efficientnet-b4"

CLS_GRAN, CLS_SLOUGH, CLS_NECRO = 1, 2, 3
COLORS = np.array([[0,0,0],[255,0,0],[255,255,0],[50,50,50]], dtype=np.uint8)
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

NONE_THR, MILD_THR, MOD_THR = 20, 40, 65
CLASS_NAMES = ["Grade 1", "Grade 2", "Grade 3", "Grade 4"]

_MODELS_READY = False
_ulcer_model = _tissue_model = _wagner_model = None
_device = None


def _ensure_loaded():
    global _MODELS_READY, _ulcer_model, _tissue_model, _wagner_model, _device
    if _MODELS_READY:
        return
    import torch
    import segmentation_models_pytorch as smp
    import timm

    _device = "cuda" if torch.cuda.is_available() else "cpu"

    def _load_seg(path, classes):
        m = smp.UnetPlusPlus(encoder_name=ENCODER, encoder_weights=None,
                             in_channels=3, classes=classes, activation=None,
                             decoder_attention_type="scse").to(_device)
        ckpt = torch.load(path, map_location=_device, weights_only=False)
        m.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
        m.eval()
        return m

    _ulcer_model  = _load_seg(ULCER_MODEL, 1)
    _tissue_model = _load_seg(TISSUE_MODEL, NUM_TISSUE)

    _wagner_model = timm.create_model("efficientnet_b4", pretrained=False,
                                      num_classes=NUM_WAGNER, drop_rate=0.3).to(_device)
    ckpt = torch.load(WAGNER_MODEL, map_location=_device, weights_only=False)
    _wagner_model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
    _wagner_model.eval()

    _MODELS_READY = True


def _preprocess(image, size):
    resized = cv2.resize(image, (size, size))
    x = resized.astype(np.float32) / 255.0
    x = (x - MEAN) / STD
    x = np.transpose(x, (2, 0, 1))[None].astype(np.float32)
    return x


def _segment_ulcer(image):
    import torch
    h, w = image.shape[:2]
    x = _preprocess(image, ULCER_SIZE)
    with torch.no_grad():
        t = torch.from_numpy(x).to(_device)
        p1 = torch.sigmoid(_ulcer_model(t))
        p2 = torch.flip(torch.sigmoid(_ulcer_model(torch.flip(t, [3]))), [3])
        pred = ((p1 + p2) / 2)[0, 0].cpu().numpy()
    mask = (pred > ULCER_THR).astype(np.uint8)
    mask = _postprocess_ulcer(mask)
    return cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)


def _postprocess_ulcer(mask, min_area=100):
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7)))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    out = np.zeros_like(mask)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            out[labels == i] = 1
    return out


def _segment_tissue(image, ulcer_mask, padding=60):
    import torch
    if ulcer_mask.sum() == 0:
        return np.zeros_like(ulcer_mask)
    ys, xs = np.where(ulcer_mask > 0)
    y1, y2 = max(0, ys.min()-padding), min(image.shape[0], ys.max()+padding)
    x1, x2 = max(0, xs.min()-padding), min(image.shape[1], xs.max()+padding)
    crop = image[y1:y2, x1:x2]

    x = _preprocess(crop, TISSUE_SIZE)
    with torch.no_grad():
        t = torch.from_numpy(x).to(_device)
        logits = _tissue_model(t)
        crop_pred = torch.argmax(logits, dim=1)[0].cpu().numpy().astype(np.uint8)

    crop_pred = cv2.resize(crop_pred, (x2-x1, y2-y1), interpolation=cv2.INTER_NEAREST)
    full = np.zeros_like(ulcer_mask)
    full[y1:y2, x1:x2] = crop_pred

    roi_bg = (full == 0) & (ulcer_mask == 1)
    if roi_bg.sum() > 0.3 * ulcer_mask.sum():
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        full[roi_bg & (gray < 70)] = CLS_NECRO
        full[roi_bg & (gray >= 70)] = CLS_GRAN
    return full


def _clean_tissue(tissue_pred, ulcer_mask, image, min_slough=30):
    out = tissue_pred.copy()
    roi = ulcer_mask.astype(bool)
    if roi.sum() == 0:
        return out
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    H, S, V = hsv[:,:,0], hsv[:,:,1], hsv[:,:,2]

    is_slough = (out == CLS_SLOUGH) & roi
    is_deep_red = ((H < 12) | (H > 168)) & (S > 110) & (V > 80)
    out[is_slough & is_deep_red] = CLS_GRAN

    necro_pct  = (out[roi] == CLS_NECRO).sum()  / roi.sum() * 100
    slough_pct = (out[roi] == CLS_SLOUGH).sum() / roi.sum() * 100
    if necro_pct > 40.0 and slough_pct < 12.0:
        out[(out == CLS_SLOUGH) & roi] = CLS_NECRO

    slough_mask = (out == CLS_SLOUGH) & roi
    n, labels, stats, _ = cv2.connectedComponentsWithStats(slough_mask.astype(np.uint8), 8)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < min_slough:
            out[labels == i] = CLS_GRAN
    return out


def _compute_severity(image, ulcer_mask, tissue_pred):
    upix = ulcer_mask.sum()
    if upix == 0:
        gran = slough = necro = redness = swelling = 0.0
        size_score = 10
    else:
        tin = tissue_pred * ulcer_mask
        gran   = float((tin == CLS_GRAN).sum()   / upix * 100)
        slough = float((tin == CLS_SLOUGH).sum() / upix * 100)
        necro  = float((tin == CLS_NECRO).sum()  / upix * 100)

        R = image[:,:,0].astype(float); G = image[:,:,1].astype(float); B = image[:,:,2].astype(float)
        reds = np.clip(R[ulcer_mask==1] - (G[ulcer_mask==1]+B[ulcer_mask==1])/2, 0, 255)
        redness = float(np.mean(reds)/255*100) if len(reds) else 0.0

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25,25))
        surround = cv2.dilate(ulcer_mask, kernel) - ulcer_mask
        swells = np.clip(R[surround==1] - (G[surround==1]+B[surround==1])/2, 0, 255)
        swelling = float(np.mean(swells)/255*100) if len(swells) else 0.0

        size_pct = (upix / (image.shape[0]*image.shape[1])) * 100
        size_score = 10 if size_pct<1 else 30 if size_pct<5 else 60 if size_pct<10 else 100

    tissue_score = min((slough*1.2) + (necro*2.0), 100)
    score = float(np.clip(
        tissue_score*0.45 + redness*0.375 + swelling*0.375 + size_score*0.15 - gran*0.10,
        0, 100
    ))

    # EXACT COLAB OVERRIDES FOR INFECTION
    override = None
    if necro > 30:
        score, override = max(score, 75), "High Necrosis → SEVERE"
    elif necro > 15:
        score, override = max(score, 60), "Mod Necrosis → MODERATE"
    elif slough > 40:
        score, override = max(score, 65), "High Slough → MODERATE"

    if score < NONE_THR:   lbl, col = "NONE INFECTION", "#27ae60"
    elif score < MILD_THR: lbl, col = "MILD INFECTION", "#f1c40f"
    elif score < MOD_THR:  lbl, col = "MODERATE INFECTION", "#e67e22"
    else:                  lbl, col = "SEVERE INFECTION", "#c0392b"

    return {
        "score": round(score, 2), "level": lbl, "color": col, "override": override,
        "features": {
            "Size": size_score,
            "Swelling": round(swelling, 2),
            "Redness": round(redness, 2),
            "Necrosis": round(necro, 2),
            "Slough": round(slough, 2),
            "Granulation": round(gran, 2),
        },
    }


def _predict_wagner(image):
    import torch
    x = _preprocess(image, WAGNER_SIZE)
    with torch.no_grad():
        t = torch.from_numpy(x).to(_device)
        logits = _wagner_model(t)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
    idx = int(np.argmax(probs))
    grade = idx + 1
    conf = float(probs[idx] * 100)
    info = {
        1: {"title": "Grade 1 — Superficial Ulcer",
            "description": "Superficial ulcer. Standard wound care."},
        2: {"title": "Grade 2 — Deep Ulcer",
            "description": "Ulcer extends to deeper tissues. Medical attention needed."},
        3: {"title": "Grade 3 — Deep + Abscess",
            "description": "Deep infection with abscess. Aggressive treatment needed."},
        4: {"title": "Grade 4 — Localized Gangrene",
            "description": "Severe: localized necrosis. Surgical debridement needed."},
    }[grade]
    colors = {1:"#f1c40f", 2:"#e67e22", 3:"#d35400", 4:"#c0392b"}
    return {"grade": grade, "confidence": round(conf, 2),
            "title": info["title"], "description": info["description"],
            "color": colors[grade]}


def _b64_png(rgb):
    ok, buf = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    return "data:image/png;base64," + base64.b64encode(buf).decode()


def _build_overlays(image, ulcer_mask, tissue_pred):
    base = image.astype(np.float32) / 255.0
    overlay = base.copy(); overlay[ulcer_mask == 1] = [1, 0, 0]
    blended = ((cv2.addWeighted(base, 0.6, overlay, 0.4, 0)) * 255).astype(np.uint8)

    boundary = image.copy()
    contours, _ = cv2.findContours(ulcer_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(boundary, contours, -1, (255, 255, 0), 3)

    t_col = COLORS[np.clip(tissue_pred * ulcer_mask, 0, 3)]
    tblend = image.astype(np.float32) / 255
    m3 = ((tissue_pred * ulcer_mask) > 0)[..., None].astype(np.float32)
    tblend = tblend * (1 - 0.5*m3) + (t_col.astype(np.float32) / 255) * (0.5*m3)
    tblend = (np.clip(tblend, 0, 1) * 255).astype(np.uint8)

    mask_rgb = cv2.cvtColor((ulcer_mask * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
    return _b64_png(blended), _b64_png(boundary), _b64_png(mask_rgb), _b64_png(tblend)


def get_severity_label(coverage_pct):
    if coverage_pct == 0: return "NO ULCER DETECTED"
    if coverage_pct < 1:  return "MILD"
    if coverage_pct < 5:  return "MODERATE"
    if coverage_pct < 15: return "SEVERE"
    return "CRITICAL"


def analyze(image):
    _ensure_loaded()
    ulcer_mask = _segment_ulcer(image)
    tissue_pred = _segment_tissue(image, ulcer_mask)
    tissue_pred = _clean_tissue(tissue_pred, ulcer_mask, image)

    infection = _compute_severity(image, ulcer_mask, tissue_pred)
    wagner = _predict_wagner(image)
    coverage = float(ulcer_mask.sum() / (image.shape[0] * image.shape[1]) * 100)
    severity = get_severity_label(coverage)

    overlay, boundary, mask_png, tissue_png = _build_overlays(image, ulcer_mask, tissue_pred)

    return {
        "coverage_pct": round(coverage, 3),
        "infection": infection,
        "wagner": wagner,
        "severity_label": severity,
        "features": infection["features"],
        "image_quality": {"ok": True, "warnings": []},
        "images": {
            "overlay": overlay,
            "boundary": boundary,
            "mask": mask_png,
            "tissue": tissue_png,
        },
    }


# =====================================================================
# EXACT COLAB PROGRESSION ENGINE
# =====================================================================
def compare_visits(old_data, new_data):
    deltas = {}

    old_g = old_data["wagner"]["grade"]
    new_g = new_data["wagner"]["grade"]
    dw = new_g - old_g
    deltas["wagner"] = {
        "old": old_g,
        "new": new_g,
        "delta": dw,
        "trend": "STABLE" if dw == 0 else ("WORSENED" if dw > 0 else "IMPROVED"),
        "icon":  "[=]" if dw == 0 else ("[^]" if dw > 0 else "[v]"),
    }

    di = new_data["infection"]["score"] - old_data["infection"]["score"]
    deltas["infection_score"] = {
        "old": round(old_data["infection"]["score"], 2),
        "new": round(new_data["infection"]["score"], 2),
        "delta": round(di, 2),
        "trend": "STABLE" if abs(di) < 3 else ("WORSENED" if di > 0 else "IMPROVED"),
        "icon":  "[=]" if abs(di) < 3 else ("[^]" if di > 0 else "[v]"),
    }

    old_cov = old_data["coverage_pct"]
    new_cov = new_data["coverage_pct"]
    dc = new_cov - old_cov
    pct_change = ((new_cov - old_cov) / old_cov * 100) if old_cov > 0 else 0
    deltas["coverage_pct"] = {
        "old": round(old_cov, 2),
        "new": round(new_cov, 2),
        "delta": round(dc, 2),
        "pct_change": round(pct_change, 2),
        "trend": "STABLE" if abs(dc) < 0.5 else ("WORSENED" if dc > 0 else "IMPROVED"),
        "icon":  "[=]" if abs(dc) < 0.5 else ("[^]" if dc > 0 else "[v]"),
    }

    deltas["features"] = {}
    old_feats = old_data["infection"]["features"]
    new_feats = new_data["infection"]["features"]
    
    for fname in old_feats:
        ov = old_feats[fname]
        nv = new_feats.get(fname, 0.0)
        d = nv - ov
        
        # Granulation: higher = better
        if fname == "Granulation":
            trend = "STABLE" if abs(d) < 2 else ("IMPROVED" if d > 0 else "WORSENED")
            icon  = "[=]"     if abs(d) < 2 else ("[v]"       if d > 0 else "[^]")
        else:
            trend = "STABLE" if abs(d) < 2 else ("WORSENED" if d > 0 else "IMPROVED")
            icon  = "[=]"     if abs(d) < 2 else ("[^]"       if d > 0 else "[v]")
            
        deltas["features"][fname] = {
            "old": round(ov, 2),
            "new": round(nv, 2),
            "delta": round(d, 2),
            "trend": trend,
            "icon": icon
        }

    # EXACT COLAB CLINICAL VERDICT ENGINE
    worsened = sum(1 for f in deltas["features"].values() if f["trend"] == "WORSENED")
    improved = sum(1 for f in deltas["features"].values() if f["trend"] == "IMPROVED")

    critical_worsening = (deltas["wagner"]["delta"] > 0 or deltas["infection_score"]["delta"] > 15)
    critical_improving = (deltas["wagner"]["delta"] < 0 or deltas["infection_score"]["delta"] < -15)

    if critical_worsening or worsened >= 4:
        verdict = {
            "status": "DETERIORATING",
            "color": "#c0392b",
            "action": "URGENT: See doctor within 24-48 hours. Wound is worsening.",
            "detail": f"{worsened}/6 features worsened"
        }
    elif critical_improving or improved >= 4:
        verdict = {
            "status": "HEALING WELL",
            "color": "#27ae60",
            "action": "Continue current treatment. Good progress!",
            "detail": f"{improved}/6 features improved"
        }
    elif worsened > improved:
        verdict = {
            "status": "MILD WORSENING",
            "color": "#e67e22",
            "action": "Monitor closely. Book clinic visit within 1 week.",
            "detail": f"{worsened} worsened vs {improved} improved"
        }
    elif improved > worsened:
        verdict = {
            "status": "SLOWLY HEALING",
            "color": "#f1c40f",
            "action": "Positive trend. Continue care and monitor.",
            "detail": f"{improved} improved vs {worsened} worsened"
        }
    else:
        verdict = {
            "status": "STABLE",
            "color": "#3498db",
            "action": "Condition stable. Continue current care.",
            "detail": "No significant change"
        }

    return {
        "deltas": deltas,
        "verdict": verdict
    }

