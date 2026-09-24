"""
Visits router — handles uploads (Phase 1 / Phase 2 / both), history,
individual visit fetch, doctor review, progression simulation, and comparison.
"""
import os, json, uuid
from datetime import datetime
from typing import Optional
import numpy as np
from PIL import Image
import io

from fastapi import APIRouter, Depends, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ..models import get_db, Visit, VisitImage, Patient, User
try:
    from ..auth import get_current_user, require_role
except ImportError:
    from .auth_router import get_current_user, require_role

from .. import inference as phase1
from .. import inference2 as phase2

router = APIRouter(prefix="/visits", tags=["visits"])
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)


def _load_image(raw_bytes) -> np.ndarray:
    return np.array(Image.open(io.BytesIO(raw_bytes)).convert("RGB"))

def _save_upload(raw_bytes, ext=".jpg") -> str:
    name = f"{uuid.uuid4().hex}{ext}"
    path = os.path.join(UPLOAD_DIR, name)
    with open(path, "wb") as f:
        f.write(raw_bytes)
    return path

def _visit_to_dict(v: Visit, include_images=True):
    d = {
        "visit_id": v.id, "public_id": v.public_id, "created_at": v.created_at.isoformat(),
        "phases_run": v.phases_run or "phase1", "patient_notes": v.patient_notes,
        "doctor_notes": v.doctor_notes, "review_status": v.review_status,
        "mask_confirmed": v.mask_confirmed,
        "image_quality": {
            "ok": v.image_quality_ok,
            "warnings": json.loads(v.image_quality_warnings) if v.image_quality_warnings else [],
        },
    }
    
    if v.coverage_pct is not None:
        p1_feats = json.loads(v.features_json) if v.features_json else {}
        # Recompute Phase 1 Wagner grade on-the-fly from stored features
        p1_wagner = None
        try:
            # Build fake mask flag by coverage sign: real mask not stored, use coverage-based logic
            necrosis = p1_feats.get("Necrosis", 0)
            pus = p1_feats.get("Pus", 0)
            coverage = v.coverage_pct or 0
            infection_score = v.infection_score or 0
            
            if coverage == 0:
                p1_wagner = {"grade": 0, "title": "Grade 0 — Pre-ulcer / At Risk",
                             "description": "No active ulcer detected. Preventive care recommended.",
                             "color": "#2ecc71", "confidence": 100.0}
            elif necrosis > 50 and coverage > 15:
                p1_wagner = {"grade": 5, "title": "Grade 5 — Extensive Gangrene",
                             "description": "Critical: extensive necrosis. Possible major amputation.",
                             "color": "#111111", "confidence": 100.0}
            elif necrosis > 30:
                p1_wagner = {"grade": 4, "title": "Grade 4 — Localized Gangrene",
                             "description": "Severe: localized necrosis. Surgical debridement needed.",
                             "color": "#c0392b", "confidence": 100.0}
            elif pus > 15 or infection_score > 65:
                p1_wagner = {"grade": 3, "title": "Grade 3 — Deep Ulcer + Abscess",
                             "description": "Deep infection with abscess. Aggressive treatment needed.",
                             "color": "#d35400", "confidence": 100.0}
            elif coverage > 5 or infection_score > 40:
                p1_wagner = {"grade": 2, "title": "Grade 2 — Deep Ulcer",
                             "description": "Ulcer extends to deeper tissues. Medical attention needed.",
                             "color": "#e67e22", "confidence": 100.0}
            elif coverage > 0:
                p1_wagner = {"grade": 1, "title": "Grade 1 — Superficial Ulcer",
                             "description": "Superficial ulcer. Standard wound care.",
                             "color": "#f1c40f", "confidence": 100.0}
            else:
                p1_wagner = {"grade": 0, "title": "Grade 0 — Pre-ulcer",
                             "description": "No active ulcer. Maintain preventive care.",
                             "color": "#2ecc71", "confidence": 100.0}
        except Exception:
            p1_wagner = None

        d["phase1"] = {
            "coverage_pct": v.coverage_pct, "severity_label": v.severity_label,
            "infection": {"score": v.infection_score, "level": v.infection_level,
                          "color": v.infection_color, "override": v.infection_override,
                          "features": p1_feats},
            "features": p1_feats,
            "wagner": p1_wagner,
        }
        if include_images and v.images:
            d["phase1"]["images"] = {"overlay": v.images.overlay_b64, "boundary": v.images.boundary_b64, "mask": v.images.mask_b64}
            
    if getattr(v, "p2_coverage_pct", None) is not None:
        p2_feats = json.loads(v.p2_features_json) if v.p2_features_json else {}
        d["phase2"] = {
            "coverage_pct": v.p2_coverage_pct, "severity_label": v.p2_severity_label,
            "infection": {"score": v.p2_infection_score, "level": v.p2_infection_level,
                          "color": v.p2_infection_color, "override": v.p2_infection_override,
                          "features": p2_feats},
            "features": p2_feats,
            "wagner": {"grade": v.p2_wagner_grade, "confidence": v.p2_wagner_confidence,
                       "title": v.p2_wagner_title, "description": v.p2_wagner_description, "color": v.p2_wagner_color},
        }
        if include_images and v.images:
            d["phase2"]["images"] = {"overlay": v.images.p2_overlay_b64, "boundary": v.images.p2_boundary_b64,
                                     "mask": v.images.p2_mask_b64, "tissue": v.images.p2_tissue_b64}
    return d


# =====================================================================
# UPLOAD
# =====================================================================
@router.post("/upload")
async def upload_visit(file: UploadFile = File(...), notes: str = Form(""), phases: str = Form("phase1"),
                       user: User = Depends(require_role("patient")), db: Session = Depends(get_db)):
    if not user.patient_profile: raise HTTPException(400, "Patient profile missing")
    raw = await file.read()
    try: image = _load_image(raw)
    except Exception: raise HTTPException(400, "Could not read image")

    wanted = {p.strip() for p in phases.split(",") if p.strip() in ("phase1", "phase2")} or {"phase1"}
    p1_res = p2_res = None
    errors = []

    if "phase1" in wanted:
        try: p1_res = phase1.analyze(image)
        except Exception as e: errors.append(f"Phase 1 failed: {e}")
    if "phase2" in wanted:
        try: p2_res = phase2.analyze(image)
        except Exception as e: errors.append(f"Phase 2 failed: {e}")

    if not p1_res and not p2_res: raise HTTPException(500, "; ".join(errors) or "Analysis failed")

    visit = Visit(public_id=uuid.uuid4().hex[:8], patient_id=user.patient_profile.id,
                  image_path=_save_upload(raw), patient_notes=notes, phases_run=",".join(sorted(wanted)))

    if p1_res:
        visit.coverage_pct = p1_res["coverage_pct"]; visit.severity_label = p1_res["severity_label"]
        visit.infection_score = p1_res["infection"]["score"]; visit.infection_level = p1_res["infection"]["level"]
        visit.infection_color = p1_res["infection"]["color"]; visit.infection_override = p1_res["infection"].get("override")
        visit.features_json = json.dumps(p1_res["features"]); visit.image_quality_ok = p1_res["image_quality"]["ok"]
        visit.image_quality_warnings = json.dumps(p1_res["image_quality"]["warnings"])
    if p2_res:
        visit.p2_coverage_pct = p2_res["coverage_pct"]; visit.p2_severity_label = p2_res["severity_label"]
        visit.p2_infection_score = p2_res["infection"]["score"]; visit.p2_infection_level = p2_res["infection"]["level"]
        visit.p2_infection_color = p2_res["infection"]["color"]; visit.p2_infection_override = p2_res["infection"].get("override")
        visit.p2_features_json = json.dumps(p2_res["features"])
        w = p2_res["wagner"]
        visit.p2_wagner_grade = w["grade"]; visit.p2_wagner_confidence = w["confidence"]; visit.p2_wagner_title = w["title"]
        visit.p2_wagner_description = w["description"]; visit.p2_wagner_color = w["color"]
        if visit.image_quality_ok is None: visit.image_quality_ok = True; visit.image_quality_warnings = json.dumps([])

    db.add(visit); db.commit(); db.refresh(visit)
    imgs = VisitImage(visit_id=visit.id)
    if p1_res:
        imgs.overlay_b64 = p1_res["images"]["overlay"]; imgs.boundary_b64 = p1_res["images"]["boundary"]; imgs.mask_b64 = p1_res["images"]["mask"]
    if p2_res:
        imgs.p2_overlay_b64 = p2_res["images"]["overlay"]; imgs.p2_boundary_b64 = p2_res["images"]["boundary"]
        imgs.p2_mask_b64 = p2_res["images"]["mask"]; imgs.p2_tissue_b64 = p2_res["images"]["tissue"]

    db.add(imgs); db.commit(); db.refresh(visit)
    out = _visit_to_dict(visit)
    if errors: out["partial_errors"] = errors
    return out


# =====================================================================
# HISTORY & FETCH
# =====================================================================
@router.get("/history")
def my_history(user: User = Depends(require_role("patient")), db: Session = Depends(get_db)):
    if not user.patient_profile: return []
    rows = db.query(Visit).filter(Visit.patient_id == user.patient_profile.id).order_by(Visit.created_at.desc()).all()
    return [{"visit_id": v.id, "created_at": v.created_at.isoformat(), "phases_run": v.phases_run or "phase1",
             "coverage_pct": v.coverage_pct if v.coverage_pct is not None else v.p2_coverage_pct,
             "severity_label": v.severity_label or v.p2_severity_label, "review_status": v.review_status} for v in rows]

@router.get("/doctor/my-patients")
def doctor_patient_visits(user: User = Depends(require_role("doctor")), db: Session = Depends(get_db)):
    if not user.doctor_profile: return []
    rows = db.query(Visit).join(Patient, Visit.patient_id == Patient.id).filter(Patient.assigned_doctor_id == user.doctor_profile.id).order_by(Visit.created_at.desc()).all()
    return [{"visit_id": v.id, "created_at": v.created_at.isoformat(), "phases_run": v.phases_run or "phase1",
             "coverage_pct": v.coverage_pct if v.coverage_pct is not None else v.p2_coverage_pct,
             "severity_label": v.severity_label or v.p2_severity_label, "review_status": v.review_status,
             "patient_name": v.patient.user.full_name, "patient_id": v.patient_id} for v in rows]

@router.get("/{visit_id}")
def get_visit(visit_id: int, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    v = db.query(Visit).filter(Visit.id == visit_id).first()
    if not v: raise HTTPException(404, "Visit not found")
    out = _visit_to_dict(v)
    out["patient_id"] = v.patient_id; out["patient_user_id"] = v.patient.user.id
    return out


# =====================================================================
# DOCTOR REVIEW
# =====================================================================
class ReviewIn(BaseModel):
    doctor_notes: str = ""; mask_confirmed: Optional[bool] = None; review_status: str = "reviewed"

@router.post("/{visit_id}/review")
def review_visit(visit_id: int, body: ReviewIn, user: User = Depends(require_role("doctor")), db: Session = Depends(get_db)):
    v = db.query(Visit).filter(Visit.id == visit_id).first()
    v.doctor_notes = body.doctor_notes; v.mask_confirmed = body.mask_confirmed; v.review_status = body.review_status
    v.reviewed_by_id = user.id; v.reviewed_at = datetime.utcnow(); db.commit(); db.refresh(v)
    return get_visit(visit_id, user, db)


# =====================================================================
# PHASE 1 SIMULATION
# =====================================================================
@router.post("/{visit_id}/simulate")
def simulate_progression(visit_id: int, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    v = db.query(Visit).filter(Visit.id == visit_id).first()
    if not v:
        raise HTTPException(404, "Visit not found")
    if not os.path.exists(v.image_path):
        raise HTTPException(404, "Image file missing on server")
    try:
        image = np.array(Image.open(v.image_path).convert("RGB"))
        return phase1.simulate_progression(image)
    except Exception as e:
        raise HTTPException(500, f"Simulation failed: {e}")


# =====================================================================
# TWO-VISIT COMPARISON
# =====================================================================
class CompareIn(BaseModel):
    old_visit_id: int; new_visit_id: int; phase: str = "phase1"

@router.post("/compare")
def compare_visits(body: CompareIn, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    old = db.query(Visit).filter(Visit.id == body.old_visit_id).first()
    new = db.query(Visit).filter(Visit.id == body.new_visit_id).first()
    if not old or not new:
        raise HTTPException(404, "One or both visits not found")
    if old.patient_id != new.patient_id:
        raise HTTPException(400, "Visits must belong to the same patient")

    old_d = _visit_to_dict(old)
    new_d = _visit_to_dict(new)

    if body.phase not in old_d or body.phase not in new_d:
        raise HTTPException(400, f"Missing {body.phase} data in one of the visits")

    if body.phase == "phase2":
        result = phase2.compare_visits(old_d["phase2"], new_d["phase2"])
        return {
            "phase": "phase2",
            "old_visit": {"id": old.id, "date": old.created_at.isoformat(), "data": old_d["phase2"]},
            "new_visit": {"id": new.id, "date": new.created_at.isoformat(), "data": new_d["phase2"]},
            "deltas": result["deltas"],
            "verdict": result["verdict"],
        }

    op = old_d["phase1"]
    np_res = new_d["phase1"]
    deltas = {
        "coverage_pct": {"old": op["coverage_pct"], "new": np_res["coverage_pct"],
                         "delta": round(np_res["coverage_pct"] - op["coverage_pct"], 3)},
        "infection_score": {"old": op["infection"]["score"], "new": np_res["infection"]["score"],
                            "delta": round(np_res["infection"]["score"] - op["infection"]["score"], 2)},
        "features": {},
    }
    for k in set(op["features"]) | set(np_res["features"]):
        ov = float(op["features"].get(k, 0) or 0)
        nv = float(np_res["features"].get(k, 0) or 0)
        deltas["features"][k] = {"old": ov, "new": nv, "delta": round(nv - ov, 2)}

    status = "DETERIORATING" if deltas["infection_score"]["delta"] > 15 else "STABLE"
    return {
        "phase": "phase1",
        "old_visit": {"id": old.id, "date": old.created_at.isoformat(), "data": op},
        "new_visit": {"id": new.id, "date": new.created_at.isoformat(), "data": np_res},
        "deltas": deltas,
        "verdict": {
            "status": status,
            "color": "#c0392b" if status == "DETERIORATING" else "#3498db",
            "action": "Urgent review needed." if status == "DETERIORATING" else "Continue current care.",
            "detail": "Phase 1 rule-based comparison.",
        },
    }