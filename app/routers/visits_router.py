import os, io, json, secrets
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
from sqlalchemy.orm import Session
from PIL import Image
import numpy as np

from ..models import get_db, User, Patient, Doctor, Visit, VisitImage
from ..auth import get_current_user, require_role
from .. import inference

router = APIRouter(prefix="/visits", tags=["visits"])
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "./uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)


def _generate_public_id(db: Session) -> str:
    """6-digit numeric visit code, checked against the DB for uniqueness."""
    for _ in range(5):
        candidate = str(secrets.randbelow(900000) + 100000)  # 100000-999999
        if not db.query(Visit).filter(Visit.public_id == candidate).first():
            return candidate
    return str(secrets.randbelow(9000000) + 1000000)  # extremely unlikely fallback, 7 digits


@router.post("/upload")
async def upload_visit(file: UploadFile = File(...),
                       notes: str = Form(""),
                       user: User = Depends(require_role("patient")),
                       db: Session = Depends(get_db)):
    patient = db.query(Patient).filter(Patient.user_id == user.id).first()
    if not patient:
        raise HTTPException(404, "Patient profile not found")

    raw = await file.read()
    try:
        image = np.array(Image.open(io.BytesIO(raw)).convert("RGB"))
    except Exception:
        raise HTTPException(400, "Could not read image")

    fname = f"{secrets.token_hex(8)}.jpg"
    path = os.path.join(UPLOAD_DIR, fname)
    Image.fromarray(image).save(path)

    result = inference.analyze(image)
    public_id = _generate_public_id(db)

    visit = Visit(
        public_id=public_id,
        patient_id=patient.id,
        image_path=path,
        coverage_pct=result["coverage_pct"],
        infection_score=result["infection"]["score"],
        infection_level=result["infection"]["level"],
        infection_color=result["infection"]["color"],
        infection_override=result["infection"].get("override"),
        severity_label=result["severity_label"],
        features_json=json.dumps(result["features"]),
        image_quality_ok=result["image_quality"]["ok"],
        image_quality_warnings=json.dumps(result["image_quality"]["warnings"]),
        patient_notes=notes.strip() or None,
        review_status="pending",
    )
    db.add(visit)
    db.commit()
    db.refresh(visit)

    imgs = result["images"]
    db.add(VisitImage(visit_id=visit.id, overlay_b64=imgs["overlay"],
                      boundary_b64=imgs["boundary"], mask_b64=imgs["mask"]))
    db.commit()

    return {"visit_id": visit.public_id, "created_at": visit.created_at,
            "patient_notes": visit.patient_notes, **result}


@router.get("/history")
def visit_history(user: User = Depends(require_role("patient")), db: Session = Depends(get_db)):
    patient = db.query(Patient).filter(Patient.user_id == user.id).first()
    if not patient:
        raise HTTPException(404, "Patient profile not found")
    visits = db.query(Visit).filter(Visit.patient_id == patient.id).order_by(Visit.created_at).all()
    return [_visit_summary(v) for v in visits]


@router.get("/doctor/my-patients")
def my_patients(user: User = Depends(require_role("doctor")), db: Session = Depends(get_db)):
    """All visits from patients assigned to the logged-in doctor, newest first."""
    doctor = db.query(Doctor).filter(Doctor.user_id == user.id).first()
    if not doctor:
        raise HTTPException(404, "Doctor profile not found")
    patients = db.query(Patient).filter(Patient.assigned_doctor_id == doctor.id).all()
    patient_ids = [p.id for p in patients]
    visits = (db.query(Visit).filter(Visit.patient_id.in_(patient_ids))
              .order_by(Visit.created_at.desc()).all())
    out = []
    for v in visits:
        row = _visit_summary(v)
        row["patient_name"] = v.patient.user.full_name
        out.append(row)
    return out


@router.get("/{visit_id}")
def get_visit(visit_id: str, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Full detail — reads everything from the DB, no model recompute."""
    visit = db.query(Visit).filter(Visit.public_id == visit_id).first()
    if not visit:
        raise HTTPException(404, "Visit not found")
    return _visit_detail(visit)


class ReviewRequest(BaseModel):
    doctor_notes: str
    mask_confirmed: bool
    review_status: str = "reviewed"


@router.post("/{visit_id}/review")
def review_visit(visit_id: str, req: ReviewRequest,
                 user: User = Depends(require_role("doctor")),
                 db: Session = Depends(get_db)):
    visit = db.query(Visit).filter(Visit.public_id == visit_id).first()
    if not visit:
        raise HTTPException(404, "Visit not found")
    visit.doctor_notes = req.doctor_notes
    visit.mask_confirmed = req.mask_confirmed
    visit.review_status = req.review_status
    visit.reviewed_by_id = user.id
    visit.reviewed_at = datetime.utcnow()
    db.commit()
    db.refresh(visit)
    return _visit_detail(visit)


def _visit_detail(v: Visit) -> dict:
    features = json.loads(v.features_json or "{}")
    images = {"overlay": v.images.overlay_b64, "boundary": v.images.boundary_b64,
              "mask": v.images.mask_b64} if v.images else {}
    return {
        "visit_id": v.public_id,
        "created_at": v.created_at,
        "review_status": v.review_status,
        "doctor_notes": v.doctor_notes,
        "patient_notes": v.patient_notes,
        "mask_confirmed": v.mask_confirmed,
        "coverage_pct": v.coverage_pct,
        "severity_label": v.severity_label,
        "features": features,
        "infection": {"score": v.infection_score, "level": v.infection_level,
                      "color": v.infection_color, "override": v.infection_override},
        "image_quality": {"ok": v.image_quality_ok, "warnings": json.loads(v.image_quality_warnings or "[]")},
        "images": images,
    }


def _visit_summary(v: Visit) -> dict:
    return {
        "visit_id": v.public_id,
        "created_at": v.created_at,
        "coverage_pct": v.coverage_pct,
        "infection_score": v.infection_score,
        "infection_level": v.infection_level,
        "severity_label": v.severity_label,
        "image_quality_ok": v.image_quality_ok,
        "image_quality_warnings": json.loads(v.image_quality_warnings or "[]"),
        "review_status": v.review_status,
        "doctor_notes": v.doctor_notes,
        "patient_notes": v.patient_notes,
        "mask_confirmed": v.mask_confirmed,
    }
