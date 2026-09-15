import os
from fastapi import APIRouter, Header, HTTPException, Depends
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from ..models import get_db, Doctor, User

router = APIRouter(prefix="/admin", tags=["admin"])
ADMIN_KEY = os.getenv("ADMIN_KEY", "change-me-admin-key")


def _check_key(x_admin_key: str = Header(...)):
    if x_admin_key != ADMIN_KEY:
        raise HTTPException(403, "Invalid admin key")


def _fmt(d: Doctor) -> dict:
    return {"doctor_id": d.id, "full_name": d.user.full_name, "email": d.user.email,
            "license_number": d.license_number, "medical_council": d.medical_council,
            "specialty": d.specialty, "has_license_doc": bool(d.license_doc_path)}


@router.get("/pending-doctors")
def pending_doctors(db: Session = Depends(get_db), _=Depends(_check_key)):
    doctors = db.query(Doctor).filter(Doctor.is_verified == False).join(User).all()
    return [_fmt(d) for d in doctors]


@router.get("/verified-doctors")
def verified_doctors(db: Session = Depends(get_db), _=Depends(_check_key)):
    doctors = db.query(Doctor).filter(Doctor.is_verified == True).join(User).all()
    return [_fmt(d) for d in doctors]


@router.get("/doctor-license/{doctor_id}")
def get_license_doc(doctor_id: int, x_admin_key: str = Header(...), db: Session = Depends(get_db)):
    if x_admin_key != ADMIN_KEY:
        raise HTTPException(403, "Invalid admin key")
    doctor = db.query(Doctor).filter(Doctor.id == doctor_id).first()
    if not doctor or not doctor.license_doc_path or not os.path.exists(doctor.license_doc_path):
        raise HTTPException(404, "No document uploaded")
    return FileResponse(doctor.license_doc_path)


@router.post("/approve-doctor/{doctor_id}")
def approve_doctor(doctor_id: int, db: Session = Depends(get_db), _=Depends(_check_key)):
    doctor = db.query(Doctor).filter(Doctor.id == doctor_id).first()
    if not doctor:
        raise HTTPException(404, "Doctor not found")
    doctor.is_verified = True
    db.commit()
    return {"status": "approved", "doctor_id": doctor.id}


@router.post("/reject-doctor/{doctor_id}")
def reject_doctor(doctor_id: int, db: Session = Depends(get_db), _=Depends(_check_key)):
    doctor = db.query(Doctor).filter(Doctor.id == doctor_id).first()
    if not doctor:
        raise HTTPException(404, "Doctor not found")
    doctor.is_verified = False
    db.commit()
    return {"status": "rejected", "doctor_id": doctor.id}
