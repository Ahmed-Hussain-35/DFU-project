import os, secrets
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel, EmailStr
from sqlalchemy.orm import Session

from ..models import get_db, User, Patient, Doctor
from ..auth import hash_password, verify_password, create_access_token, require_role

router = APIRouter(prefix="/auth", tags=["auth"])
LICENSE_DIR = os.getenv("LICENSE_DIR", "./licenses")


class RegisterRequest(BaseModel):
    email: EmailStr
    password: str
    full_name: str
    role: str  # "patient" | "doctor"
    doctor_id: int | None = None        # required when role == "patient"
    license_number: str | None = None   # self-reported, when role == "doctor"
    medical_council: str | None = None  # e.g. "Telangana State Medical Council"


class ChangeDoctorRequest(BaseModel):
    doctor_id: int


@router.get("/doctors")
def list_doctors(db: Session = Depends(get_db)):
    """Public list for the patient signup dropdown and change-doctor picker.
    Only ADMIN-VERIFIED doctors show up here — a freshly self-registered
    doctor account is invisible to patients until approved."""
    doctors = db.query(Doctor).join(User).filter(Doctor.is_verified == True).all()
    return [{"doctor_id": d.id, "full_name": d.user.full_name, "specialty": d.specialty} for d in doctors]


@router.post("/register")
def register(req: RegisterRequest, db: Session = Depends(get_db)):
    if req.role not in ("patient", "doctor"):
        raise HTTPException(400, "role must be 'patient' or 'doctor'")
    if db.query(User).filter(User.email == req.email).first():
        raise HTTPException(400, "Email already registered")
    if req.role == "patient" and req.doctor_id is None:
        raise HTTPException(400, "Please choose a doctor")

    user = User(email=req.email, hashed_password=hash_password(req.password),
                role=req.role, full_name=req.full_name)
    db.add(user)
    db.commit()
    db.refresh(user)

    if req.role == "patient":
        if not db.query(Doctor).filter(Doctor.id == req.doctor_id).first():
            raise HTTPException(400, "Selected doctor not found")
        db.add(Patient(user_id=user.id, assigned_doctor_id=req.doctor_id))
    else:
        db.add(Doctor(user_id=user.id, license_number=req.license_number,
                      medical_council=req.medical_council, is_verified=False))
    db.commit()

    token = create_access_token({"sub": str(user.id), "role": user.role})
    return {"access_token": token, "token_type": "bearer", "role": user.role, "full_name": user.full_name}


@router.post("/login")
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == form_data.username).first()
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(401, "Incorrect email or password")
    token = create_access_token({"sub": str(user.id), "role": user.role})
    resp = {"access_token": token, "token_type": "bearer", "role": user.role, "full_name": user.full_name}
    if user.role == "doctor":
        doctor = db.query(Doctor).filter(Doctor.user_id == user.id).first()
        resp["is_verified"] = doctor.is_verified if doctor else False
    return resp


@router.post("/doctor/upload-license")
async def upload_license(file: UploadFile = File(...), user: User = Depends(require_role("doctor")),
                         db: Session = Depends(get_db)):
    """Doctor uploads their license certificate (image or PDF) for admin review.
    This is EVIDENCE for the admin to look at — uploading a file proves nothing
    by itself, it just gives the admin something concrete to check instead of
    trusting a typed-in number alone."""
    doctor = db.query(Doctor).filter(Doctor.user_id == user.id).first()
    if not doctor:
        raise HTTPException(404, "Doctor profile not found")
    os.makedirs(LICENSE_DIR, exist_ok=True)
    ext = os.path.splitext(file.filename or "")[1] or ".bin"
    fname = f"{secrets.token_hex(8)}{ext}"
    path = os.path.join(LICENSE_DIR, fname)
    raw = await file.read()
    with open(path, "wb") as f:
        f.write(raw)
    doctor.license_doc_path = path
    db.commit()
    return {"status": "uploaded"}


@router.post("/patient/change-doctor")
def change_doctor(req: ChangeDoctorRequest, user: User = Depends(require_role("patient")),
                  db: Session = Depends(get_db)):
    patient = db.query(Patient).filter(Patient.user_id == user.id).first()
    if not patient:
        raise HTTPException(404, "Patient profile not found")
    doctor = db.query(Doctor).filter(Doctor.id == req.doctor_id, Doctor.is_verified == True).first()
    if not doctor:
        raise HTTPException(400, "Selected doctor not found or not verified")
    patient.assigned_doctor_id = req.doctor_id
    db.commit()
    return {"status": "updated", "doctor_id": doctor.id, "doctor_name": doctor.user.full_name}


@router.get("/doctor/status")
def doctor_status(user: User = Depends(require_role("doctor")), db: Session = Depends(get_db)):
    doctor = db.query(Doctor).filter(Doctor.user_id == user.id).first()
    if not doctor:
        raise HTTPException(404, "Doctor profile not found")
    return {"is_verified": doctor.is_verified}
