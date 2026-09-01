from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel, EmailStr
from sqlalchemy.orm import Session

from ..models import get_db, User, Patient, Doctor
from ..auth import hash_password, verify_password, create_access_token

router = APIRouter(prefix="/auth", tags=["auth"])


class RegisterRequest(BaseModel):
    email: EmailStr
    password: str
    full_name: str
    role: str  # "patient" | "doctor"
    doctor_id: int | None = None  # required when role == "patient"


@router.get("/doctors")
def list_doctors(db: Session = Depends(get_db)):
    """Public list for the patient signup dropdown — no auth needed."""
    doctors = db.query(Doctor).join(User).all()
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
        db.add(Doctor(user_id=user.id))
    db.commit()

    token = create_access_token({"sub": str(user.id), "role": user.role})
    return {"access_token": token, "token_type": "bearer", "role": user.role, "full_name": user.full_name}


@router.post("/login")
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == form_data.username).first()
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(401, "Incorrect email or password")
    token = create_access_token({"sub": str(user.id), "role": user.role})
    return {"access_token": token, "token_type": "bearer", "role": user.role, "full_name": user.full_name}
