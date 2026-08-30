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


@router.post("/register")
def register(req: RegisterRequest, db: Session = Depends(get_db)):
    if req.role not in ("patient", "doctor"):
        raise HTTPException(400, "role must be 'patient' or 'doctor'")
    if db.query(User).filter(User.email == req.email).first():
        raise HTTPException(400, "Email already registered")

    user = User(email=req.email, hashed_password=hash_password(req.password),
                role=req.role, full_name=req.full_name)
    db.add(user)
    db.commit()
    db.refresh(user)

    if req.role == "patient":
        db.add(Patient(user_id=user.id))
    else:
        db.add(Doctor(user_id=user.id))
    db.commit()

    token = create_access_token({"sub": str(user.id), "role": user.role})
    return {"access_token": token, "token_type": "bearer", "role": user.role}


@router.post("/login")
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == form_data.username).first()
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(401, "Incorrect email or password")
    token = create_access_token({"sub": str(user.id), "role": user.role})
    return {"access_token": token, "token_type": "bearer", "role": user.role}
