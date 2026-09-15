from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ..models import get_db, User, Patient, Doctor, Message
from ..auth import require_role

router = APIRouter(prefix="/messages", tags=["messages"])


class SendRequest(BaseModel):
    content: str


def _fmt(m: Message, users_by_id: dict) -> dict:
    return {
        "id": m.id,
        "sender_id": m.sender_id,
        "sender_name": users_by_id.get(m.sender_id, "Unknown"),
        "content": m.content,
        "created_at": m.created_at,
    }


def _thread(db: Session, user_a_id: int, user_b_id: int):
    msgs = (db.query(Message)
            .filter(((Message.sender_id == user_a_id) & (Message.receiver_id == user_b_id)) |
                    ((Message.sender_id == user_b_id) & (Message.receiver_id == user_a_id)))
            .order_by(Message.created_at).all())
    ids = {user_a_id, user_b_id}
    users = {u.id: u.full_name for u in db.query(User).filter(User.id.in_(ids)).all()}
    return [_fmt(m, users) for m in msgs]


@router.get("/patient/contact")
def patient_contact(user: User = Depends(require_role("patient")), db: Session = Depends(get_db)):
    """Assigned doctor's info — used both to open the WS chat and to
    pre-select the current doctor in the change-doctor dropdown."""
    patient = db.query(Patient).filter(Patient.user_id == user.id).first()
    if not patient or not patient.assigned_doctor_id:
        raise HTTPException(404, "No doctor assigned")
    doctor = db.query(Doctor).filter(Doctor.id == patient.assigned_doctor_id).first()
    return {"user_id": doctor.user_id, "doctor_id": doctor.id, "full_name": doctor.user.full_name}


@router.get("/patient/thread")
def patient_thread(user: User = Depends(require_role("patient")), db: Session = Depends(get_db)):
    patient = db.query(Patient).filter(Patient.user_id == user.id).first()
    if not patient or not patient.assigned_doctor_id:
        raise HTTPException(404, "No doctor assigned")
    doctor = db.query(Doctor).filter(Doctor.id == patient.assigned_doctor_id).first()
    return _thread(db, user.id, doctor.user_id)


@router.post("/patient/send")
def patient_send(req: SendRequest, user: User = Depends(require_role("patient")), db: Session = Depends(get_db)):
    patient = db.query(Patient).filter(Patient.user_id == user.id).first()
    if not patient or not patient.assigned_doctor_id:
        raise HTTPException(404, "No doctor assigned")
    doctor = db.query(Doctor).filter(Doctor.id == patient.assigned_doctor_id).first()
    db.add(Message(sender_id=user.id, receiver_id=doctor.user_id, content=req.content))
    db.commit()
    return {"status": "sent"}


@router.get("/doctor/thread/{patient_id}")
def doctor_thread(patient_id: int, user: User = Depends(require_role("doctor")), db: Session = Depends(get_db)):
    doctor = db.query(Doctor).filter(Doctor.user_id == user.id).first()
    patient = db.query(Patient).filter(Patient.id == patient_id, Patient.assigned_doctor_id == doctor.id).first()
    if not patient:
        raise HTTPException(404, "Patient not found or not assigned to you")
    return _thread(db, user.id, patient.user_id)


@router.post("/doctor/send/{patient_id}")
def doctor_send(patient_id: int, req: SendRequest, user: User = Depends(require_role("doctor")), db: Session = Depends(get_db)):
    doctor = db.query(Doctor).filter(Doctor.user_id == user.id).first()
    patient = db.query(Patient).filter(Patient.id == patient_id, Patient.assigned_doctor_id == doctor.id).first()
    if not patient:
        raise HTTPException(404, "Patient not found or not assigned to you")
    db.add(Message(sender_id=user.id, receiver_id=patient.user_id, content=req.content))
    db.commit()
    return {"status": "sent"}
