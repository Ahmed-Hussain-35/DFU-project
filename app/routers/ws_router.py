import json
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query

from ..models import SessionLocal, User, Patient, Doctor, Message
from ..auth import get_user_from_token_sync
from ..email_utils import send_email

router = APIRouter()


class ConnectionManager:
    """In-memory registry: user_id -> set of open WebSocket connections
    (a user may have several tabs open at once)."""
    def __init__(self):
        self.active = {}

    async def connect(self, user_id, ws):
        await ws.accept()
        self.active.setdefault(user_id, set()).add(ws)

    def disconnect(self, user_id, ws):
        if user_id in self.active:
            self.active[user_id].discard(ws)
            if not self.active[user_id]:
                del self.active[user_id]

    async def send_to(self, user_id, data):
        for ws in list(self.active.get(user_id, [])):
            try:
                await ws.send_json(data)
            except Exception:
                pass


manager = ConnectionManager()


def _pair_allowed(db, user, other_user_id) -> bool:
    """Only let a patient message their own assigned doctor, and a doctor
    message a patient actually assigned to them — same rule the REST
    endpoints already enforce."""
    if user.role == "patient":
        patient = db.query(Patient).filter(Patient.user_id == user.id).first()
        if not patient or not patient.assigned_doctor_id:
            return False
        doctor = db.query(Doctor).filter(Doctor.id == patient.assigned_doctor_id).first()
        return bool(doctor and doctor.user_id == other_user_id)
    if user.role == "doctor":
        doctor = db.query(Doctor).filter(Doctor.user_id == user.id).first()
        if not doctor:
            return False
        patient = db.query(Patient).filter(Patient.user_id == other_user_id,
                                           Patient.assigned_doctor_id == doctor.id).first()
        return patient is not None
    return False


@router.websocket("/ws/chat")
async def chat_ws(websocket: WebSocket, token: str = Query(...)):
    db = SessionLocal()
    user = get_user_from_token_sync(token, db)
    if not user:
        await websocket.close(code=4001)
        db.close()
        return

    await manager.connect(user.id, websocket)
    try:
        while True:
            raw = await websocket.receive_text()
            try:
                payload = json.loads(raw)
                to_user_id = int(payload["to_user_id"])
                content = payload["content"].strip()
            except (json.JSONDecodeError, KeyError, ValueError):
                continue
            if not content or not _pair_allowed(db, user, to_user_id):
                continue

            msg = Message(sender_id=user.id, receiver_id=to_user_id, content=content)
            db.add(msg)
            db.commit()
            db.refresh(msg)

            out = {"id": msg.id, "sender_id": user.id, "sender_name": user.full_name,
                  "content": content, "created_at": msg.created_at.isoformat()}
            await manager.send_to(to_user_id, out)
            await manager.send_to(user.id, out)  # echo to sender's other tabs

            receiver = db.query(User).filter(User.id == to_user_id).first()
            if receiver:
                send_email(receiver.email, "New message — DFU Clinical Platform",
                           f"{user.full_name} sent you a message: {content}")
    except WebSocketDisconnect:
        pass
    finally:
        manager.disconnect(user.id, websocket)
        db.close()
