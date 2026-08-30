"""
Database models for the DFU clinical platform.
SQLite by default (zero setup) — swap DATABASE_URL for Postgres later with no code changes.
"""
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, ForeignKey, Text, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
import os

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./dfu_app.db")
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    role = Column(String, nullable=False)  # "patient" | "doctor" | "admin"
    full_name = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    patient_profile = relationship("Patient", back_populates="user", uselist=False)
    doctor_profile = relationship("Doctor", back_populates="user", uselist=False)


class Doctor(Base):
    __tablename__ = "doctors"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), unique=True, nullable=False)
    specialty = Column(String, default="General")

    user = relationship("User", back_populates="doctor_profile")
    patients = relationship("Patient", back_populates="assigned_doctor")


class Patient(Base):
    __tablename__ = "patients"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), unique=True, nullable=False)
    assigned_doctor_id = Column(Integer, ForeignKey("doctors.id"), nullable=True)
    date_of_birth = Column(String, nullable=True)
    diabetes_type = Column(String, nullable=True)

    user = relationship("User", back_populates="patient_profile")
    assigned_doctor = relationship("Doctor", back_populates="patients")
    visits = relationship("Visit", back_populates="patient", order_by="Visit.created_at")


class Visit(Base):
    """One wound-photo upload + AI analysis + (optional) clinician review."""
    __tablename__ = "visits"
    id = Column(Integer, primary_key=True, index=True)
    public_id = Column(String, unique=True, index=True, nullable=False)
    patient_id = Column(Integer, ForeignKey("patients.id"), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    # image storage (path on disk; swap for S3/blob later without schema change)
    image_path = Column(String, nullable=False)

    # AI results
    coverage_pct = Column(Float, nullable=True)
    infection_score = Column(Float, nullable=True)
    infection_level = Column(String, nullable=True)
    severity_label = Column(String, nullable=True)
    image_quality_ok = Column(Boolean, default=True)
    image_quality_warnings = Column(Text, nullable=True)  # JSON string

    # full heuristic result, stored so doctor view never needs to recompute
    features_json = Column(Text, nullable=True)      # JSON: {Redness, Necrosis, Pus, ...}
    infection_color = Column(String, nullable=True)
    infection_override = Column(Text, nullable=True)

    # clinician review
    review_status = Column(String, default="pending")  # pending | reviewed | flagged
    doctor_notes = Column(Text, nullable=True)
    mask_confirmed = Column(Boolean, nullable=True)  # doctor confirms/rejects AI mask
    reviewed_by_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    reviewed_at = Column(DateTime, nullable=True)

    patient = relationship("Patient", back_populates="visits")
    images = relationship("VisitImage", back_populates="visit", uselist=False)


class VisitImage(Base):
    """Stored separately from Visit (linked by visit_id) so the lightweight
    visit rows/history queries never have to pull large base64 blobs."""
    __tablename__ = "visit_images"
    id = Column(Integer, primary_key=True, index=True)
    visit_id = Column(Integer, ForeignKey("visits.id"), unique=True, nullable=False)
    overlay_b64 = Column(Text, nullable=False)
    boundary_b64 = Column(Text, nullable=False)
    mask_b64 = Column(Text, nullable=False)

    visit = relationship("Visit", back_populates="images")


def init_db():
    Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
