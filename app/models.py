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
    role = Column(String, nullable=False)
    full_name = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    patient_profile = relationship("Patient", back_populates="user", uselist=False)
    doctor_profile = relationship("Doctor", back_populates="user", uselist=False)


class Doctor(Base):
    __tablename__ = "doctors"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), unique=True, nullable=False)
    specialty = Column(String, default="General")
    license_number = Column(String, nullable=True)
    medical_council = Column(String, nullable=True)
    license_doc_path = Column(String, nullable=True)
    is_verified = Column(Boolean, default=False)

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
    """One wound-photo upload + AI analysis (Phase 1 and/or Phase 2) + optional clinician review."""
    __tablename__ = "visits"
    id = Column(Integer, primary_key=True, index=True)
    public_id = Column(String, unique=True, index=True, nullable=False)
    patient_id = Column(Integer, ForeignKey("patients.id"), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    image_path = Column(String, nullable=False)
    phases_run = Column(String, default="phase1")   # "phase1" | "phase2" | "phase1,phase2"

    # Phase 1 (rule-based) — original columns
    coverage_pct = Column(Float, nullable=True)
    infection_score = Column(Float, nullable=True)
    infection_level = Column(String, nullable=True)
    severity_label = Column(String, nullable=True)
    image_quality_ok = Column(Boolean, default=True)
    image_quality_warnings = Column(Text, nullable=True)
    patient_notes = Column(Text, nullable=True)
    features_json = Column(Text, nullable=True)
    infection_color = Column(String, nullable=True)
    infection_override = Column(Text, nullable=True)

    # Phase 2 (ML pipeline) — new columns
    p2_coverage_pct = Column(Float, nullable=True)
    p2_infection_score = Column(Float, nullable=True)
    p2_infection_level = Column(String, nullable=True)
    p2_infection_color = Column(String, nullable=True)
    p2_infection_override = Column(Text, nullable=True)
    p2_severity_label = Column(String, nullable=True)
    p2_features_json = Column(Text, nullable=True)
    p2_wagner_grade = Column(Integer, nullable=True)
    p2_wagner_confidence = Column(Float, nullable=True)
    p2_wagner_title = Column(String, nullable=True)
    p2_wagner_description = Column(Text, nullable=True)
    p2_wagner_color = Column(String, nullable=True)

    # Review
    review_status = Column(String, default="pending")
    doctor_notes = Column(Text, nullable=True)
    mask_confirmed = Column(Boolean, nullable=True)
    reviewed_by_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    reviewed_at = Column(DateTime, nullable=True)

    patient = relationship("Patient", back_populates="visits")
    images = relationship("VisitImage", back_populates="visit", uselist=False)


class VisitImage(Base):
    __tablename__ = "visit_images"
    id = Column(Integer, primary_key=True, index=True)
    visit_id = Column(Integer, ForeignKey("visits.id"), unique=True, nullable=False)

    # Phase 1 overlays
    overlay_b64 = Column(Text, nullable=True)
    boundary_b64 = Column(Text, nullable=True)
    mask_b64 = Column(Text, nullable=True)

    # Phase 2 overlays
    p2_overlay_b64 = Column(Text, nullable=True)
    p2_boundary_b64 = Column(Text, nullable=True)
    p2_mask_b64 = Column(Text, nullable=True)
    p2_tissue_b64 = Column(Text, nullable=True)

    visit = relationship("Visit", back_populates="images")


class Message(Base):
    __tablename__ = "messages"
    id = Column(Integer, primary_key=True, index=True)
    sender_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    receiver_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)


def init_db():
    Base.metadata.create_all(bind=engine)
    # Lightweight in-place migration for SQLite: add any missing Phase 2 columns.
    if "sqlite" in DATABASE_URL:
        _sqlite_migrate()


def _sqlite_migrate():
    from sqlalchemy import inspect, text
    insp = inspect(engine)

    def add_cols(table, wanted):
        existing = {c["name"] for c in insp.get_columns(table)}
        with engine.begin() as conn:
            for col_name, col_type in wanted.items():
                if col_name not in existing:
                    conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {col_name} {col_type}"))

    add_cols("visits", {
        "phases_run": "TEXT DEFAULT 'phase1'",
        "p2_coverage_pct": "FLOAT",
        "p2_infection_score": "FLOAT",
        "p2_infection_level": "TEXT",
        "p2_infection_color": "TEXT",
        "p2_infection_override": "TEXT",
        "p2_severity_label": "TEXT",
        "p2_features_json": "TEXT",
        "p2_wagner_grade": "INTEGER",
        "p2_wagner_confidence": "FLOAT",
        "p2_wagner_title": "TEXT",
        "p2_wagner_description": "TEXT",
        "p2_wagner_color": "TEXT",
    })
    add_cols("visit_images", {
        "p2_overlay_b64": "TEXT",
        "p2_boundary_b64": "TEXT",
        "p2_mask_b64": "TEXT",
        "p2_tissue_b64": "TEXT",
    })


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()