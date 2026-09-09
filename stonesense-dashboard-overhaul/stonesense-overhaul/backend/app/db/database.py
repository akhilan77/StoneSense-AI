"""
Database engine + session factory for StoneSense-AI.
Drop at: backend/app/db/database.py

Uses SQLite for now (zero extra infra) — swap DATABASE_URL for Postgres later
without touching any route code, since routes only depend on get_db().
"""
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db.models import Base

DATABASE_URL = os.getenv("STONESENSE_DATABASE_URL", "sqlite:///./stonesense.db")

connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db() -> None:
    """Call once on backend startup (see INTEGRATION.md)."""
    Base.metadata.create_all(bind=engine)


def get_db():
    """FastAPI dependency — yields a scoped session per request."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
