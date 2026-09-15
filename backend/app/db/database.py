"""
Database engine + session factory for StoneSense-AI.
Drop at: backend/app/db/database.py

Uses SQLite for now (zero extra infra) — swap DATABASE_URL for Postgres later
without touching any route code, since routes only depend on get_db().
"""
import os
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import sessionmaker

from app.db.models import Base

DATABASE_URL = os.getenv("STONESENSE_DATABASE_URL", "sqlite:///./stonesense.db")

connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db() -> None:
    """Call once on backend startup (see INTEGRATION.md)."""
    Base.metadata.create_all(bind=engine)
    if DATABASE_URL.startswith("sqlite"):
        with engine.begin() as connection:
            columns = {column["name"] for column in inspect(connection).get_columns("model_versions")}
            migrations = {
                "model_versions": {"round_id": "INTEGER", "precision": "FLOAT", "recall": "FLOAT"},
                "federated_rounds": {"selected_hospital_ids": "JSON"},
                "hospitals": {
                    "dataset_size": "INTEGER",
                    "class_distribution": "JSON",
                    "dataset_version": "VARCHAR(64)",
                    "dataset_split_counts": "JSON",
                    "dataset_is_valid": "BOOLEAN",
                    "dataset_updated_at": "DATETIME",
                    "dataset_validated_at": "DATETIME",
                    "current_model_version": "VARCHAR(64)",
                },
            }
            for table, table_columns in migrations.items():
                columns = {column["name"] for column in inspect(connection).get_columns(table)}
                for name, definition in table_columns.items():
                    if name not in columns:
                        connection.execute(text(f"ALTER TABLE {table} ADD COLUMN {name} {definition}"))


def get_db():
    """FastAPI dependency — yields a scoped session per request."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
