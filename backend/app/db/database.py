import os
from pathlib import Path
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import sessionmaker

from app.db.models import Base

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SQLITE_PATH = (PROJECT_ROOT / "stonesense.db").resolve()
DATABASE_URL = os.getenv("STONESENSE_DATABASE_URL", f"sqlite:///{DEFAULT_SQLITE_PATH}")

connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db() -> None:
    """Call once on backend startup (see INTEGRATION.md)."""
    Base.metadata.create_all(bind=engine)
    if DATABASE_URL.startswith("sqlite"):
        with engine.begin() as connection:
            migrations = {
                "model_versions": {
                    "round_id": "INTEGER",
                    "precision": "FLOAT",
                    "recall": "FLOAT",
                    "status": "VARCHAR(32) DEFAULT 'pending_review'",
                    "gate_report": "JSON",
                    "approved_by": "VARCHAR(128)",
                    "approved_at": "DATETIME",
                    "deployed_at": "DATETIME",
                    "previous_deployed_version_id": "INTEGER",
                },
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

            # Legacy status migration and cleanup
            # 1. Update deployed rows to status='deployed'
            connection.execute(
                text(
                    "UPDATE model_versions SET status = 'deployed' "
                    "WHERE is_deployed = 1 AND (status IS NULL OR status = 'pending_review')"
                )
            )
            # 2. Update non-deployed legacy rows to status='archived' with gate_report
            connection.execute(
                text(
                    "UPDATE model_versions SET status = 'archived', gate_report = '{\"legacy\": true}' "
                    "WHERE (is_deployed = 0 OR is_deployed IS NULL) AND (status IS NULL OR status = 'pending_review') AND gate_report IS NULL"
                )
            )
            # 3. Enforce strictly at most one deployed row per model_family (keep most recently trained)
            cursor = connection.execute(
                text("SELECT id, model_family, is_deployed FROM model_versions WHERE is_deployed = 1 ORDER BY trained_at DESC")
            )
            deployed_rows = cursor.fetchall()
            seen_families = set()
            for row_id, family, _ in deployed_rows:
                if family in seen_families:
                    connection.execute(
                        text("UPDATE model_versions SET is_deployed = 0, status = 'archived' WHERE id = :rid"),
                        {"rid": row_id}
                    )
                else:
                    seen_families.add(family)


def get_db():
    """FastAPI dependency — yields a scoped session per request."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
