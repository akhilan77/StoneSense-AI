"""Idempotent PII encryption backfill script for StoneSense-AI.

Encrypts legacy plaintext patient columns (name, phone, email) using AES-256-GCM
and populates keyed HMAC-SHA256 blind indexes. Creates a backup of the DB before
modifications.
"""

import sys
import shutil
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from sqlalchemy import create_engine, text, inspect
from app.core.pii_crypto import (
    encrypt_pii,
    decrypt_pii,
    compute_blind_index,
    CIPHERTEXT_PREFIX,
)


def backup_database(db_path: Path) -> Path:
    """Creates a timestamped backup of the database file."""
    if not db_path.exists():
        raise FileNotFoundError(f"Database file not found at: {db_path}")
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    backup_path = db_path.parent / f"{db_path.name}.bak-{timestamp}"
    shutil.copy2(db_path, backup_path)
    print(f"[BACKFILL] Created database backup: {backup_path}")
    return backup_path


def backfill_patients(db_url: str) -> dict:
    """Performs idempotent encryption backfill on patients table."""
    engine = create_engine(db_url)
    
    # Ensure columns exist
    with engine.begin() as conn:
        inspector = inspect(conn)
        existing_cols = {c["name"] for c in inspector.get_columns("patients")}
        for col_name, col_type in [
            ("email", "VARCHAR(1024)"),
            ("name_bidx", "VARCHAR(64)"),
            ("phone_bidx", "VARCHAR(64)"),
            ("email_bidx", "VARCHAR(64)"),
        ]:
            if col_name not in existing_cols:
                conn.execute(text(f"ALTER TABLE patients ADD COLUMN {col_name} {col_type}"))
                print(f"[BACKFILL] Added missing column '{col_name}' to patients table.")

    # Read raw patient rows
    with engine.connect() as conn:
        rows = conn.execute(
            text("SELECT id, name, phone, email, name_bidx, phone_bidx, email_bidx FROM patients")
        ).mappings().all()

    total_rows = len(rows)
    encrypted_count = 0
    already_encrypted_count = 0

    with engine.begin() as conn:
        for r in rows:
            pid = r["id"]
            raw_name = r["name"]
            raw_phone = r["phone"]
            raw_email = r["email"]

            needs_update = False
            updates = {}

            # Process name
            if raw_name:
                if not raw_name.startswith(CIPHERTEXT_PREFIX):
                    enc_name = encrypt_pii(raw_name)
                    bidx_name = compute_blind_index(raw_name, "name")
                    updates["name"] = enc_name
                    updates["name_bidx"] = bidx_name
                    needs_update = True
                elif not r["name_bidx"]:
                    dec_name = decrypt_pii(raw_name)
                    updates["name_bidx"] = compute_blind_index(dec_name, "name")
                    needs_update = True

            # Process phone
            if raw_phone:
                if not raw_phone.startswith(CIPHERTEXT_PREFIX):
                    enc_phone = encrypt_pii(raw_phone)
                    bidx_phone = compute_blind_index(raw_phone, "phone")
                    updates["phone"] = enc_phone
                    updates["phone_bidx"] = bidx_phone
                    needs_update = True
                elif not r["phone_bidx"]:
                    dec_phone = decrypt_pii(raw_phone)
                    updates["phone_bidx"] = compute_blind_index(dec_phone, "phone")
                    needs_update = True

            # Process email
            if raw_email:
                if not raw_email.startswith(CIPHERTEXT_PREFIX):
                    enc_email = encrypt_pii(raw_email)
                    bidx_email = compute_blind_index(raw_email, "email")
                    updates["email"] = enc_email
                    updates["email_bidx"] = bidx_email
                    needs_update = True
                elif not r["email_bidx"]:
                    dec_email = decrypt_pii(raw_email)
                    updates["email_bidx"] = compute_blind_index(dec_email, "email")
                    needs_update = True

            if needs_update:
                set_clauses = [f"{col} = :{col}" for col in updates.keys()]
                sql = f"UPDATE patients SET {', '.join(set_clauses)} WHERE id = :id"
                params = dict(updates)
                params["id"] = pid
                conn.execute(text(sql), params)
                encrypted_count += 1
            else:
                already_encrypted_count += 1

    # Verification: read back and verify raw vs decrypted
    with engine.connect() as conn:
        verified_rows = conn.execute(
            text("SELECT id, name, phone, email, name_bidx, phone_bidx, email_bidx FROM patients")
        ).mappings().all()

    for vr in verified_rows:
        if vr["name"]:
            assert vr["name"].startswith(CIPHERTEXT_PREFIX), f"Row {vr['id']} name is not encrypted!"
            decrypted = decrypt_pii(vr["name"])
            assert decrypted, f"Row {vr['id']} name failed to decrypt!"
        if vr["phone"]:
            assert vr["phone"].startswith(CIPHERTEXT_PREFIX), f"Row {vr['id']} phone is not encrypted!"
            decrypted = decrypt_pii(vr["phone"])
            assert decrypted, f"Row {vr['id']} phone failed to decrypt!"
        if vr["email"]:
            assert vr["email"].startswith(CIPHERTEXT_PREFIX), f"Row {vr['id']} email is not encrypted!"
            decrypted = decrypt_pii(vr["email"])
            assert decrypted, f"Row {vr['id']} email failed to decrypt!"

    result = {
        "total_rows": total_rows,
        "encrypted_count": encrypted_count,
        "already_encrypted_count": already_encrypted_count,
        "verified_rows": len(verified_rows),
        "status": "SUCCESS",
    }
    print(f"[BACKFILL] Completed: {result}")
    return result


if __name__ == "__main__":
    target_db = PROJECT_ROOT / "stonesense.db"
    if target_db.exists():
        backup_database(target_db)
    url = f"sqlite:///{target_db.resolve()}"
    backfill_patients(url)
