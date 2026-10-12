"""Unit and integration tests for Patient PII Column-Level Authenticated Encryption.

Verifies:
1. Encrypt/decrypt round trip.
2. Nonce uniqueness (fresh 96-bit random nonce per write).
3. Tamper detection (bit flipping in ciphertext or tag fails authentication).
4. Wrong key rejection.
5. Key rotation (decrypting older versioned ciphertexts while writing with newest key).
6. Deterministic HMAC-SHA256 blind indexing with canonical normalization.
7. Legacy plaintext tolerance on read.
8. Fail-closed startup behavior in production when keys are missing.
9. Database-level verification: raw column contains ciphertext prefix and zero plaintext,
   while ORM transparently decrypts and blind indexes match.
"""

import os
import base64
import pytest
from sqlalchemy import text
from app.core.pii_crypto import (
    PIICryptoManager,
    DecryptionError,
    KeyManagementError,
    encrypt_pii,
    decrypt_pii,
    compute_blind_index,
    CIPHERTEXT_PREFIX,
    pii_crypto,
)
from app.db.models import Patient


def test_encrypt_decrypt_round_trip():
    original = "Dr. Jane Doe, MD - Sensitive Patient Information"
    ciphertext = encrypt_pii(original)
    assert ciphertext is not None
    assert ciphertext.startswith(CIPHERTEXT_PREFIX)
    assert original not in ciphertext

    decrypted = decrypt_pii(ciphertext)
    assert decrypted == original


def test_nonce_uniqueness():
    """Identical plaintexts must produce distinct ciphertexts due to fresh 96-bit nonces."""
    plaintext = "Repeat Clinical Diagnosis"
    ciphertexts = {encrypt_pii(plaintext) for _ in range(50)}
    assert len(ciphertexts) == 50, "Expected 50 distinct ciphertexts for identical plaintext"


def test_tamper_detection():
    """Flipping a single bit in the ciphertext or authentication tag must raise DecryptionError."""
    original = "Top Secret Record"
    ciphertext = encrypt_pii(original)
    prefix_and_kid, b64_payload = ciphertext.rsplit(":", 1)
    raw = bytearray(base64.urlsafe_b64decode(b64_payload.encode("ascii")))

    # Tamper with the last byte (authentication tag)
    raw[-1] ^= 0x01
    tampered_b64 = base64.urlsafe_b64encode(bytes(raw)).decode("ascii")
    tampered_ct = f"{prefix_and_kid}:{tampered_b64}"

    with pytest.raises(DecryptionError, match="Authenticated decryption failed"):
        decrypt_pii(tampered_ct)


def test_wrong_key_fails():
    """Ciphertext encrypted with key A cannot be decrypted with only key B."""
    mgr = PIICryptoManager()
    key_a = os.urandom(32)
    key_b = os.urandom(32)
    mgr.set_keys_for_testing({"v1": key_a}, active_key_id="v1")
    ct = mgr.encrypt("Confidential Data")

    # Re-initialize manager with only key_b
    mgr.set_keys_for_testing({"v1": key_b}, active_key_id="v1")
    with pytest.raises(DecryptionError):
        mgr.decrypt(ct)


def test_key_rotation():
    """Manager with multiple versioned keys decrypts older versions and encrypts with newest."""
    mgr = PIICryptoManager()
    key_v1 = os.urandom(32)
    key_v2 = os.urandom(32)

    # Initial state: only v1 active
    mgr.set_keys_for_testing({"v1": key_v1}, active_key_id="v1")
    v1_ciphertext = mgr.encrypt("Old Record From 2025")
    assert v1_ciphertext.startswith(f"{CIPHERTEXT_PREFIX}v1:")

    # Rotation: v2 is now active, v1 retained for decryption
    mgr.set_keys_for_testing({"v1": key_v1, "v2": key_v2}, active_key_id="v2")
    v2_ciphertext = mgr.encrypt("New Record From 2026")
    assert v2_ciphertext.startswith(f"{CIPHERTEXT_PREFIX}v2:")

    # Both decrypt accurately
    assert mgr.decrypt(v1_ciphertext) == "Old Record From 2025"
    assert mgr.decrypt(v2_ciphertext) == "New Record From 2026"


def test_blind_index_deterministic_and_normalized():
    """Blind index must be deterministic and normalize case, whitespace, and formatting."""
    # Names: case & multiple whitespace normalization
    bidx1 = compute_blind_index("  John   Doe  ", "name")
    bidx2 = compute_blind_index("john doe", "name")
    bidx3 = compute_blind_index("Jane Doe", "name")
    assert bidx1 == bidx2
    assert bidx1 != bidx3

    # Phones: non-digit stripping
    phone1 = compute_blind_index("+1 (555) 019-2834", "phone")
    phone2 = compute_blind_index("15550192834", "phone")
    phone3 = compute_blind_index("+1-555-999-9999", "phone")
    assert phone1 == phone2
    assert phone1 != phone3

    # Emails: lowercasing
    email1 = compute_blind_index("Patient.Test@Hospital.Org", "email")
    email2 = compute_blind_index("patient.test@hospital.org", "email")
    assert email1 == email2


def test_legacy_plaintext_handling():
    """Unencrypted legacy strings must be returned as-is by decrypt_pii without crashing."""
    legacy = "Unencrypted Plaintext Legacy Name"
    assert decrypt_pii(legacy) == legacy
    assert decrypt_pii(None) is None


def test_startup_fails_closed_in_production(monkeypatch):
    """Refuse to start in production without STONESENSE_PII_KEYS."""
    monkeypatch.setenv("STONESENSE_ENV", "production")
    monkeypatch.delenv("STONESENSE_PII_KEYS", raising=False)

    with pytest.raises(KeyManagementError, match="CRITICAL SECURITY FAILURE"):
        PIICryptoManager()


def test_raw_db_column_storage_and_blind_index_lookup(client):
    """Direct database inspection: columns are ciphertext in raw SQL, but decrypted in ORM."""
    from app.db.database import SessionLocal

    db = SessionLocal()
    try:
        patient = Patient(
            hospital_id=1,
            reference_code="SEC-PII-001",
            name="Alice Wonderland",
            phone="+1-555-888-0001",
            email="alice@wonderland.health",
            blood_group="O+",
            urine_features={},
        )
        db.add(patient)
        db.commit()
        pid = patient.id
    finally:
        db.close()

    # 1. Query raw SQL: verify raw bytes in DB file contain ciphertext prefix and NOT plaintext
    db = SessionLocal()
    try:
        raw_row = db.execute(
            text("SELECT name, phone, email, name_bidx, phone_bidx, email_bidx FROM patients WHERE id = :id"),
            {"id": pid},
        ).mappings().first()

        assert raw_row is not None
        assert raw_row["name"].startswith(CIPHERTEXT_PREFIX)
        assert "Alice" not in raw_row["name"]
        assert raw_row["phone"].startswith(CIPHERTEXT_PREFIX)
        assert "555" not in raw_row["phone"]
        assert raw_row["email"].startswith(CIPHERTEXT_PREFIX)
        assert "alice" not in raw_row["email"]

        # Blind indexes are correctly populated
        expected_name_bidx = compute_blind_index("Alice Wonderland", "name")
        expected_phone_bidx = compute_blind_index("+1-555-888-0001", "phone")
        expected_email_bidx = compute_blind_index("alice@wonderland.health", "email")

        assert raw_row["name_bidx"] == expected_name_bidx
        assert raw_row["phone_bidx"] == expected_phone_bidx
        assert raw_row["email_bidx"] == expected_email_bidx

        # 2. Query via ORM: transparently decrypts
        orm_patient = db.query(Patient).filter(Patient.id == pid).first()
        assert orm_patient.name == "Alice Wonderland"
        assert orm_patient.phone == "+1-555-888-0001"
        assert orm_patient.email == "alice@wonderland.health"

        # 3. Query via blind index
        search_match = db.query(Patient).filter(Patient.name_bidx == expected_name_bidx).first()
        assert search_match is not None
        assert search_match.id == pid
    finally:
        db.close()
