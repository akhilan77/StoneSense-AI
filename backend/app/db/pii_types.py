"""SQLAlchemy TypeDecorator for column-level AES-256-GCM PII authenticated encryption."""

from sqlalchemy import String
from sqlalchemy.types import TypeDecorator
from app.core.pii_crypto import encrypt_pii, decrypt_pii


class EncryptedString(TypeDecorator):
    """SQLAlchemy TypeDecorator that transparently encrypts strings on write

    and decrypts on read using AES-256-GCM with fresh 96-bit nonces.
    Tolerates legacy plaintext values during migrations.
    """
    impl = String
    cache_ok = True

    def __init__(self, length=255, **kwargs):
        # Increase underlying column size to accommodate headers, key ID, nonce, and ciphertext
        storage_len = max(length * 4, 512)
        super().__init__(storage_len, **kwargs)

    def process_bind_param(self, value, dialect):
        if value is None:
            return None
        if isinstance(value, str) and value.startswith("enc:v1:"):
            return value
        return encrypt_pii(str(value))

    def process_result_value(self, value, dialect):
        if value is None:
            return None
        return decrypt_pii(value)
