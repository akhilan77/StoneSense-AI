"""StoneSense-AI Column-Level PII Encryption and Blind Indexing Engine.

Implements AES-256-GCM authenticated encryption at rest with 96-bit nonces,
key versioning/rotation, fail-closed enforcement for production, and keyed
HMAC-SHA256 blind indexing for exact-match searches.
"""

import os
import re
import hmac
import base64
import hashlib
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.exceptions import InvalidTag

logger = logging.getLogger("StoneSensePIICrypto")

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEV_KEY_FILE = PROJECT_ROOT / ".stonesense_pii_dev_key"
CIPHERTEXT_PREFIX = "enc:v1:"


class DecryptionError(Exception):
    """Raised when ciphertext authentication tag fails, data is tampered, or key is missing."""
    pass


class KeyManagementError(RuntimeError):
    """Raised when key configuration fails closed in production or non-dev environments."""
    pass


class PIICryptoManager:
    """Manages AES-256-GCM keys, authenticated encryption/decryption, and blind indexes."""

    def __init__(self):
        self._keys: Dict[str, bytes] = {}
        self._active_key_id: str = "v1"
        self._blind_index_key: bytes = b""
        self._initialized: bool = False
        self._initialize_keys()

    def _initialize_keys(self) -> None:
        """Loads master keys from environment, failing closed in production."""
        env = os.getenv("STONESENSE_ENV", os.getenv("ENVIRONMENT", "production")).strip().lower()
        is_dev_or_test = env in ("development", "dev", "testing", "test")

        raw_keys = os.getenv("STONESENSE_PII_KEYS", "").strip()
        active_id = os.getenv("STONESENSE_PII_ACTIVE_KEY_ID", "v1").strip()
        raw_bidx_key = os.getenv("STONESENSE_PII_BLIND_INDEX_KEY", "").strip()

        if raw_keys:
            # Parse keyid:base64key pairs
            pairs = [p.strip() for p in raw_keys.split(",") if p.strip()]
            for pair in pairs:
                if ":" not in pair:
                    raise KeyManagementError(f"Invalid PII key format in STONESENSE_PII_KEYS: '{pair}'. Expected keyid:base64key")
                kid, b64_key = pair.split(":", 1)
                try:
                    key_bytes = base64.b64decode(b64_key.strip())
                    if len(key_bytes) != 32:
                        raise KeyManagementError(f"Key '{kid}' is {len(key_bytes)} bytes. AES-256 requires exactly 32 bytes.")
                    self._keys[kid.strip()] = key_bytes
                except Exception as e:
                    raise KeyManagementError(f"Failed to base64-decode key '{kid}': {e}")
            self._active_key_id = active_id
        elif is_dev_or_test:
            # Development / Testing fallback: generate or load local dev key
            self._load_or_generate_dev_keys()
        else:
            # Production: FAIL CLOSED
            raise KeyManagementError(
                "CRITICAL SECURITY FAILURE: STONESENSE_PII_KEYS environment variable is not set. "
                "Production environment refuses to start without explicit authenticated encryption keys."
            )

        # Ensure active key exists
        if self._active_key_id not in self._keys:
            if self._keys:
                # Default to first available key if active_id not found
                self._active_key_id = next(iter(self._keys.keys()))
            elif not is_dev_or_test:
                raise KeyManagementError(f"Active key ID '{self._active_key_id}' not found in configured keys.")

        # Blind index key setup
        if raw_bidx_key:
            try:
                bidx_bytes = base64.b64decode(raw_bidx_key)
                if len(bidx_bytes) < 16:
                    raise KeyManagementError("STONESENSE_PII_BLIND_INDEX_KEY must be at least 16 bytes.")
                self._blind_index_key = bidx_bytes
            except Exception as e:
                raise KeyManagementError(f"Failed to decode STONESENSE_PII_BLIND_INDEX_KEY: {e}")
        else:
            # Derive deterministic blind index key from active master key
            active_key = self._keys[self._active_key_id]
            self._blind_index_key = hashlib.sha256(b"stonesense_blind_index_salt:" + active_key).digest()

        self._initialized = True

    def _load_or_generate_dev_keys(self) -> None:
        """In development/testing only: loads or creates a persistent dev key in a gitignored file."""
        if DEV_KEY_FILE.exists():
            try:
                content = DEV_KEY_FILE.read_text(encoding="utf-8").strip()
                lines = [l.strip() for l in content.splitlines() if l.strip() and not l.startswith("#")]
                for line in lines:
                    kid, b64_k = line.split(":", 1)
                    self._keys[kid.strip()] = base64.b64decode(b64_k.strip())
                self._active_key_id = "dev_v1"
                logger.warning("DEVELOPMENT: Loaded existing auto-generated dev PII keys from .stonesense_pii_dev_key")
                return
            except Exception as e:
                logger.warning(f"Could not read existing dev key file: {e}. Generating new key.")

        # Generate fresh 32-byte master key
        dev_key = os.urandom(32)
        dev_b64 = base64.b64encode(dev_key).decode("ascii")
        self._keys["dev_v1"] = dev_key
        self._active_key_id = "dev_v1"

        try:
            DEV_KEY_FILE.write_text(
                f"# Automatically generated development PII master key. DO NOT COMMIT.\ndev_v1:{dev_b64}\n",
                encoding="utf-8",
            )
            logger.warning("DEVELOPMENT: Generated new development PII master key saved to .stonesense_pii_dev_key (gitignored).")
        except Exception as e:
            logger.warning(f"Could not write dev key file: {e}. Key held in-memory only.")

    def set_keys_for_testing(self, keys: Dict[str, bytes], active_key_id: str, blind_index_key: Optional[bytes] = None) -> None:
        """Utility for unit testing key rotation and multiple keys."""
        self._keys = dict(keys)
        self._active_key_id = active_key_id
        if blind_index_key:
            self._blind_index_key = blind_index_key
        else:
            self._blind_index_key = hashlib.sha256(b"stonesense_bidx:" + keys[active_key_id]).digest()

    def encrypt(self, plaintext: Optional[str], key_id: Optional[str] = None) -> Optional[str]:
        """Encrypts plaintext using AES-256-GCM with a fresh 96-bit random nonce.

        Format: enc:v1:{key_id}:{base64(nonce_12B + ciphertext_and_tag)}
        """
        if plaintext is None:
            return None
        kid = key_id or self._active_key_id
        if kid not in self._keys:
            raise KeyManagementError(f"Encryption key '{kid}' not found in active key registry.")

        key = self._keys[kid]
        aesgcm = AESGCM(key)

        # Fresh 96-bit random nonce for every single encryption
        nonce = os.urandom(12)
        aad = f"v1:{kid}".encode("utf-8")
        data = plaintext.encode("utf-8")

        ciphertext_and_tag = aesgcm.encrypt(nonce, data, aad)
        payload = base64.urlsafe_b64encode(nonce + ciphertext_and_tag).decode("ascii")

        return f"{CIPHERTEXT_PREFIX}{kid}:{payload}"

    def decrypt(self, ciphertext: Optional[str]) -> Optional[str]:
        """Decrypts AES-256-GCM ciphertext, or returns legacy plaintext unmodified.

        Raises DecryptionError on tamper, corrupted tag, or missing key.
        """
        if ciphertext is None:
            return None

        # Legacy plaintext tolerance during backfill migration
        if not ciphertext.startswith(CIPHERTEXT_PREFIX):
            return ciphertext

        # Parse format: enc:v1:{key_id}:{payload}
        stripped = ciphertext[len(CIPHERTEXT_PREFIX):]
        if ":" not in stripped:
            raise DecryptionError("Malformed ciphertext format: missing key ID delimiter.")

        kid, b64_payload = stripped.split(":", 1)
        if kid not in self._keys:
            raise DecryptionError(f"Cannot decrypt: key ID '{kid}' is not available in key registry.")

        key = self._keys[kid]
        try:
            raw_bytes = base64.urlsafe_b64decode(b64_payload.encode("ascii"))
        except Exception as e:
            raise DecryptionError(f"Base64 decoding failed for ciphertext payload: {e}")

        if len(raw_bytes) < 28:  # 12-byte nonce + minimum 16-byte GCM tag
            raise DecryptionError("Ciphertext payload is shorter than required nonce and tag length.")

        nonce = raw_bytes[:12]
        ct_and_tag = raw_bytes[12:]
        aad = f"v1:{kid}".encode("utf-8")

        aesgcm = AESGCM(key)
        try:
            plaintext_bytes = aesgcm.decrypt(nonce, ct_and_tag, aad)
            return plaintext_bytes.decode("utf-8")
        except InvalidTag:
            raise DecryptionError("Authenticated decryption failed: invalid tag or data was tampered.")
        except Exception as e:
            raise DecryptionError(f"Decryption error: {e}")

    def compute_blind_index(self, value: Optional[str], field_type: str = "generic") -> Optional[str]:
        """Computes a keyed HMAC-SHA256 blind index over a normalized value for exact matching."""
        if not value:
            return None

        normalized = self._normalize_for_indexing(value, field_type)
        if not normalized:
            return None

        mac = hmac.new(self._blind_index_key, normalized.encode("utf-8"), hashlib.sha256).hexdigest()
        return mac

    def _normalize_for_indexing(self, value: str, field_type: str) -> str:
        """Canonical normalization to guarantee deterministic exact-match indexing."""
        val = value.strip().lower()
        if field_type == "phone":
            # Extract digits only for phone numbers (e.g. +1 (555) 019-2831 -> 15550192831)
            digits = re.sub(r"\D", "", val)
            return digits if digits else val
        elif field_type == "email":
            return val
        elif field_type == "name":
            # Collapse multiple spaces and strip punctuation
            return re.sub(r"\s+", " ", val)
        return val


# Singleton manager instance
pii_crypto = PIICryptoManager()


def encrypt_pii(plaintext: Optional[str], key_id: Optional[str] = None) -> Optional[str]:
    """Helper function to encrypt a PII string."""
    return pii_crypto.encrypt(plaintext, key_id=key_id)


def decrypt_pii(ciphertext: Optional[str]) -> Optional[str]:
    """Helper function to decrypt a PII string (tolerant to legacy plaintext)."""
    return pii_crypto.decrypt(ciphertext)


def compute_blind_index(value: Optional[str], field_type: str = "generic") -> Optional[str]:
    """Helper function to compute HMAC-SHA256 blind index."""
    return pii_crypto.compute_blind_index(value, field_type)
