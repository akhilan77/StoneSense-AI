"""Security utilities: Argon2/PBKDF2 password hashing and PyJWT token lifecycle."""

import hashlib
import os
import secrets
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

import jwt

from app.config.settings import settings

try:
    from argon2 import PasswordHasher
    from argon2.exceptions import VerifyMismatchError

    _argon2_hasher = PasswordHasher(
        time_cost=2,
        memory_cost=65536,
        parallelism=2,
        hash_len=32,
        salt_len=16,
    )
    _HAS_ARGON2 = True
except ImportError:
    _argon2_hasher = None
    _HAS_ARGON2 = False


def hash_password(plain_password: str) -> str:
    """Hashes a plaintext password using Argon2 or PBKDF2 (>=600,000 iterations)."""
    if _HAS_ARGON2 and _argon2_hasher:
        return _argon2_hasher.hash(plain_password)

    # Fallback to PBKDF2-HMAC-SHA256 with 600,000 iterations
    salt = secrets.token_hex(16)
    dk = hashlib.pbkdf2_hmac(
        "sha256",
        plain_password.encode("utf-8"),
        salt.encode("utf-8"),
        600_000,
    )
    return f"pbkdf2_sha256$600000${salt}${dk.hex()}"


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verifies a plaintext password against an Argon2 or PBKDF2 hash."""
    if not hashed_password or not plain_password:
        return False

    if hashed_password.startswith("$argon2"):
        if _HAS_ARGON2 and _argon2_hasher:
            try:
                return _argon2_hasher.verify(hashed_password, plain_password)
            except VerifyMismatchError:
                return False
            except Exception:
                return False
        return False

    if hashed_password.startswith("pbkdf2_sha256$"):
        parts = hashed_password.split("$")
        if len(parts) != 4:
            return False
        iterations = int(parts[1])
        salt = parts[2]
        expected_hex = parts[3]
        dk = hashlib.pbkdf2_hmac(
            "sha256",
            plain_password.encode("utf-8"),
            salt.encode("utf-8"),
            iterations,
        )
        return secrets.compare_digest(dk.hex(), expected_hex)

    return False


def create_access_token(
    data: Dict[str, Any],
    expires_delta: Optional[timedelta] = None,
) -> str:
    """Generates a short-lived JWT access token with token_type='access'."""
    to_encode = data.copy()
    now = datetime.now(timezone.utc)
    if expires_delta:
        expire = now + expires_delta
    else:
        expire = now + timedelta(minutes=settings.access_token_expire_minutes)

    to_encode.update({
        "exp": int(expire.timestamp()),
        "iat": int(now.timestamp()),
        "token_type": "access",
    })
    secret = settings.get_jwt_secret()
    return jwt.encode(to_encode, secret, algorithm=settings.jwt_algorithm)


def create_refresh_token(
    data: Dict[str, Any],
    expires_delta: Optional[timedelta] = None,
) -> str:
    """Generates a long-lived JWT refresh token with token_type='refresh'."""
    to_encode = data.copy()
    now = datetime.now(timezone.utc)
    if expires_delta:
        expire = now + expires_delta
    else:
        expire = now + timedelta(days=settings.refresh_token_expire_days)

    to_encode.update({
        "exp": int(expire.timestamp()),
        "iat": int(now.timestamp()),
        "token_type": "refresh",
    })
    secret = settings.get_jwt_secret()
    return jwt.encode(to_encode, secret, algorithm=settings.jwt_algorithm)


def decode_token(
    token: str,
    expected_type: Optional[str] = "access",
) -> Dict[str, Any]:
    """Decodes and validates a JWT token signature, expiry, and token_type claim."""
    secret = settings.get_jwt_secret()
    try:
        payload = jwt.decode(
            token,
            secret,
            algorithms=[settings.jwt_algorithm],
        )
    except jwt.ExpiredSignatureError:
        raise ValueError("Token has expired")
    except jwt.InvalidTokenError as err:
        raise ValueError(f"Invalid token: {str(err)}")

    if expected_type is not None:
        actual_type = payload.get("token_type")
        if actual_type != expected_type:
            raise ValueError(
                f"Invalid token type: expected '{expected_type}', got '{actual_type}'"
            )

    return payload
