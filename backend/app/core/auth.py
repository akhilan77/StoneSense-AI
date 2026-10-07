"""Authentication dependencies, role-based authorization, rate limiting, and audit logging."""

import time
from collections import defaultdict
from threading import Lock
from typing import Any, Callable, Dict, List, Optional

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from app.config.settings import settings
from app.core.security import decode_token
from app.db.database import get_db
from app.db.models import AuditLog, User

oauth2_scheme = HTTPBearer(auto_error=False)


# -------------------------------------------------------------------------
# Login Rate Limiter
# -------------------------------------------------------------------------

class LoginRateLimiter:
    """Thread-safe in-memory sliding window rate limiter for login attempts."""

    def __init__(self, max_attempts: int = 10, window_sec: int = 60):
        self.max_attempts = max_attempts
        self.window_sec = window_sec
        self.attempts: Dict[str, List[float]] = defaultdict(list)
        self.lock = Lock()

    def check(self, key: str) -> None:
        """Raises HTTP 429 if the request count from key exceeds max_attempts in window."""
        now = time.time()
        with self.lock:
            cutoff = now - self.window_sec
            valid_attempts = [t for t in self.attempts[key] if t > cutoff]
            if len(valid_attempts) >= self.max_attempts:
                retry_after = int(valid_attempts[0] + self.window_sec - now) + 1
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail="Too many login attempts. Please try again later.",
                    headers={"Retry-After": str(max(1, retry_after))},
                )
            valid_attempts.append(now)
            self.attempts[key] = valid_attempts


login_rate_limiter = LoginRateLimiter(
    max_attempts=settings.login_rate_limit_max,
    window_sec=settings.login_rate_limit_window_sec,
)


# -------------------------------------------------------------------------
# User Authentication Dependencies
# -------------------------------------------------------------------------

def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(oauth2_scheme),
    db: Session = Depends(get_db),
) -> User:
    """Authenticates caller via Bearer access token and fetches active User entity."""
    if not credentials or not credentials.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication credentials were not provided.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = credentials.credentials
    try:
        payload = decode_token(token, expected_type="access")
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(exc),
            headers={"WWW-Authenticate": "Bearer"},
        )

    user_id = payload.get("user_id")
    email = payload.get("sub")

    user = None
    if user_id is not None:
        user = db.query(User).filter(User.id == user_id).first()
    elif email:
        user = db.query(User).filter(User.email == email).first()

    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User account associated with this token was not found.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User account is deactivated.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return user


def require_role(*allowed_roles: str) -> Callable[[User], User]:
    """Dependency factory ensuring current_user has one of the required roles."""

    def role_checker(current_user: User = Depends(get_current_user)) -> User:
        if current_user.role not in allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Access forbidden: requires one of roles: {', '.join(allowed_roles)}",
            )
        return current_user

    return role_checker


def require_hospital_scope(
    hospital_id: int,
    current_user: User,
) -> int:
    """Validates that a hospital_user only accesses their own hospital data."""
    if current_user.role == "hospital_user":
        if current_user.hospital_id != hospital_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=(
                    f"Access forbidden: User belongs to hospital ID {current_user.hospital_id}, "
                    f"and cannot access hospital ID {hospital_id}."
                ),
            )
        return current_user.hospital_id

    # Developer or admin can access any specified hospital
    return hospital_id


def get_scoped_hospital_id(
    current_user: User,
    requested_id: Optional[int] = None,
) -> int:
    """Returns the effective hospital_id: strictly enforcing user's hospital for hospital_user."""
    if current_user.role == "hospital_user":
        if current_user.hospital_id is None:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Hospital user account has no assigned hospital scope.",
            )
        return current_user.hospital_id

    # For developer or admin, use requested or default to 1
    if requested_id is not None and requested_id > 0:
        return requested_id
    return 1


# -------------------------------------------------------------------------
# Audit Logger Utility
# -------------------------------------------------------------------------

# Sanitized keys allowed in audit details (strictly no raw patient PII / features)
ALLOWED_AUDIT_KEYS = {
    "patient_id",
    "reference_code",
    "prediction_id",
    "prediction_type",
    "model_version",
    "model_family",
    "result_label",
    "confidence",
    "version_tag",
    "previous_version_id",
    "round_number",
    "action_source",
    "count",
    "status",
}


def log_audit(
    db: Session,
    action: str,
    resource_type: str,
    resource_id: Optional[str] = None,
    hospital_id: Optional[int] = None,
    user_id: Optional[int] = None,
    user_email: Optional[str] = None,
    ip_address: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None,
) -> Optional[AuditLog]:
    """Writes an immutable, non-PII audit trail entry."""
    sanitized_details: Optional[Dict[str, Any]] = None
    if details:
        sanitized_details = {
            k: v for k, v in details.items()
            if k in ALLOWED_AUDIT_KEYS and not isinstance(v, (list, dict))
        }

    audit_entry = AuditLog(
        user_id=user_id,
        user_email=user_email,
        action=action,
        resource_type=resource_type,
        resource_id=str(resource_id) if resource_id is not None else None,
        hospital_id=hospital_id,
        ip_address=ip_address,
        details=sanitized_details,
    )
    try:
        db.add(audit_entry)
        db.commit()
        return audit_entry
    except Exception:
        db.rollback()
        return None
