"""Authentication API endpoints: login, token refresh, and user profile."""

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy.orm import Session

from app.config.settings import settings
from app.core.auth import get_current_user, log_audit, login_rate_limiter
from app.core.security import (
    create_access_token,
    create_refresh_token,
    decode_token,
    verify_password,
)
from app.db.database import get_db
from app.db.models import Hospital, User
from app.schemas.auth import LoginRequest, RefreshRequest, TokenResponse, UserOut

router = APIRouter(prefix="/auth", tags=["auth"])


def _build_user_out(user: User, db: Session) -> UserOut:
    hospital_code = None
    if user.hospital_id:
        h = db.query(Hospital).filter(Hospital.id == user.hospital_id).first()
        if h:
            hospital_code = h.hospital_code
    return UserOut(
        id=user.id,
        email=user.email,
        role=user.role,
        hospital_id=user.hospital_id,
        hospital_code=hospital_code,
        is_active=user.is_active,
        created_at=user.created_at,
    )


@router.post("/login", response_model=TokenResponse)
def login(
    payload: LoginRequest,
    request: Request,
    db: Session = Depends(get_db),
):
    """Authenticates user credentials and returns JWT access and refresh tokens."""
    client_ip = request.client.host if request.client else "unknown"
    rate_limit_key = f"{client_ip}:{payload.email.strip().lower()}"
    login_rate_limiter.check(rate_limit_key)

    user = db.query(User).filter(User.email == payload.email.strip().lower()).first()
    if not user or not verify_password(payload.password, user.password_hash):
        log_audit(
            db=db,
            action="AUTH_LOGIN_FAILED",
            resource_type="auth",
            ip_address=client_ip,
            details={"email": payload.email.strip().lower(), "reason": "invalid_credentials"},
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not user.is_active:
        log_audit(
            db=db,
            action="AUTH_LOGIN_FAILED",
            resource_type="auth",
            user_id=user.id,
            user_email=user.email,
            ip_address=client_ip,
            details={"reason": "account_inactive"},
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User account is deactivated. Contact system administrator.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token_claims = {
        "sub": user.email,
        "user_id": user.id,
        "role": user.role,
        "hospital_id": user.hospital_id,
    }
    access_token = create_access_token(token_claims)
    refresh_token = create_refresh_token(token_claims)

    log_audit(
        db=db,
        action="AUTH_LOGIN_SUCCESS",
        resource_type="auth",
        user_id=user.id,
        user_email=user.email,
        hospital_id=user.hospital_id,
        ip_address=client_ip,
    )

    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        token_type="bearer",
        expires_in_seconds=settings.access_token_expire_minutes * 60,
        user=_build_user_out(user, db),
    )


@router.post("/refresh")
def refresh_token(
    payload: RefreshRequest,
    db: Session = Depends(get_db),
):
    """Issues a new access token using a valid refresh token."""
    try:
        token_data = decode_token(payload.refresh_token, expected_type="refresh")
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(exc),
            headers={"WWW-Authenticate": "Bearer"},
        )

    user_id = token_data.get("user_id")
    email = token_data.get("sub")
    user = None
    if user_id is not None:
        user = db.query(User).filter(User.id == user_id).first()
    elif email:
        user = db.query(User).filter(User.email == email).first()

    if not user or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User account associated with refresh token is invalid or deactivated.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token_claims = {
        "sub": user.email,
        "user_id": user.id,
        "role": user.role,
        "hospital_id": user.hospital_id,
    }
    new_access_token = create_access_token(token_claims)
    return {
        "access_token": new_access_token,
        "token_type": "bearer",
        "expires_in_seconds": settings.access_token_expire_minutes * 60,
    }


@router.get("/me", response_model=UserOut)
def get_current_user_profile(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Returns profile and active scope for the authenticated user."""
    return _build_user_out(current_user, db)
