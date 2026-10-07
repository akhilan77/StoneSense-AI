"""Pydantic schemas for authentication requests, JWT token payloads, and user profiles."""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, ConfigDict, EmailStr, Field


class LoginRequest(BaseModel):
    email: str = Field(..., description="User account email address")
    password: str = Field(..., min_length=6, description="Plaintext account password")


class RefreshRequest(BaseModel):
    refresh_token: str = Field(..., description="Valid JWT refresh token")


class UserOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    email: str
    role: str
    hospital_id: Optional[int] = None
    hospital_code: Optional[str] = None
    is_active: bool
    created_at: datetime


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in_seconds: int
    user: UserOut
