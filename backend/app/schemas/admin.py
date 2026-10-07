"""Pydantic schemas for admin management and audit trail reporting."""

from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel


class AuditLogItem(BaseModel):
    id: int
    user_id: Optional[int] = None
    user_email: Optional[str] = None
    action: str
    resource_type: str
    resource_id: Optional[str] = None
    hospital_id: Optional[int] = None
    ip_address: Optional[str] = None
    details: Optional[Dict[str, Any]] = None
    created_at: datetime

    class Config:
        from_attributes = True


class PaginatedAuditLogsResponse(BaseModel):
    total: int
    page: int
    page_size: int
    total_pages: int
    items: List[AuditLogItem]
