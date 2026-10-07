"""Admin routes for StoneSense-AI (Audit trail query & management)."""

from math import ceil
from typing import Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy import desc
from sqlalchemy.orm import Session

from app.core.auth import require_role
from app.db.database import get_db
from app.db.models import AuditLog, User
from app.schemas.admin import PaginatedAuditLogsResponse

router = APIRouter(prefix="/admin", tags=["admin"])


@router.get("/audit-logs", response_model=PaginatedAuditLogsResponse)
def get_audit_logs(
    page: int = Query(default=1, ge=1, description="Page number starting at 1"),
    page_size: int = Query(default=20, ge=1, le=100, description="Items per page"),
    action: Optional[str] = Query(default=None, description="Filter by action name"),
    user_email: Optional[str] = Query(default=None, description="Filter by user email"),
    hospital_id: Optional[int] = Query(default=None, description="Filter by hospital ID"),
    resource_type: Optional[str] = Query(default=None, description="Filter by resource type"),
    current_user: User = Depends(require_role("admin")),
    db: Session = Depends(get_db),
) -> PaginatedAuditLogsResponse:
    """Returns paginated, searchable system audit logs for administrative compliance."""
    query = db.query(AuditLog)

    if action:
        query = query.filter(AuditLog.action == action)
    if user_email:
        query = query.filter(AuditLog.user_email == user_email)
    if hospital_id is not None:
        query = query.filter(AuditLog.hospital_id == hospital_id)
    if resource_type:
        query = query.filter(AuditLog.resource_type == resource_type)

    total = query.count()
    offset = (page - 1) * page_size
    items = query.order_by(desc(AuditLog.created_at)).offset(offset).limit(page_size).all()

    total_pages = ceil(total / page_size) if total > 0 else 1

    return PaginatedAuditLogsResponse(
        total=total,
        page=page,
        page_size=page_size,
        total_pages=total_pages,
        items=items,
    )
