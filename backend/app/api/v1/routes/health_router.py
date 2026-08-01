from fastapi import APIRouter

router = APIRouter(tags=["health"])


@router.get("/health")
def health_check() -> dict[str, str]:
    """Return the readiness response for infrastructure health checks."""
    return {"status": "healthy"}
