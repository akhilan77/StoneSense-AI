from fastapi import APIRouter

router = APIRouter(tags=["root"])


@router.get("/")
def read_root() -> dict[str, str]:
    """Return the service landing message for the root endpoint."""
    return {"message": "StoneSense AI Backend Running 🚀"}
