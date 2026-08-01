from pydantic import BaseModel, Field


class ImageUpload(BaseModel):
    """Schema supporting the image-based detection request contract.

    The file payload is represented as a high-level upload descriptor for
    the API contract. Actual multipart handling is performed at the route
    boundary so the service layer stays decoupled from transport details.
    """

    image: str = Field(..., description="Filename or object key representing the uploaded image file.")
