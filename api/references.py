from fastapi import APIRouter, HTTPException, Query

from services.reference_service import (
    find_character_references,
)
from services.ip_adapter import prepare_reference


router = APIRouter(
    prefix="/references",
    tags=["references"],
)


@router.get("/character")
def get_character_references(
    name: str = Query(min_length=1),
    post_limit: int = Query(
        default=100,
        ge=10,
        le=200,
    ),
    maximum_references: int = Query(
        default=3,
        ge=1,
        le=5,
    ),
):
    try:
        result = find_character_references(
            character_name=name,
            post_limit=post_limit,
            maximum_references=maximum_references,
        )
    except RuntimeError as exc:
        raise HTTPException(
            status_code=502,
            detail=str(exc),
        ) from exc

    return result
    
@router.get("/prepare")
def prepare_character_reference(
    name: str = Query(min_length=1),
):
    try:
        result = prepare_reference(
            character_name=name,
        )
    except RuntimeError as exc:
        raise HTTPException(
            status_code=502,
            detail=str(exc),
        ) from exc

    image_base64 = result.pop("image_base64")

    return {
        **result,
        "base64_length": len(image_base64),
        "base64_preview": image_base64[:100],
    }