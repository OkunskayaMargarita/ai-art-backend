from fastapi import APIRouter

from services.style_service import get_all_styles


router = APIRouter(
    prefix="/styles",
    tags=["styles"],
)


@router.get("")
def get_styles():
    styles = get_all_styles()

    return {
        "styles": [
            {
                "id": style_id,
                "name": style.get("name", style_id),
            }
            for style_id, style in styles.items()
        ]
    }