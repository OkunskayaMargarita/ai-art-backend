from fastapi import APIRouter, HTTPException, Query

from services.danbooru_service import (
    enrich_character,
    resolve_character_tag,
    search_character_tags,
)
from services.character_cache import (
    clear_character_cache,
    delete_character_cache,
)
from services.character_prompt_enhancer import (
    enhance_character_prompt,
)


router = APIRouter(
    prefix="/characters",
    tags=["characters"],
)


@router.get("/search")
def search_characters(
    name: str = Query(min_length=1),
    limit: int = Query(default=10, ge=1, le=50),
):
    try:
        results = search_character_tags(
            character_name=name,
            limit=limit,
        )
    except RuntimeError as exc:
        raise HTTPException(
            status_code=502,
            detail=str(exc),
        ) from exc

    return {
        "query": name,
        "results": results,
    }


@router.get("/resolve")
def resolve_character(
    name: str = Query(min_length=1),
):
    try:
        result = resolve_character_tag(name)
    except RuntimeError as exc:
        raise HTTPException(
            status_code=502,
            detail=str(exc),
        ) from exc

    return result
    
@router.get("/enrich")
def enrich_character_endpoint(
    name: str = Query(min_length=1),
post_limit: int = Query(
    default=100,
    ge=5,
    le=200,
),
    maximum_tags: int = Query(
        default=20,
        ge=1,
        le=50,
    ),
):
    try:
        result = enrich_character(
            character_name=name,
            post_limit=post_limit,
            maximum_tags=maximum_tags,
        )
    except RuntimeError as exc:
        raise HTTPException(
            status_code=502,
            detail=str(exc),
        ) from exc

    return result
    
@router.delete("/cache")
def clear_all_character_cache():
    deleted_count = clear_character_cache()

    return {
        "status": "cache_cleared",
        "deleted_files": deleted_count,
    }


@router.delete("/cache/{character_name}")
def delete_cached_character(character_name: str):
    deleted = delete_character_cache(character_name)

    if not deleted:
        raise HTTPException(
            status_code=404,
            detail="Кэш этого персонажа не найден.",
        )

    return {
        "status": "character_cache_deleted",
        "character": character_name,
    }
    
@router.get("/enhance-prompt")
def enhance_character_prompt_endpoint(
    name: str = Query(min_length=1),
    post_limit: int = Query(
        default=100,
        ge=10,
        le=200,
    ),
    maximum_identity_tags: int = Query(
        default=20,
        ge=1,
        le=50,
    ),
):
    try:
        result = enhance_character_prompt(
            character_name=name,
            post_limit=post_limit,
            maximum_identity_tags=maximum_identity_tags,
        )
    except RuntimeError as exc:
        raise HTTPException(
            status_code=502,
            detail=str(exc),
        ) from exc

    return result