from fastapi import APIRouter, HTTPException

from models.profile import ProfileCreate, ProfileData
from services.profile_service import (
    create_profile,
    delete_profile,
    get_all_profiles,
    get_profile,
    update_profile,
)

router = APIRouter(
    prefix="/profiles",
    tags=["profiles"],
)


@router.get("")
def list_profiles():
    profiles = get_all_profiles()

    return {
        "profiles": profiles,
    }


@router.get("/{name}")
def read_profile(name: str):
    profile = get_profile(name)

    if profile is None:
        raise HTTPException(
            status_code=404,
            detail="Профиль не найден.",
        )

    return {
        "name": name,
        "settings": profile,
    }


@router.post("")
def add_profile(profile: ProfileCreate):
    try:
        saved_settings = create_profile(
            name=profile.name.strip(),
            settings=profile.settings,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=409,
            detail=str(exc),
        ) from exc

    return {
        "status": "profile_created",
        "name": profile.name.strip(),
        "settings": saved_settings,
    }


@router.put("/{name}")
def edit_profile(name: str, settings: ProfileData):
    try:
        saved_settings = update_profile(
            current_name=name,
            settings=settings,
        )
    except KeyError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        ) from exc

    return {
        "status": "profile_updated",
        "name": name,
        "settings": saved_settings,
    }


@router.delete("/{name}")
def remove_profile(name: str):
    try:
        delete_profile(name)
    except KeyError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        ) from exc

    return {
        "status": "profile_deleted",
        "name": name,
    }