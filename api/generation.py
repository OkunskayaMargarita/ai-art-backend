import time

from fastapi import APIRouter, HTTPException, Request

from generators.comfy_client import generate_with_comfy
from models.generation import GenerateRequest
from models.profile import ProfileData
from services.profile_service import get_profile
from services.prompt_builder import build_prompt


router = APIRouter()


@router.post("/generate")
def generate_image(
    data: GenerateRequest,
    http_request: Request,
):
    started_at = time.perf_counter()

    if not data.profile_name.strip():
        raise HTTPException(
            status_code=400,
            detail="Не выбран профиль генерации.",
        )

    profile_dict = get_profile(data.profile_name)

    if profile_dict is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Профиль «{data.profile_name}» "
                "не найден."
            ),
        )

    try:
        profile = ProfileData.model_validate(profile_dict)
    except ValueError as exc:
        raise HTTPException(
            status_code=500,
            detail=(
                "Выбранный профиль содержит "
                "некорректные настройки."
            ),
        ) from exc

    if profile.engine != "anima":
        raise HTTPException(
            status_code=400,
            detail=(
                "Выбранный профиль не предназначен "
                "для Anima."
            ),
        )

    effective_data = data.model_copy(
        update={
            "steps": data.steps,
            "cfg": data.cfg,
            "sampler_name": data.sampler_name,
            "scheduler": data.scheduler,
            "clip_skip": data.clip_skip,
            "hires_enabled": False,
        }
    )

    prompt_data = build_prompt(
        data=effective_data,
        profile=profile,
    )

    try:
        generation_result = generate_with_comfy(
            prompt=prompt_data["prompt"],
            negative_prompt=prompt_data["negative_prompt"],
            settings=effective_data,
        )
    except RuntimeError as exc:
        raise HTTPException(
            status_code=502,
            detail=str(exc),
        ) from exc

    images = generation_result.get("images", [])

    if not images:
        raise HTTPException(
            status_code=502,
            detail=(
                "ComfyUI завершил запрос, но не вернул "
                "ни одного изображения."
            ),
        )

    base_url = str(http_request.base_url).rstrip("/")

    for image in images:
        image_url = image.get("image_url")

        if image_url and image_url.startswith("/"):
            image["image_url"] = f"{base_url}{image_url}"

    generation_time = round(
        time.perf_counter() - started_at,
        2,
    )

    return {
        "status": "images_generated",
        "engine": "anima",
        "profile_name": data.profile_name,
        "positive_prompt": prompt_data["prompt"],
        "negative_prompt": prompt_data["negative_prompt"],
        "selected_pose": prompt_data["selected_pose"],
        "selected_composition": prompt_data[
            "selected_composition"
        ],
        "character_data": prompt_data["character_data"],
        "character_warning": prompt_data[
            "character_warning"
        ],
        "images": images,
        "actual_seed": generation_result.get("actual_seed"),
        "generation_time_seconds": generation_time,
        "requested_settings": data.model_dump(),
        "effective_settings": effective_data.model_dump(),
        "pose_result": prompt_data["pose_result"],
    }