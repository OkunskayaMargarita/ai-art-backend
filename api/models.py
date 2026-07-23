import requests
from fastapi import APIRouter, HTTPException

from config.settings import FORGE_API_URL
from pydantic import BaseModel

router = APIRouter(
    prefix="/models",
    tags=["models"],
)


@router.get("")
def list_models():
    try:
        response = requests.get(
            f"{FORGE_API_URL}/sdapi/v1/sd-models",
            timeout=30,
        )
    except requests.ConnectionError as exc:
        raise HTTPException(
            status_code=502,
            detail="Не удалось подключиться к Forge.",
        ) from exc
    except requests.Timeout as exc:
        raise HTTPException(
            status_code=504,
            detail="Forge слишком долго не отвечает.",
        ) from exc

    if not response.ok:
        raise HTTPException(
            status_code=502,
            detail=(
                f"Forge вернул HTTP {response.status_code}: "
                f"{response.text}"
            ),
        )

    forge_models = response.json()

    models = [
        {
            "title": model.get("title"),
            "model_name": model.get("model_name"),
            "filename": model.get("filename"),
            "hash": model.get("hash"),
        }
        for model in forge_models
    ]

    return {
        "models": models,
    }

class ModelSelection(BaseModel):
    model_title: str

@router.get("/current")
def get_current_model():
    try:
        response = requests.get(
            f"{FORGE_API_URL}/sdapi/v1/options",
            timeout=30,
        )
    except requests.ConnectionError as exc:
        raise HTTPException(
            status_code=502,
            detail="Не удалось подключиться к Forge.",
        ) from exc
    except requests.Timeout as exc:
        raise HTTPException(
            status_code=504,
            detail="Forge слишком долго не отвечает.",
        ) from exc

    if not response.ok:
        raise HTTPException(
            status_code=502,
            detail=(
                f"Forge вернул HTTP {response.status_code}: "
                f"{response.text}"
            ),
        )

    options = response.json()
    current_checkpoint = options.get("sd_model_checkpoint", "")

    try:
        models_response = requests.get(
            f"{FORGE_API_URL}/sdapi/v1/sd-models",
            timeout=30,
        )
        models_response.raise_for_status()
        models = models_response.json()
    except requests.RequestException:
        models = []

    matched_title = current_checkpoint

    for model in models:
        title = model.get("title", "")
        model_name = model.get("model_name", "")

        if (
            title == current_checkpoint
            or model_name == current_checkpoint
            or title.startswith(current_checkpoint)
            or current_checkpoint.startswith(model_name)
        ):
            matched_title = title
            break

    return {
        "current_model": matched_title,
    }

@router.post("/select")
def select_model(selection: ModelSelection):
    try:
        response = requests.post(
            f"{FORGE_API_URL}/sdapi/v1/options",
            json={
                "sd_model_checkpoint": selection.model_title,
            },
            timeout=300,
        )
    except requests.ConnectionError as exc:
        raise HTTPException(
            status_code=502,
            detail="Не удалось подключиться к Forge.",
        ) from exc
    except requests.Timeout as exc:
        raise HTTPException(
            status_code=504,
            detail="Forge слишком долго переключает модель.",
        ) from exc

    if not response.ok:
        raise HTTPException(
            status_code=502,
            detail=(
                f"Не удалось выбрать модель: "
                f"HTTP {response.status_code}: {response.text}"
            ),
        )

    return {
        "status": "model_selected",
        "model_title": selection.model_title,
    }