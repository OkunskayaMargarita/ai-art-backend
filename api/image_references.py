from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse
from starlette.concurrency import run_in_threadpool

from services.image_reference_service import (
    MAX_BYTES, analyze_reference, reference_path, save_reference, reference_prompt_fields, save_reference_mask,
)

router = APIRouter(prefix="/image-references", tags=["image-references"])


@router.post("")
async def upload_reference(request: Request):
    content = bytearray()
    async for chunk in request.stream():
        content.extend(chunk)
        if len(content) > MAX_BYTES:
            raise HTTPException(413, "Размер изображения должен быть не больше 15 МБ.")
    try:
        return await run_in_threadpool(save_reference, bytes(content))
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@router.get("/{reference_id}")
def preview_reference(reference_id: str):
    try:
        return FileResponse(reference_path(reference_id), media_type="image/png")
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc


@router.post("/{reference_id}/mask")
async def upload_edit_mask(reference_id: str, request: Request):
    content = bytearray()
    async for chunk in request.stream():
        content.extend(chunk)
        if len(content) > MAX_BYTES:
            raise HTTPException(413, "Маска слишком большая.")
    try:
        mask_id = await run_in_threadpool(save_reference_mask, reference_id, bytes(content))
        return {"mask_id": mask_id}
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@router.post("/{reference_id}/analyze")
def describe_reference(reference_id: str):
    try:
        analysis = analyze_reference(reference_id, return_details=True)
        prompt = analysis["prompt"] if isinstance(analysis, dict) else analysis
        characters = analysis.get("characters", []) if isinstance(analysis, dict) else []
        return {"prompt": prompt, "fields": reference_prompt_fields(prompt, characters), "source": "wd-tagger",
                "character_warning": analysis.get("character_warning") if isinstance(analysis, dict) else None}
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(503, str(exc)) from exc
