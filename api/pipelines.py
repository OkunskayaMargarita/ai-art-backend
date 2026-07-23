from fastapi import APIRouter, HTTPException, Query

from services.pipeline_selector import (
    load_pipelines,
    select_character_pipeline,
)
from services.model_manager import switch_model
from pydantic import BaseModel


router = APIRouter(
    prefix="/pipelines",
    tags=["pipelines"],
)
class PipelineActivationRequest(BaseModel):
    pipeline_name: str


@router.get("")
def get_pipelines():
    try:
        pipelines = load_pipelines()
    except RuntimeError as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc

    return {
        "pipelines": pipelines,
    }


@router.get("/select")
def select_pipeline(
    character: str = Query(min_length=1),
):
    try:
        result = select_character_pipeline(character)
    except RuntimeError as exc:
        raise HTTPException(
            status_code=502,
            detail=str(exc),
        ) from exc

    return result
    
@router.post("/activate")
def activate_pipeline(
    data: PipelineActivationRequest,
):
    try:
        pipelines = load_pipelines()

        pipeline = pipelines.get(
            data.pipeline_name
        )

        if pipeline is None:
            raise HTTPException(
                status_code=404,
                detail="Указанный пайплайн не найден.",
            )

        model_title = pipeline.get(
            "model_title",
            "",
        )

        model_result = switch_model(model_title)

    except HTTPException:
        raise
    except RuntimeError as exc:
        raise HTTPException(
            status_code=502,
            detail=str(exc),
        ) from exc

    return {
        "status": "pipeline_activated",
        "pipeline_name": data.pipeline_name,
        "pipeline": pipeline,
        "model_result": model_result,
    }