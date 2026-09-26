from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from services.pose_service import (
    build_pose_prompt,
    choose_layout,
    get_pose_library_warnings,
    get_pose_preset,
    list_pose_presets,
)


router = APIRouter(prefix="/poses", tags=["poses"])

class ResolvePoseRequest(BaseModel):
    pose_preset_id: str

@router.get("")
def get_poses():
    poses = list_pose_presets()

    return {
        "poses": [
            {
                "id": pose.id,
                "name": pose.name,
                "description": pose.description,
                "search_terms": pose.search_terms,
                "compatible_contexts": pose.compatible_contexts,
                "layout_count": len(pose.layouts),
            }
            for pose in poses
        ],
        "warnings": get_pose_library_warnings(),
    }

@router.post("/resolve")
def resolve_pose_for_editing(
    data: ResolvePoseRequest,
):
    preset = get_pose_preset(data.pose_preset_id)

    if preset is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Пресет позы "
                f"«{data.pose_preset_id}» не найден."
            ),
        )

    try:
        layout = choose_layout(preset)

        prompt, selected_variants = build_pose_prompt(
            preset=preset,
            layout=layout,
        )

        return {
            "pose_preset_id": preset.id,
            "pose_preset_name": preset.name,
            "prompt": prompt,
        }

    except RuntimeError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc