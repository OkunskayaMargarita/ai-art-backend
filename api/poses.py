from fastapi import APIRouter

from services.pose_service import get_pose_library_warnings, list_pose_presets


router = APIRouter(prefix="/poses", tags=["poses"])


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
