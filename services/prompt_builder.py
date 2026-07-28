from models.generation import GenerateRequest
from models.profile import ProfileData
from services.character_prompt_enhancer import enhance_character_prompt
from services.pose_service import resolve_pose


def clean_part(value: str | None) -> str:
    if not value:
        return ""

    return value.strip().strip(",")


def humanize_tag(value: str) -> str:
    return value.replace("_", " ").strip()


def build_character_prompt(character_name: str) -> tuple[str, dict, str | None]:
    character_name = clean_part(character_name)

    if not character_name:
        return "", {
            "found": False,
            "input": "",
            "enhanced_prompt": "",
        }, None

    try:
        character_data = enhance_character_prompt(
            character_name=character_name,
            post_limit=100,
            maximum_identity_tags=20,
        )

        prompt_parts = character_data.get("prompt_parts", [])

        readable_parts = [
            humanize_tag(part)
            for part in prompt_parts
            if clean_part(part)
        ]

        if readable_parts:
            character_prompt = ", ".join(
                dict.fromkeys(readable_parts)
            )
        else:
            character_prompt = character_name

        return character_prompt, character_data, None

    except RuntimeError as exc:
        return (
            character_name,
            {
                "found": False,
                "input": character_name,
                "enhanced_prompt": character_name,
            },
            str(exc),
        )


def choose_composition(pose: str) -> str:
    pose_lower = pose.lower()

    if any(
        word in pose_lower
        for word in ("lying", "reclining", "on the bed")
    ):
        return (
            "full body composition, natural camera angle, "
            "the whole pose clearly visible"
        )

    if any(
        word in pose_lower
        for word in ("sitting", "kneeling", "crouching")
    ):
        return (
            "medium full shot, character filling most of the frame, "
            "eye-level anime camera angle, clear readable silhouette, "
            "face and body clearly visible"
        )

    if any(
        word in pose_lower
        for word in ("portrait", "close-up", "close up")
    ):
        return "portrait composition, face clearly visible"

    return (
        "medium full shot, character filling most of the frame, "
        "balanced anime composition, clear silhouette, "
        "natural eye-level camera angle"
    )


def join_prompt_parts(parts: list[str]) -> str:
    cleaned_parts = [
        clean_part(part)
        for part in parts
        if clean_part(part)
    ]

    unique_parts = list(dict.fromkeys(cleaned_parts))

    return ", ".join(unique_parts)


def build_prompt(
    data: GenerateRequest,
    profile: ProfileData,
) -> dict:
    (
        character_prompt,
        character_data,
        character_warning,
    ) = build_character_prompt(data.character)

    pose_result = resolve_pose(
        pose_mode=data.pose_mode,
        manual_pose=data.pose,
        pose_preset_id=data.pose_preset_id,
        environment=data.background,
        width=data.width,
        height=data.height,
    )
    final_pose = pose_result["prompt"]

    if pose_result["source"] == "free":
        automatic_composition = ""
    else:
        automatic_composition = choose_composition(final_pose)

    final_prompt = join_prompt_parts(
        [
            profile.base_positive,
            character_prompt,
            data.clothes,
            final_pose,
            automatic_composition,
            data.background,
            data.extra_tags,
        ]
    )

    final_negative_prompt = join_prompt_parts(
        [
            profile.base_negative,
            data.user_negative,
        ]
    )

    return {
        "prompt": final_prompt,
        "negative_prompt": final_negative_prompt,
        "character_data": character_data,
        "character_warning": character_warning,
        "selected_pose": final_pose,
        "selected_composition": automatic_composition,
        "pose_result": pose_result,
    }