import re

from models.generation import GenerateRequest
from models.profile import ProfileData
from services.character_prompt_enhancer import enhance_character_prompt
from services.pose_service import resolve_pose
from services.style_service import get_style_prompt


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


def choose_composition(
    pose: str,
    pose_result: dict,
) -> str:
    selected_variants = pose_result.get(
        "selected_variants",
        {}
    )

    if selected_variants.get("camera"):
        return ""
        
    pose_lower = pose.lower()

    if any(
        word in pose_lower
        for word in ("lying", "reclining", "on the bed")
    ):
        return "full body composition"

    if any(
        word in pose_lower
        for word in ("sitting", "kneeling", "crouching")
    ):
        return "medium full shot"

    if any(
        word in pose_lower
        for word in ("portrait", "close-up", "close up")
    ):
        return "portrait composition"

    return "full body composition"


def join_prompt_parts(parts: list[str], *, semantic: bool = False) -> str:
    # Split only at top-level commas, preserving weighted expressions and LoRAs.
    unique_parts = []
    seen = set()
    for part in parts:
        start = depth = 0
        text = clean_part(part)
        escaped = False
        for index, char in enumerate(text + ","):
            if escaped:
                escaped = False
                continue
            if char == "\\":
                escaped = True
            elif char in "([{<":
                depth += 1
            elif char in ")]}>":
                depth = max(0, depth - 1)
            elif char == "," and depth == 0:
                tag = text[start:index].strip()
                key = " ".join(tag.replace("_", " ").casefold().split())
                if key and key not in seen:
                    unique_parts.append(tag)
                    seen.add(key)
                start = index + 1
        # Keep malformed/unclosed user expressions intact rather than discard them.
        if start < len(text):
            tag = text[start:].strip()
            key = " ".join(tag.replace("_", " ").casefold().split())
            if key and key not in seen:
                unique_parts.append(tag)
                seen.add(key)
    if semantic:
        from services.tag_semantics import remove_implied_tags
        unique_parts = remove_implied_tags(unique_parts)
    return ", ".join(unique_parts)
    
def apply_profile_prompt_replacements(
    prompt: str,
    profile_name: str | None,
) -> str:
    if not prompt:
        return prompt

    if profile_name and profile_name.casefold() == "futa":
        prompt = re.sub(
            r"\bvagina\b",
            "penis",
            prompt,
            flags=re.IGNORECASE,
        )

    return prompt


def build_prompt(
    data: GenerateRequest,
    profile: ProfileData,
) -> dict:
    if data.reference_instruction_edit:
        instruction = (
            f"Edit image 1: {data.reference_edit_prompt.strip()}. "
            "Apply the requested change completely, replacing conflicting original features. "
            "Preserve only unrequested details, including identity and framing. "
        )
        if data.reference_preserve_style:
            from services.reference_style_service import reference_rendering_style

            style = reference_rendering_style(data.reference_id)
            if style:
                instruction += (
                    "Image 2 shows a close-up of the original rendering technique, not new content. "
                    "Unless explicitly requested otherwise, use exactly that technique for all changed "
                    "and new elements, including the background. "
                )
                instruction += f"{style['instruction']} "
            else:
                instruction += (
                    "Unless explicitly requested otherwise, preserve the reference's rendering technique "
                    "in all new elements: match its marks, edges, shading, texture and abstraction. "
                )
        else:
            instruction += "Apply the requested style change consistently throughout the image. "
        instruction += "Change only what was requested. Do not add text, signatures or watermarks."
        return {
            "prompt": instruction,
            "negative_prompt": join_prompt_parts([profile.base_negative, data.user_negative]),
            "character_data": {},
            "character_warning": None, "selected_style": "none", "style_prompt": "",
            "selected_pose": "", "selected_composition": "",
            "pose_result": {"source": "instruction_edit", "prompt": ""},
        }
    if data.reference_enabled and data.reference_mask_id:
        return {
            "prompt": join_prompt_parts([profile.base_positive, data.reference_edit_prompt], semantic=True),
            "negative_prompt": join_prompt_parts([profile.base_negative, data.user_negative]),
            "character_data": {}, "character_warning": None,
            "selected_style": "none", "style_prompt": "", "selected_pose": "",
            "selected_composition": "", "pose_result": {"source": "reference_edit", "prompt": ""},
        }
    # Reference mode deliberately avoids automatic background, pose and identity additions.
    if data.reference_enabled and not data.reference_prompt_from_fields:
        return {
            "prompt": join_prompt_parts([data.reference_prompt, data.reference_changes], semantic=True),
            "negative_prompt": join_prompt_parts([profile.base_negative, data.user_negative]),
            "character_data": {}, "character_warning": None,
            "selected_style": "default", "style_prompt": "",
            "selected_pose": "", "selected_composition": "",
            "pose_result": {"source": "reference", "prompt": ""},
        }

    (
        character_prompt,
        character_data,
        character_warning,
    ) = ((data.character, {"source": "reference_tags"}, None)
         if data.tags_from_reference else build_character_prompt(data.character))
    if (data.tags_from_reference and data.reference_character_name
            and data.character == data.reference_character_name):
        # Traits were confirmed during analysis; keep the snapshot so a network
        # failure on generation cannot lose traits omitted from the visible form.
        character_prompt = join_prompt_parts([data.character, *data.reference_character_traits])
        character_data = {"source": "reference_character", "found": True,
                          "input": data.character, "enhanced_prompt": character_prompt}
    
    style_prompt = get_style_prompt(data.style_name)

    pose_override = clean_part(data.pose_override)

    if (
        data.pose_mode == "preset"
        and pose_override
    ):
        pose_result = {
            "source": "preset_override",
            "preset_id": data.pose_preset_id,
            "preset_name": None,
            "layout_id": None,
            "layout_name": None,
            "prompt": pose_override,
            "selected_variants": {},
        }
    else:
        pose_result = resolve_pose(
            pose_mode=data.pose_mode,
            manual_pose=data.pose,
            pose_preset_id=data.pose_preset_id,
            environment=data.background,
            width=data.width,
            height=data.height,
        )

    final_pose = pose_result["prompt"]

    if pose_result["source"] in ("free", "manual"):
        automatic_composition = ""
    else:
        automatic_composition = choose_composition(
            final_pose,
            pose_result,
        )
        
    background_prompt = (
        data.background.strip()
        if data.background.strip()
        else "character in detailed furnished and light room"
    )

    final_prompt = join_prompt_parts(
        [
            profile.base_positive,
            style_prompt,
            character_prompt,
            data.clothes,
            final_pose,
            automatic_composition,
            background_prompt,
            data.extra_tags,
        ]
    )

    final_negative_prompt = join_prompt_parts(
        [
            profile.base_negative,
            data.user_negative,
        ]
    )
    
    final_prompt = apply_profile_prompt_replacements(
        final_prompt,
        data.profile_name,
    )
    final_prompt = join_prompt_parts([final_prompt], semantic=True)

    return {
        "prompt": final_prompt,
        "negative_prompt": final_negative_prompt,
        "character_data": character_data,
        "character_warning": character_warning,
        "selected_style": data.style_name,
        "style_prompt": style_prompt,
        "selected_pose": final_pose,
        "selected_composition": automatic_composition,
        "pose_result": pose_result,
    }
