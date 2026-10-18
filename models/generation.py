from pydantic import BaseModel, Field
from typing import Literal

class GenerateRequest(BaseModel):
    tags_from_reference: bool = False
    reference_character_name: str = Field(default="", max_length=1000)
    reference_character_traits: list[str] = Field(default_factory=list, max_length=64)
    reference_id: str = Field(default="", pattern=r"^(?:[a-f0-9]{32})?$")
    reference_enabled: bool = False
    reference_prompt_from_fields: bool = False
    reference_instruction_edit: bool = False
    reference_preserve_style: bool = True
    reference_mask_id: str = Field(default="", pattern=r"^(?:[a-f0-9]{32})?$")
    reference_edit_prompt: str = Field(default="", max_length=8000)
    reference_prompt: str = Field(default="", max_length=16000)
    reference_changes: str = Field(default="", max_length=8000)
    reference_strength: float = Field(default=0.35, ge=0.05, le=0.95)

    profile_name: str = ""
    style_name: str = "default"

    character: str = ""
    background: str = ""
    clothes: str = ""
    
    pose_mode: Literal[
        "automatic",
        "preset",
        "manual",
        "free",
    ] = "automatic"
    
    pose: str = ""
    pose_preset_id: str = ""
    pose_override: str = ""
    
    extra_tags: str = ""
    user_negative: str = ""

    width: int = Field(default=768, ge=64, le=4096)
    height: int = Field(default=1024, ge=64, le=4096)

    steps: int = Field(default=10, ge=1, le=50)
    cfg: float = Field(default=1.0, ge=0.0, le=30.0)
    seed: int = -1

    batch_size: int = Field(default=1, ge=1, le=8)
    batch_count: int = Field(default=1, ge=1)

    sampler_name: str = "er_sde"
    scheduler: str = "simple"
    clip_skip: int = Field(default=1, ge=1, le=12)

    hires_enabled: bool = False
    hires_steps: int = Field(default=20, ge=0, le=100)
    hires_scale: float = Field(default=1.5, ge=1.0, le=4.0)
    hires_upscaler: str = "R-ESRGAN 4x+ Anime6B"
