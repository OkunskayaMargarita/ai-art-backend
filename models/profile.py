from pydantic import BaseModel, Field
from typing import Literal


class ProfileData(BaseModel):
    base_positive: str = "masterpiece, best quality, score_7, no censor, no watermark, no text"
    base_negative: str = "worst quality, low quality, score_1, score_2, score_3, artist name, blurry, jpeg artifacts, lowres, censor"

    width: int = Field(default=768, ge=64, le=2048)
    height: int = Field(default=768, ge=64, le=2048)

    steps: int = Field(default=10, ge=1, le=50)
    cfg: float = Field(default=1.0, ge=0.0, le=30.0)

    batch_size: int = Field(default=1, ge=1, le=8)
    batch_count: int = Field(default=1, ge=1, le=20)

    sampler_name: str = "er_sde"
    scheduler: str = "simple"
    clip_skip: int = Field(default=1, ge=1, le=12)

    hires_enabled: bool = False
    hires_steps: int = Field(default=20, ge=0, le=100)
    hires_scale: float = Field(default=1.5, ge=1.0, le=4.0)
    hires_upscaler: str = "R-ESRGAN 4x+ Anime6B"

    lora_tags: str = ""
    
    engine: Literal["anima"] = "anima"


class ProfileCreate(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    settings: ProfileData