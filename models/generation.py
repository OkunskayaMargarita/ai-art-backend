from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
    profile_name: str = ""

    character: str = ""
    background: str = ""
    clothes: str = ""
    pose: str = ""
    extra_tags: str = ""
    user_negative: str = ""

    width: int = Field(default=768, ge=64, le=4096)
    height: int = Field(default=1024, ge=64, le=4096)

    steps: int = Field(default=10, ge=1, le=50)
    cfg: float = Field(default=1.0, ge=0.0, le=30.0)
    seed: int = -1

    batch_size: int = Field(default=1, ge=1, le=8)
    batch_count: int = Field(default=1, ge=1, le=20)

    sampler_name: str = "er_sde"
    scheduler: str = "simple"
    clip_skip: int = Field(default=1, ge=1, le=12)

    hires_enabled: bool = False
    hires_steps: int = Field(default=20, ge=0, le=100)
    hires_scale: float = Field(default=1.5, ge=1.0, le=4.0)
    hires_upscaler: str = "R-ESRGAN 4x+ Anime6B"