from pydantic import BaseModel
from typing import Optional


class GenerateRequest(BaseModel):
    character: str
    clothes: Optional[str] = ""
    background: Optional[str] = ""
    pose: Optional[str] = ""
    extra_prompt: Optional[str] = ""
    negative_prompt: Optional[str] = ""

    width: int = 1024
    height: int = 1024
    steps: int = 32
    cfg: float = 4.0
    seed: int = -1