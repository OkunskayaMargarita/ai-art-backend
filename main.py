from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional

app = FastAPI()


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


@app.get("/")
def home():
    return {
        "status": "ok",
        "message": "AI Art Backend is running"
    }


@app.post("/generate")
def generate_image(data: GenerateRequest):
    prompt_parts = [
        "masterpiece",
        "best quality",
        data.character,
        data.clothes,
        data.background,
        data.pose,
        data.extra_prompt,
    ]

    final_prompt = ", ".join([part for part in prompt_parts if part])
    final_negative = data.negative_prompt or "low quality, bad anatomy, blurry"

    return {
        "status": "prompt_created",
        "prompt": final_prompt,
        "negative_prompt": final_negative,
        "settings": {
            "width": data.width,
            "height": data.height,
            "steps": data.steps,
            "cfg": data.cfg,
            "seed": data.seed
        }
    }