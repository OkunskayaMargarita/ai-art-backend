from fastapi import APIRouter
from models.generation import GenerateRequest
from services.prompt_builder import build_prompt
from generators.forge_client import generate_with_forge

router = APIRouter()


@router.post("/generate")
def generate_image(data: GenerateRequest):
    prompt_data = build_prompt(data)

    image_data = generate_with_forge(
        prompt=prompt_data["prompt"],
        negative_prompt=prompt_data["negative_prompt"],
        width=data.width,
        height=data.height,
        steps=data.steps,
        cfg=data.cfg,
        seed=data.seed
    )

    return {
        "status": "image_generated",
        "prompt": prompt_data["prompt"],
        "negative_prompt": prompt_data["negative_prompt"],
        "image": image_data,
        "settings": {
            "width": data.width,
            "height": data.height,
            "steps": data.steps,
            "cfg": data.cfg,
            "seed": data.seed
        }
    }