from models.generation import GenerateRequest


def build_prompt(data: GenerateRequest) -> dict:
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
        "prompt": final_prompt,
        "negative_prompt": final_negative
    }