import base64
import os
import uuid
import requests

from config.settings import FORGE_API_URL, OUTPUT_DIR


def generate_with_forge(prompt, negative_prompt, width, height, steps, cfg, seed):
    payload = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "width": width,
        "height": height,
        "steps": steps,
        "cfg_scale": cfg,
        "seed": seed,
        "sampler_name": "Euler a",
        "batch_size": 1,
        "n_iter": 1
    }

    response = requests.post(
        f"{FORGE_API_URL}/sdapi/v1/txt2img",
        json=payload,
        timeout=1800
    )

    response.raise_for_status()

    result = response.json()
    image_base64 = result["images"][0]

    if "," in image_base64:
        image_base64 = image_base64.split(",", 1)[1]

    image_bytes = base64.b64decode(image_base64)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    filename = f"{uuid.uuid4()}.png"
    filepath = os.path.join(OUTPUT_DIR, filename)

    with open(filepath, "wb") as file:
        file.write(image_bytes)

    return {
        "filename": filename,
        "filepath": filepath
    }