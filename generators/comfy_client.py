import copy
import json
import math
import os
import random
import threading
import time
import uuid

import requests

from config.settings import (
    ANIMA_WORKFLOW_PATH,
    COMFY_API_URL,
    OUTPUT_DIR,
    REFERENCE_EDIT_WORKFLOW_PATH,
    REFERENCE_EDIT_NEGATIVE_CFG,
)
from models.generation import GenerateRequest
from services.generation_control import (
    is_stop_requested,
    reset_stop_request,
)
from pathlib import Path


generation_lock = threading.Lock()

image_counter_lock = threading.Lock()

BACKEND_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BACKEND_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_COUNTER_FILE = DATA_DIR / "image_counter.json"


def get_next_image_number() -> int:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    last_number = 0

    if os.path.exists(IMAGE_COUNTER_FILE):
        try:
            with open(
                IMAGE_COUNTER_FILE,
                "r",
                encoding="utf-8",
            ) as counter_file:
                counter_data = json.load(counter_file)

            last_number = int(
                counter_data.get("last_image_number", 0)
            )

        except (OSError, ValueError, TypeError):
            raise RuntimeError(
                "Не удалось прочитать счётчик изображений."
            )

    # Дополнительная защита от совпадения с уже
    # существующими файлами.
    for filename in os.listdir(OUTPUT_DIR):
        name, extension = os.path.splitext(filename)

        if extension.lower() == ".png" and name.isdigit():
            last_number = max(last_number, int(name))

    return last_number + 1


def save_image_counter(number: int) -> None:
    temporary_file = IMAGE_COUNTER_FILE.with_suffix(".json.tmp")

    with open(
        temporary_file,
        "w",
        encoding="utf-8",
    ) as counter_file:
        json.dump(
            {"last_image_number": number},
            counter_file,
            indent=2,
        )

    os.replace(
        temporary_file,
        IMAGE_COUNTER_FILE,
    )

PROMPT_NODE_ID = "60:11"
NEGATIVE_PROMPT_NODE_ID = "60:12"
SAMPLER_NODE_ID = "60:19"
LATENT_NODE_ID = "60:28"
SAVE_IMAGE_NODE_ID = "46"
OUTPUT_SCALE_NODE_ID = "waigen_output_scale"

# Keep existing presets through 2560x1440 unchanged; this is a compatibility
# threshold, not a guaranteed model limit. Use a conservative base for 4K.
MAX_DIRECT_GENERATION_PIXELS = 2560 * 1440
LARGE_OUTPUT_BASE_PIXELS = 1536 * 864


def load_anima_workflow() -> dict:
    try:
        with open(
            ANIMA_WORKFLOW_PATH,
            "r",
            encoding="utf-8",
        ) as workflow_file:
            return json.load(workflow_file)
    except FileNotFoundError as exc:
        raise RuntimeError(
            "Не найден workflow Anima: "
            f"{ANIMA_WORKFLOW_PATH}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            "Workflow Anima содержит некорректный JSON."
        ) from exc


def prepare_workflow(
    prompt: str,
    negative_prompt: str,
    settings: GenerateRequest,
    seed: int,
    reference_image: str | None = None,
    reference_mask: str | None = None,
) -> dict:
    if settings.reference_instruction_edit:
        return prepare_instruction_edit_workflow(prompt, settings, seed, reference_image, negative_prompt=negative_prompt)
    workflow = copy.deepcopy(load_anima_workflow())

    workflow[PROMPT_NODE_ID]["inputs"]["text"] = prompt
    workflow[NEGATIVE_PROMPT_NODE_ID]["inputs"]["text"] = (
        negative_prompt
    )

    workflow[LATENT_NODE_ID]["inputs"]["width"] = settings.width
    workflow[LATENT_NODE_ID]["inputs"]["height"] = settings.height
    workflow[LATENT_NODE_ID]["inputs"]["batch_size"] = (
        settings.batch_size
    )

    output_pixels = settings.width * settings.height
    if output_pixels > MAX_DIRECT_GENERATION_PIXELS:
        scale = math.sqrt(LARGE_OUTPUT_BASE_PIXELS / output_pixels)
        # Align the base latent to the model grid, retaining the aspect ratio
        # to within rounding. Resize decoded pixels, never the noisy latent.
        workflow[LATENT_NODE_ID]["inputs"]["width"] = max(
            64, int(settings.width * scale) // 16 * 16
        )
        workflow[LATENT_NODE_ID]["inputs"]["height"] = max(
            64, int(settings.height * scale) // 16 * 16
        )
        workflow[OUTPUT_SCALE_NODE_ID] = {
            "class_type": "ImageScale",
            "inputs": {
                "image": workflow[SAVE_IMAGE_NODE_ID]["inputs"]["images"],
                "upscale_method": "lanczos",
                "width": settings.width,
                "height": settings.height,
                "crop": "disabled",
            },
        }
        workflow[SAVE_IMAGE_NODE_ID]["inputs"]["images"] = [
            OUTPUT_SCALE_NODE_ID, 0
        ]

    sampler_inputs = workflow[SAMPLER_NODE_ID]["inputs"]

    sampler_inputs["seed"] = seed

    # Для Anima Turbo используем проверенные настройки.
    sampler_inputs["steps"] = settings.steps
    sampler_inputs["cfg"] = settings.cfg
    sampler_inputs["sampler_name"] = "er_sde"
    sampler_inputs["scheduler"] = "simple"
    sampler_inputs["denoise"] = 1
    if settings.reference_enabled:
        if not reference_image:
            raise RuntimeError("Референс не загружен в ComfyUI.")
        base = workflow[LATENT_NODE_ID]["inputs"]
        workflow["waigen_reference_load"] = {
            "class_type": "LoadImage", "inputs": {"image": reference_image},
        }
        workflow["waigen_reference_resize"] = {
            "class_type": "ImageScale", "inputs": {
                "image": ["waigen_reference_load", 0], "upscale_method": "lanczos",
                "width": base["width"], "height": base["height"],
                "crop": "disabled" if settings.reference_mask_id else "center",
            },
        }
        workflow["waigen_reference_encode"] = {
            "class_type": "VAEEncode", "inputs": {
                "pixels": ["waigen_reference_resize", 0], "vae": ["60:15", 0],
            },
        }
        if settings.reference_mask_id:
            if not reference_mask:
                raise RuntimeError("Маска не загружена в ComfyUI.")
            workflow["waigen_edit_mask_load"] = {
                "class_type": "LoadImage", "inputs": {"image": reference_mask},
            }
            workflow["waigen_edit_mask_resize"] = {
                "class_type": "ImageScale", "inputs": {
                    "image": ["waigen_edit_mask_load", 0], "upscale_method": "nearest-exact",
                    "width": base["width"], "height": base["height"], "crop": "disabled",
                },
            }
            workflow["waigen_edit_mask"] = {
                "class_type": "ImageToMask", "inputs": {
                    "image": ["waigen_edit_mask_resize", 0], "channel": "red",
                },
            }
            workflow["waigen_reference_encode"]["class_type"] = "VAEEncodeForInpaint"
            workflow["waigen_reference_encode"]["inputs"].update({
                "mask": ["waigen_edit_mask", 0], "grow_mask_by": 6,
            })
        workflow["waigen_reference_batch"] = {
            "class_type": "RepeatLatentBatch", "inputs": {
                "samples": ["waigen_reference_encode", 0], "amount": settings.batch_size,
            },
        }
        sampler_inputs["latent_image"] = ["waigen_reference_batch", 0]
        sampler_inputs["denoise"] = settings.reference_strength

    workflow[SAVE_IMAGE_NODE_ID]["inputs"]["filename_prefix"] = (
        f"Anima_{uuid.uuid4().hex[:8]}"
    )

    return workflow


def prepare_instruction_edit_workflow(prompt: str, settings: GenerateRequest, seed: int,
                                      reference_image: str | None, *, negative_prompt: str = "") -> dict:
    if not reference_image:
        raise RuntimeError("Референс не загружен для редактирования.")
    path = BACKEND_DIR / REFERENCE_EDIT_WORKFLOW_PATH
    with path.open(encoding="utf-8") as file:
        workflow = json.load(file)
    workflow["edit_image"]["inputs"]["image"] = reference_image
    workflow["edit_text"]["inputs"]["text"] = prompt
    if negative_prompt.strip():
        workflow["edit_negative_text"] = {"class_type": "CLIPTextEncode", "inputs": {
            "text": negative_prompt, "clip": ["edit_clip", 0],
        }}
        workflow["edit_negative"]["inputs"]["conditioning"] = ["edit_negative_text", 0]
        # CFG=1 skips the negative branch entirely in ComfyUI.
        workflow["edit_guider"]["inputs"]["cfg"] = REFERENCE_EDIT_NEGATIVE_CFG
    workflow["edit_noise"]["inputs"]["noise_seed"] = seed
    for node in ("edit_resize", "edit_sigmas", "edit_latent"):
        workflow[node]["inputs"].update({"width": settings.width, "height": settings.height})
    workflow["edit_latent"]["inputs"]["batch_size"] = settings.batch_size
    from services.reference_style_service import reference_rendering_style
    if settings.reference_preserve_style and reference_rendering_style(settings.reference_id):
        # Keep the complete editing schedule: partial denoising prevented scene
        # and garment changes. Style is supplied through visual conditioning.
        # A magnified detail exposes surface marks lost in the full-frame view.
        # Both references remain image conditioning, never a pixel overlay.
        width = max(32, settings.width // 4 // 16 * 16)
        height = max(32, settings.height // 4 // 16 * 16)
        workflow["edit_style_crop"] = {
            "class_type": "ImageCrop", "inputs": {
                "image": ["edit_resize", 0], "width": width, "height": height,
                "x": (settings.width - width) // 2, "y": settings.height // 10,
            },
        }
        workflow["edit_style_encode"] = {
            "class_type": "VAEEncode", "inputs": {"pixels": ["edit_style_crop", 0], "vae": ["edit_vae", 0]},
        }
        for branch in ("positive", "negative"):
            name = f"edit_style_{branch}"
            workflow[name] = {"class_type": "ReferenceLatent", "inputs": {
                "conditioning": [f"edit_{branch}", 0], "latent": ["edit_style_encode", 0],
            }}
            workflow["edit_guider"]["inputs"][branch] = [name, 0]
    workflow[SAVE_IMAGE_NODE_ID]["inputs"]["filename_prefix"] = f"WaiGenEdit_{uuid.uuid4().hex[:8]}"
    return workflow


def queue_workflow(workflow: dict) -> str:
    try:
        response = requests.post(
            f"{COMFY_API_URL}/prompt",
            json={"prompt": workflow},
            timeout=30,
        )
    except requests.ConnectionError as exc:
        raise RuntimeError(
            "Не удалось подключиться к ComfyUI. "
            "Проверь, что ComfyUI запущен на порту 8188."
        ) from exc
    except requests.Timeout as exc:
        raise RuntimeError(
            "ComfyUI не ответил при отправке workflow."
        ) from exc

    if not response.ok:
        raise RuntimeError(
            f"ComfyUI вернул HTTP {response.status_code}: "
            f"{response.text}"
        )

    result = response.json()
    prompt_id = result.get("prompt_id")

    if not prompt_id:
        raise RuntimeError(
            "ComfyUI не вернул prompt_id."
        )

    return prompt_id


def wait_for_history(
    prompt_id: str,
    timeout_seconds: int = 1800,
) -> dict:
    started_at = time.monotonic()

    while time.monotonic() - started_at < timeout_seconds:
        try:
            response = requests.get(
                f"{COMFY_API_URL}/history/{prompt_id}",
                timeout=30,
            )
        except requests.RequestException as exc:
            raise RuntimeError(
                "Ошибка при чтении статуса ComfyUI."
            ) from exc

        if not response.ok:
            raise RuntimeError(
                f"ComfyUI history вернул HTTP "
                f"{response.status_code}: {response.text}"
            )

        history = response.json()
        prompt_history = history.get(prompt_id)

        if prompt_history:
            status = prompt_history.get("status", {})

            if status.get("status_str") == "error":
                messages = status.get("messages", [])
                raise RuntimeError(
                    "ComfyUI завершил генерацию с ошибкой: "
                    f"{messages}"
                )

            outputs = prompt_history.get("outputs", {})

            if outputs:
                return prompt_history

        time.sleep(0.5)

    raise RuntimeError(
        "ComfyUI слишком долго не завершает генерацию."
    )


def download_output_image(image_data: dict, settings: GenerateRequest | None = None) -> dict:
    params = {
        "filename": image_data["filename"],
        "subfolder": image_data.get("subfolder", ""),
        "type": image_data.get("type", "output"),
    }

    try:
        response = requests.get(
            f"{COMFY_API_URL}/view",
            params=params,
            timeout=120,
        )
    except requests.RequestException as exc:
        raise RuntimeError(
            "Не удалось скачать изображение из ComfyUI."
        ) from exc

    if not response.ok:
        raise RuntimeError(
            f"ComfyUI /view вернул HTTP "
            f"{response.status_code}: {response.text}"
        )

    content = response.content
    output_size = None
    if settings and settings.reference_mask_id:
        from services.image_reference_service import composite_reference_edit
        try:
            content, output_size = composite_reference_edit(settings.reference_id, settings.reference_mask_id, content)
        except (OSError, ValueError) as exc:
            raise RuntimeError("Не удалось восстановить исходные пиксели вне маски. Результат не сохранён.") from exc
    elif settings and settings.reference_instruction_edit:
        from PIL import Image
        from io import BytesIO
        from services.image_reference_service import reference_path
        with Image.open(reference_path(settings.reference_id)) as original:
            output_size = original.size
        with Image.open(BytesIO(content)) as generated:
            resized = generated.convert("RGB").resize(output_size, Image.Resampling.LANCZOS)
        buffer = BytesIO()
        resized.save(buffer, format="PNG")
        content = buffer.getvalue()

    with image_counter_lock:
        image_number = get_next_image_number()
        
        filename = f"{image_number}.png"
        filepath = os.path.join(OUTPUT_DIR, filename)

    # Режим xb запрещает перезаписывать существующий файл.
        with open(filepath, "xb") as output_file:
            output_file.write(content)

        save_image_counter(image_number)

        return {
            "filename": filename,
            "filepath": filepath,
            "image_url": f"/outputs/{filename}",
            **({"width": output_size[0], "height": output_size[1]} if output_size else {}),
        }


def extract_images(prompt_history: dict, settings: GenerateRequest | None = None) -> list[dict]:
    outputs = prompt_history.get("outputs", {})
    saved_images = []

    for node_output in outputs.values():
        for image_data in node_output.get("images", []):
            saved_images.append(
                download_output_image(image_data, settings)
            )

    if not saved_images:
        raise RuntimeError(
            "ComfyUI завершил workflow, но не вернул изображений."
        )

    return saved_images


def generate_with_comfy(
    prompt: str,
    negative_prompt: str,
    settings: GenerateRequest,
) -> dict:
    actual_seed = (
        settings.seed
        if settings.seed >= 0
        else random.randint(0, 2**63 - 1)
    )

    all_images = []

    reset_stop_request()

    try:
        with generation_lock:
            reference_image = None
            reference_mask = None
            if settings.reference_enabled:
                from services.image_reference_service import reference_path
                try:
                    path = reference_path(settings.reference_id)
                    with path.open("rb") as image_file:
                        response = requests.post(
                            f"{COMFY_API_URL}/upload/image",
                            files={"image": (f"waigen_{settings.reference_id}.png", image_file, "image/png")},
                            data={"type": "input", "overwrite": "true"}, timeout=60,
                        )
                    response.raise_for_status()
                    uploaded = response.json()
                    name = uploaded["name"]
                    subfolder = uploaded.get("subfolder", "")
                    reference_image = f"{subfolder}/{name}" if subfolder else name
                    if settings.reference_mask_id:
                        from services.image_reference_service import reference_mask_path
                        mask_path = reference_mask_path(settings.reference_id, settings.reference_mask_id)
                        with mask_path.open("rb") as mask_file:
                            response = requests.post(
                                f"{COMFY_API_URL}/upload/image",
                                files={"image": (f"waigen_mask_{settings.reference_mask_id}.png", mask_file, "image/png")},
                                data={"type": "input", "overwrite": "true"}, timeout=60,
                            )
                        response.raise_for_status()
                        uploaded = response.json()
                        name = uploaded["name"]
                        subfolder = uploaded.get("subfolder", "")
                        reference_mask = f"{subfolder}/{name}" if subfolder else name
                except (ValueError, OSError, requests.RequestException, KeyError) as exc:
                    raise RuntimeError("Не удалось передать референс в ComfyUI. Проверьте подключение и повторите загрузку.") from exc
            for iteration_index in range(settings.batch_count):
                iteration_seed = actual_seed + iteration_index

                workflow = prepare_workflow(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    settings=settings,
                    seed=iteration_seed,
                    reference_image=reference_image,
                    reference_mask=reference_mask,
                )

                prompt_id = queue_workflow(workflow)
                history = wait_for_history(prompt_id)
                images = extract_images(history, settings) if (settings.reference_mask_id or settings.reference_instruction_edit) else extract_images(history)

                all_images.extend(images)

                if is_stop_requested():
                    break

    except KeyError as exc:
        raise RuntimeError(
            "В workflow Anima отсутствует ожидаемый узел: "
            f"{exc}"
        ) from exc

    return {
        "images": all_images,
        "actual_seed": actual_seed,
    }
