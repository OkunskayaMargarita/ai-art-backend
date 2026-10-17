import copy
import json
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
) -> dict:
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

    sampler_inputs = workflow[SAMPLER_NODE_ID]["inputs"]

    sampler_inputs["seed"] = seed

    # Для Anima Turbo используем проверенные настройки.
    sampler_inputs["steps"] = settings.steps
    sampler_inputs["cfg"] = settings.cfg
    sampler_inputs["sampler_name"] = "er_sde"
    sampler_inputs["scheduler"] = "simple"
    sampler_inputs["denoise"] = 1

    workflow[SAVE_IMAGE_NODE_ID]["inputs"]["filename_prefix"] = (
        f"Anima_{uuid.uuid4().hex[:8]}"
    )

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


def download_output_image(image_data: dict) -> dict:
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

    with image_counter_lock:
        image_number = get_next_image_number()
        
        filename = f"{image_number}.png"
        filepath = os.path.join(OUTPUT_DIR, filename)

    # Режим xb запрещает перезаписывать существующий файл.
        with open(filepath, "xb") as output_file:
            output_file.write(response.content)

        save_image_counter(image_number)

        return {
            "filename": filename,
            "filepath": filepath,
            "image_url": f"/outputs/{filename}",
        }


def extract_images(prompt_history: dict) -> list[dict]:
    outputs = prompt_history.get("outputs", {})
    saved_images = []

    for node_output in outputs.values():
        for image_data in node_output.get("images", []):
            saved_images.append(
                download_output_image(image_data)
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
            for iteration_index in range(settings.batch_count):
                iteration_seed = actual_seed + iteration_index

                workflow = prepare_workflow(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    settings=settings,
                    seed=iteration_seed,
                )

                prompt_id = queue_workflow(workflow)
                history = wait_for_history(prompt_id)
                images = extract_images(history)

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