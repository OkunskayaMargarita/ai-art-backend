"""Local visual-medium recognition for reference edits, independent of character tags."""
import json
import threading
from functools import lru_cache
from pathlib import Path

from PIL import Image, ImageOps

from services.image_reference_service import reference_path

MODEL_REPO = "Xenova/clip-vit-base-patch32"
MODEL_REVISION = "d15189d7028b43f1d3e65039190477f6af591c2a"
CACHE_DIR = Path(__file__).resolve().parents[1] / "cache" / "style_models"
_style_lock = threading.Lock()


def _model_file(name):
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    arguments = dict(repo_id=MODEL_REPO, filename=name, revision=MODEL_REVISION, cache_dir=CACHE_DIR)
    try:
        return hf_hub_download(**arguments, local_files_only=True)
    except LocalEntryNotFoundError:
        return hf_hub_download(**arguments)


def _session(name):
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.enable_cpu_mem_arena = False
    options.enable_mem_pattern = False
    return ort.InferenceSession(_model_file(name), sess_options=options, providers=["CPUExecutionProvider"])


@lru_cache(maxsize=1)
def _style_embeddings():
    import numpy as np
    from tokenizers import Tokenizer

    path = Path(__file__).resolve().parents[1] / "presets" / "reference_rendering_styles.json"
    styles = json.loads(path.read_text(encoding="utf-8"))
    tokenizer = Tokenizer.from_file(_model_file("tokenizer.json"))
    tokenizer.enable_padding(pad_id=49407, pad_token="<|endoftext|>")
    tokenizer.enable_truncation(max_length=77)
    labels = [label for style in styles for label in style["labels"]]
    encoded = tokenizer.encode_batch(labels)
    session = _session("onnx/text_model_quantized.onnx")
    vectors = session.run(None, {"input_ids": np.array([item.ids for item in encoded], dtype=np.int64)})[0]
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    # Average the caption ensemble, not its softmax probabilities.
    means, offset = [], 0
    for style in styles:
        count = len(style["labels"])
        means.append(vectors[offset:offset + count].mean(axis=0))
        offset += count
    means = np.array(means)
    means /= np.linalg.norm(means, axis=1, keepdims=True)
    return styles, means


def select_rendering_style(styles, similarities):
    """Uncertain images keep visual conditioning without an invented medium label."""
    ranked = sorted(range(len(styles)), key=lambda index: float(similarities[index]), reverse=True)
    first, second = ranked[:2]
    if similarities[first] < 0.22 or similarities[first] - similarities[second] < 0.005:
        return None
    return styles[first]


@lru_cache(maxsize=64)
def reference_rendering_style(reference_id):
    import numpy as np

    # References are immutable UUID files; cache only small results/embeddings,
    # never ONNX sessions alongside ComfyUI's models.
    with _style_lock:
        try:
            path = reference_path(reference_id)
            styles, embeddings = _style_embeddings()
            with Image.open(path) as image:
                image = ImageOps.fit(image.convert("RGB"), (224, 224), method=Image.Resampling.BICUBIC)
                pixels = np.asarray(image, dtype=np.float32) / 255
            pixels = (pixels - [0.48145466, 0.4578275, 0.40821073]) / [0.26862954, 0.26130258, 0.27577711]
            session = _session("onnx/vision_model_quantized.onnx")
            vector = session.run(None, {"pixel_values": pixels.transpose(2, 0, 1)[None].astype(np.float32)})[0][0]
            vector /= np.linalg.norm(vector)
            return select_rendering_style(styles, embeddings @ vector)
        except Exception as exc:
            raise RuntimeError(
                "Не удалось проанализировать технику референса. Проверьте зависимости backend "
                "и загрузку локальной CLIP-модели (около 156 МБ при первом запуске)."
            ) from exc
