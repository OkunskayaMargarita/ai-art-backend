"""User supplied references; independent of Danbooru character lookup."""
import csv
import io
import json
import re
import threading
import uuid
from functools import lru_cache
from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError

REFERENCE_DIR = Path(__file__).resolve().parents[1] / "cache" / "image_references"
MAX_BYTES = 15 * 1024 * 1024
MAX_PIXELS = 24_000_000
_analysis_lock = threading.Lock()


@lru_cache(maxsize=1)
def reference_tag_filter() -> dict[str, set[str]]:
    path = Path(__file__).resolve().parents[1] / "presets" / "reference_tag_filter.json"
    with path.open(encoding="utf-8") as file:
        policy = json.load(file)
    return {name: set(values) for name, values in policy.items()}


def is_excluded_reference_overlay(tag: str) -> bool:
    """Reject lettering/attribution families, including known aliases/implications."""
    from services.danbooru_service import META_TAGS
    from services.tag_semantics import load_tag_semantics

    policy = reference_tag_filter()
    excluded = policy["excluded_overlay_tags"] | (META_TAGS - {"comic", "manga"})
    words = policy["excluded_overlay_words"]
    key = "_".join(tag.casefold().replace("_", " ").split())
    if re.search(r"https?://|www\.|(?:^|\s)@\w|^(?:artist|author|source):", tag.strip(), re.IGNORECASE):
        return True
    graph, aliases = load_tag_semantics()
    pending = [key]
    visited = set()
    while pending:
        name = pending.pop()
        if name in visited:
            continue
        visited.add(name)
        if name in excluded or words.intersection(name.split("_")):
            return True
        pending.extend(graph.get(name, ()))
        if name in aliases:
            pending.append(aliases[name])
    return False


def is_low_priority_reference_tag(tag: str) -> bool:
    """Omit incidental anatomy from automatic descriptions, not user prompts.

    Match whole tags/simple visibility qualifiers, never body-part substrings:
    poses such as hands_on_hips and features such as a shoulder tattoo survive.
    """
    policy = reference_tag_filter()
    anatomy, modifiers = policy["low_priority_anatomy"], policy["visibility_modifiers"]
    key = "_".join(tag.casefold().replace("_", " ").split())
    if key in anatomy:
        return True
    return any(key in {f"{modifier}_{part}", f"{part}_{modifier}"}
               for part in anatomy for modifier in modifiers)


def select_reference_character(prompt: str, characters: list[dict]) -> dict | None:
    candidates = sorted(characters, key=lambda item: item["score"], reverse=True)
    tags = {"_".join(tag.casefold().split()) for tag in prompt.split(",")}
    multiple = (any(re.match(r"(?:[2-9]\d*(?:girls|boys)|multiple_)", tag) for tag in tags)
                or {"1girl", "1boy"}.issubset(tags))
    if (candidates and candidates[0]["score"] >= 0.85 and not multiple
            and (len(candidates) == 1 or candidates[0]["score"] - candidates[1]["score"] >= 0.2)):
        return candidates[0]
    return None


def supplemental_character_candidates(path: Path) -> list[dict]:
    """Anthropomorphic character vocabulary; retain WD's general tag output."""
    import numpy as np
    import onnxruntime as ort
    from huggingface_hub import hf_hub_download

    repo = "toynya/Z3D-E621-Convnext"
    revision = "39b081cafc120253513e6e9c21283892c6994e62"
    cache = REFERENCE_DIR.parent / "tagger_models"
    model = hf_hub_download(repo, "model.onnx", revision=revision, cache_dir=cache)
    labels = hf_hub_download(repo, "tags-selected.csv", revision=revision, cache_dir=cache)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.enable_cpu_mem_arena = False
    options.enable_mem_pattern = False
    session = ort.InferenceSession(model, sess_options=options, providers=["CPUExecutionProvider"])
    input_node = session.get_inputs()[0]
    size = input_node.shape[1]
    with Image.open(path) as image:
        # White square, bicubic, BGR NHWC, raw [0,255] values.
        image = image.convert("RGB")
        side = max(image.size)
        square = Image.new("RGB", (side, side), "white")
        square.paste(image, ((side - image.width) // 2, (side - image.height) // 2))
        square = square.resize((size, size), Image.Resampling.BICUBIC)
        array = np.asarray(square, dtype=np.float32)[:, :, ::-1].copy()[None]
    scores = session.run(None, {input_node.name: array})[0][0]
    with open(labels, encoding="utf-8") as file:
        mapping = list(csv.DictReader(file))
    return [{"name": value["name"], "score": float(score)}
            for value, score in zip(mapping, scores)
            if value["category"] == "4" and score >= 0.5]


def reference_detail_field(tag: str) -> str | None:
    """Classify actions and garment details before their objects/materials.

    Token boundaries and implication ancestors cover families of tags, while
    preserving the distinction between e.g. an outfit and holding an outfit.
    """
    from services.danbooru_service import CLOTHING_WORDS
    from services.tag_semantics import load_tag_semantics

    tag = "_".join(tag.casefold().split())
    actions = {
        "holding", "carrying", "reaching", "pointing", "touching", "grabbing",
        "hugging", "kissing", "eating", "drinking", "sitting", "standing",
        "kneeling", "lying", "leaning", "walking", "running", "jumping",
        "stretching", "waving", "crouching", "squatting", "applying", "adjusting",
        "brushing", "combing", "tying", "braiding", "painting", "removing",
    }
    garment_parts = CLOTHING_WORDS | {
        "sleeve", "sleeves", "cuff", "cuffs", "neckline", "hem", "hems",
        "button", "buttons", "zipper", "zippers", "hood", "pocket", "pockets",
        "apron", "vest", "cardigan", "stockings", "socks", "pantyhose",
        "tights", "leggings", "scarf", "belt", "suspenders", "lapels",
    }

    def direct(name):
        if any(name == action or name.startswith(action + "_") for action in actions):
            return "pose"
        if name in garment_parts or any(
                name.startswith(part + "_") or name.endswith("_" + part)
                for part in garment_parts):
            return "clothes"
        return None

    field = direct(tag)
    if field:
        return field
    graph, aliases = load_tag_semantics()
    visited = set()
    pending = {tag}
    # Prefer the nearest classified ancestor; do not guess when equally near
    # ancestors disagree (an implication can describe more than one fact).
    while pending:
        next_level = set()
        fields = set()
        for name in pending - visited:
            visited.add(name)
            field = direct(name)
            if field:
                fields.add(field)
            next_level.update(graph.get(name, ()))
            if name in aliases:
                next_level.add(aliases[name])
        if fields:
            return fields.pop() if len(fields) == 1 else None
        pending = next_level - visited
    return None


def is_reference_identity_trait(tag: str) -> bool:
    """Appearance/design families, not expressions, gestures, or clothing."""
    from services.danbooru_service import classify_character_tag
    from services.tag_semantics import load_tag_semantics

    key = "_".join(tag.casefold().replace("_", " ").split())
    if key in {"closed_eyes", "half-closed_eyes", "half_closed_eyes", "one_eye_closed",
               "wide_eyes", "rolling_eyes", "teary_eyes", "squinting"}:
        return False
    if reference_detail_field(key) is not None:
        return False
    policy = reference_tag_filter()
    roots = policy["identity_trait_roots"] | policy["identity_trait_tags"]
    words = policy["identity_trait_words"]
    graph, aliases = load_tag_semantics()
    visited = set()
    pending = [key]
    while pending:
        name = pending.pop()
        if name in visited:
            continue
        visited.add(name)
        if (name in roots or words.intersection(name.split("_"))
                or classify_character_tag(name) in {"species", "appearance"}):
            return True
        pending.extend(graph.get(name, ()))
        if name in aliases:
            pending.append(aliases[name])
    return False


def reference_character_label(name: str, sources: list[str]) -> str:
    """One recognized name and at most one source, including parenthetical ones."""
    display_name = " ".join(name.replace("_", " ").split())
    if re.search(r"\([^()]+\)\s*$", display_name):
        return display_name
    name_key = display_name.casefold()
    seen = {name_key}
    for source in sources:
        display_source = " ".join(source.replace("_", " ").split())
        source_key = display_source.casefold()
        if source_key and source_key not in seen:
            return f"{display_name}, {display_source}"
        seen.add(source_key)
    return display_name


def reference_prompt_fields(prompt: str, characters: list[dict] | None = None) -> dict:
    """Route observed tags to existing form fields without inventing details."""
    from services.danbooru_service import classify_character_tag, CLOTHING_WORDS
    from services.character_prompt_enhancer import (
        remove_generic_duplicates, remove_furry_skin_duplicates, remove_redundant_gender_tags,
        enhance_character_prompt,
    )
    from services.prompt_builder import join_prompt_parts

    tags = [tag.strip().replace(" ", "_").lower()
            for tag in join_prompt_parts([prompt], semantic=True).split(",")
            if tag.strip() and not is_low_priority_reference_tag(tag)
            and not is_excluded_reference_overlay(tag)]
    tags = remove_redundant_gender_tags(remove_furry_skin_duplicates(remove_generic_duplicates(tags)))
    character_name = ""
    identity = []
    selected_character = select_reference_character(prompt, characters or [])
    if selected_character:
        candidate = selected_character["name"]
        # Visual recognition must not depend on Danbooru being reachable.
        # If enrichment fails, keep the recognized name AND observed traits.
        character_name = reference_character_label(candidate, [])
        try:
            known = enhance_character_prompt(candidate, post_limit=100, maximum_identity_tags=20)
        except RuntimeError:
            known = {}
        normalize = lambda tag: "_".join(tag.casefold().split())
        if known.get("found") and normalize(known.get("resolved_tag", "")) == normalize(candidate):
            identity = [tag for tag in known.get("cleaned_identity_tags", [])
                        if not is_low_priority_reference_tag(tag)
                        and not is_excluded_reference_overlay(tag)][:20]
            copyrights = [tag for tag in known.get("copyright_tags", [])
                          if not is_excluded_reference_overlay(tag)][:3]
            character_name = reference_character_label(candidate, copyrights)
            covered = {normalize(tag) for tag in [candidate, *identity, *copyrights]}
            # Only omit traits which will actually be restored at generation.
            specific = set(remove_generic_duplicates([*tags, *identity]))
            tags = [tag for tag in tags if tag in specific and (
                normalize(tag) not in covered or reference_detail_field(tag) in {"clothes", "pose"})]
        else:
            tags = [tag for tag in tags if normalize(tag) != normalize(candidate)]
    # The model already orders tags by confidence. Retain a compact description.
    tags = tags[:40]
    fields = {name: [] for name in ("character", "clothes", "pose", "background", "extra_tags")}
    pose_tags = {"standing", "sitting", "kneeling", "lying", "lying_on_back", "lying_on_side",
                 "lying_on_stomach", "squatting", "crouching", "walking", "running", "jumping",
                 "leaning_forward", "leaning_back", "arms_up", "arms_behind_back", "crossed_arms",
                 "crossed_legs", "spread_legs", "looking_at_viewer", "looking_back", "looking_away",
                 "from_above", "from_below", "from_side", "from_behind", "full_body", "upper_body",
                 "portrait", "close-up", "dutch_angle", "outstretched_arms", "hand_on_hip"}
    background_tags = {"indoors", "outdoors", "sky", "cloud", "clouds", "sun", "moon", "stars",
                       "forest", "tree", "trees", "grass", "flower", "flowers", "beach", "ocean",
                       "sea", "water", "mountain", "mountains", "snow", "rain", "city", "street",
                       "bedroom", "bed", "room", "window", "chair", "table", "couch", "night",
                       "day", "sunset", "sunrise", "building", "buildings", "scenery"}
    character_tags = {"1girl", "1boy", "2girls", "2boys", "multiple_girls", "multiple_boys",
                      "solo", "no_humans", "smile", "blush", "open_mouth", "closed_eyes"}
    for tag in tags:
        category = classify_character_tag(tag)
        detail_field = reference_detail_field(tag)
        if detail_field == "pose" or tag in pose_tags:
            field = "pose"
        elif detail_field == "clothes":
            field = "clothes"
        elif tag in background_tags or tag.endswith("_background"):
            field = "background"
        elif category == "clothes" or any(tag.endswith("_" + word) for word in CLOTHING_WORDS):
            field = "clothes"
        elif category in {"species", "appearance", "body"} or tag in character_tags:
            field = "character"
        else:
            field = "extra_tags"
        fields[field].append(tag.replace("_", " "))
    if character_name:
        # A limited Danbooru identity list is not exhaustive. Keep the remaining
        # observed appearance in the same snapshot, not in visible extra tags.
        observed_traits = []
        other_tags = []
        for tag in [*fields["character"], *fields["extra_tags"]]:
            (observed_traits if is_reference_identity_trait(tag) else other_tags).append(tag)
        identity = join_prompt_parts([*identity, *observed_traits], semantic=True).split(", ") if identity or observed_traits else []
        fields["extra_tags"] = other_tags
        fields["character"] = [character_name]
    return {**{name: ", ".join(tags) for name, tags in fields.items()},
            "reference_character_name": character_name,
            "reference_character_traits": [tag.replace("_", " ") for tag in identity],
            "style_name": "none", "pose_mode": "manual" if fields["pose"] else "free"}


def reference_path(reference_id: str) -> Path:
    if not re.fullmatch(r"[a-f0-9]{32}", reference_id):
        raise ValueError("Загрузите изображение-референс.")
    path = REFERENCE_DIR / f"{reference_id}.png"
    if not path.is_file():
        raise ValueError("Референс не найден. Загрузите изображение снова.")
    return path


def reference_mask_path(reference_id: str, mask_id: str) -> Path:
    reference_path(reference_id)
    if not re.fullmatch(r"[a-f0-9]{32}", mask_id):
        raise ValueError("Выделите область изменения на картинке.")
    path = REFERENCE_DIR / f"{reference_id}_{mask_id}_mask.png"
    if not path.is_file():
        raise ValueError("Маска не найдена. Выделите область снова.")
    return path


def save_reference_mask(reference_id: str, content: bytes) -> str:
    original = reference_path(reference_id)
    if not content or len(content) > MAX_BYTES:
        raise ValueError("Некорректный размер маски.")
    try:
        with Image.open(original) as source, Image.open(io.BytesIO(content)) as image:
            if image.size != source.size or image.format != "PNG":
                raise ValueError("Маска должна быть PNG того же размера, что и референс.")
            mask = image.convert("L")
            if mask.getbbox() is None:
                raise ValueError("Маска пуста. Закрасьте область, которую нужно изменить.")
    except (OSError, UnidentifiedImageError, Image.DecompressionBombError) as exc:
        raise ValueError("Не удалось прочитать маску.") from exc
    mask_id = uuid.uuid4().hex
    mask.save(REFERENCE_DIR / f"{reference_id}_{mask_id}_mask.png")
    return mask_id


def composite_reference_edit(reference_id: str, mask_id: str, generated: bytes) -> tuple[bytes, tuple[int, int]]:
    """Restore original RGB pixels wherever the user mask is zero."""
    with Image.open(reference_path(reference_id)) as image:
        original = image.convert("RGB")
    with Image.open(reference_mask_path(reference_id, mask_id)) as image:
        mask = image.convert("L")
    with Image.open(io.BytesIO(generated)) as image:
        replacement = image.convert("RGB").resize(original.size, Image.Resampling.LANCZOS)
    result = Image.composite(replacement, original, mask)
    output = io.BytesIO()
    result.save(output, format="PNG")
    return output.getvalue(), original.size


def save_reference(content: bytes) -> dict:
    if not content or len(content) > MAX_BYTES:
        raise ValueError("Размер изображения должен быть не больше 15 МБ.")
    try:
        with Image.open(io.BytesIO(content)) as source:
            if source.format not in {"PNG", "JPEG", "WEBP"}:
                raise ValueError("Поддерживаются PNG, JPEG и WebP.")
            if source.width * source.height > MAX_PIXELS:
                raise ValueError("Изображение должно содержать не больше 24 млн пикселей.")
            image = ImageOps.exif_transpose(source).convert("RGBA")
            background = Image.new("RGB", image.size, "white")
            background.paste(image, mask=image.getchannel("A"))
            metadata_prompt = ""
            # Only extract plain text, never execute imported graphs.
            try:
                graph = json.loads(source.info.get("prompt", "{}"))
                if isinstance(graph, dict):
                    positive_ids = set()
                    for node in graph.values():
                        if isinstance(node, dict) and node.get("class_type") in {"KSampler", "KSamplerAdvanced"}:
                            link = node.get("inputs", {}).get("positive")
                            if isinstance(link, list) and link and isinstance(link[0], str):
                                positive_ids.add(link[0])
                    for node_id, node in graph.items():
                        if isinstance(node, dict) and node.get("class_type") == "CLIPTextEncode":
                            title = str(node.get("_meta", {}).get("title", "")).lower()
                            if node_id in positive_ids or "positive" in title:
                                text = node.get("inputs", {}).get("text")
                                if isinstance(text, str) and text.strip():
                                    metadata_prompt = text[:16000]
                                    break
            except (ValueError, TypeError, AttributeError):
                pass
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ValueError("Не удалось прочитать изображение.") from exc
    REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    reference_id = uuid.uuid4().hex
    background.save(REFERENCE_DIR / f"{reference_id}.png")
    return {"reference_id": reference_id, "width": background.width,
            "height": background.height, "metadata_prompt": metadata_prompt}


def analyze_reference(reference_id: str, *, return_details: bool = False) -> str | dict:
    path = reference_path(reference_id)
    # CPU inference avoids competing with ComfyUI for video memory. Release the
    # session after each request so idle analysis does not retain model memory.
    with _analysis_lock:
        try:
            import numpy as np
            import onnxruntime as ort
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise RuntimeError("Установите зависимости backend из requirements.txt для распознавания изображения.") from exc
        try:
            repo = "SmilingWolf/wd-vit-tagger-v3"
            revision = "7f6b584d0bd3f55c4531f14ba3d4761b2bccdc0f"
            cache_dir = REFERENCE_DIR.parent / "tagger_models"
            model = hf_hub_download(repo, "model.onnx", revision=revision, cache_dir=cache_dir)
            labels = hf_hub_download(repo, "selected_tags.csv", revision=revision, cache_dir=cache_dir)
            session = ort.InferenceSession(model, providers=["CPUExecutionProvider"])
            input_node = session.get_inputs()[0]
            size = input_node.shape[1]
            with Image.open(path) as image:
                square = ImageOps.pad(image.convert("RGB"), (size, size), color="white")
                array = np.asarray(square, dtype=np.float32)[:, :, ::-1].copy()[None]
            scores = session.run(None, {input_node.name: array})[0][0]
            with open(labels, encoding="utf-8") as file:
                tags = list(csv.DictReader(file))
            selected = [(float(score), tag["name"].replace("_", " "))
                        for tag, score in zip(tags, scores)
                        if ((tag["category"] == "0" and score >= 0.35)
                            or (not return_details and tag["category"] == "4" and score >= 0.85))
                        and not is_excluded_reference_overlay(tag["name"])]
            selected.sort(reverse=True)
            characters = [{"name": tag["name"], "score": float(score)}
                          for tag, score in zip(tags, scores)
                          if tag["category"] == "4" and score >= 0.5]
            prompt = ", ".join(tag for _, tag in selected[:100])
            if not prompt:
                raise RuntimeError("Не удалось уверенно распознать теги. Введите описание вручную.")
            warning = None
            if return_details and not select_reference_character(prompt, characters):
                # Avoid keeping both ONNX models resident at once.
                del session
                import gc
                gc.collect()
                try:
                    extra = supplemental_character_candidates(path)
                    # Scores from separately trained models are not calibrated
                    # against each other. Evaluate each model's own margin.
                    if select_reference_character(prompt, extra):
                        characters = extra
                except Exception:
                    import logging
                    logging.getLogger(__name__).exception("Supplemental character recognition failed")
                    warning = "Теги получены, но дополнительное распознавание персонажей недоступно. Проверьте загрузку модели и свободную память."
            return {"prompt": prompt, "characters": characters, "character_warning": warning} if return_details else prompt
        except RuntimeError:
            raise
        except Exception as exc:
            raise RuntimeError("Не удалось выполнить анализ. При первом запуске нужен интернет для загрузки WD Tagger; проверьте соединение и свободную память.") from exc
