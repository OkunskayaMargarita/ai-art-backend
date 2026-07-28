import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path

from models.pose import PoseLayout, PosePreset, WeightedPoseVariant

logger = logging.getLogger(__name__)
BASE_DIR = Path(__file__).resolve().parent.parent
POSES_DIR = BASE_DIR / "presets" / "poses"


@dataclass
class PoseLibrarySnapshot:
    presets: list[PosePreset]
    warnings: list[str]


def _normalise_text(value: str) -> str:
    return " ".join(value.lower().strip().split())


def _contains_context(context: str, phrase: str) -> bool:
    context = _normalise_text(context)
    phrase = _normalise_text(phrase)
    return bool(context and phrase and phrase in context)


def load_pose_library() -> PoseLibrarySnapshot:
    presets: list[PosePreset] = []
    warnings: list[str] = []
    known_ids: set[str] = set()

    if not POSES_DIR.exists():
        return PoseLibrarySnapshot([], [f"Папка библиотеки поз не найдена: {POSES_DIR}"])

    files = sorted(POSES_DIR.rglob("*.json"))
    if not files:
        return PoseLibrarySnapshot([], [f"В папке {POSES_DIR} нет JSON-пресетов."])

    for file_path in files:
        try:
            preset = PosePreset.model_validate(
                json.loads(file_path.read_text(encoding="utf-8"))
            )
            if preset.id in known_ids:
                warnings.append(
                    f"Повторяющийся ID «{preset.id}» в файле {file_path.name}. Файл пропущен."
                )
                continue
            if not preset.layouts:
                warnings.append(f"В файле {file_path.name} нет layouts. Файл пропущен.")
                continue
            known_ids.add(preset.id)
            presets.append(preset)
        except Exception as exc:
            warnings.append(f"{file_path.name}: {exc}")

    for warning in warnings:
        logger.warning("Pose library: %s", warning)

    return PoseLibrarySnapshot(presets, warnings)


def list_pose_presets() -> list[PosePreset]:
    return [p for p in load_pose_library().presets if p.enabled]


def get_pose_library_warnings() -> list[str]:
    return load_pose_library().warnings


def get_pose_preset(pose_id: str) -> PosePreset | None:
    pose_id = pose_id.strip()
    if not pose_id:
        return None
    return next((p for p in list_pose_presets() if p.id == pose_id), None)


def detect_orientation(width: int, height: int) -> str:
    if width > height:
        return "landscape"
    if height > width:
        return "portrait"
    return "square"


def context_score(preset: PosePreset, context: str, orientation: str) -> float:
    context = _normalise_text(context)
    score = preset.selection_weight

    score += 1.0 if ("any" in preset.orientation or orientation in preset.orientation) else -2.0

    for phrase in preset.incompatible_contexts:
        if _contains_context(context, phrase):
            return -1000.0

    matches = sum(
        1 for phrase in preset.compatible_contexts
        if phrase != "any" and _contains_context(context, phrase)
    )
    score += matches * 4.0

    has_any = "any" in preset.compatible_contexts
    if has_any:
        score += 0.5

    if preset.required_context_any:
        required_match = any(
            _contains_context(context, phrase)
            for phrase in preset.required_context_any
        )
        if required_match:
            score += 3.0
        elif context:
            score -= 4.0
        else:
            score -= 2.0

    if not context:
        if has_any:
            score += 2.0
        elif not preset.required_context_any:
            score += 1.0
    elif matches == 0 and not has_any:
        score -= 1.0

    return score


def weighted_variant_choice(variants: list[WeightedPoseVariant]) -> str | None:
    if not variants:
        return None
    selected = random.choices(
        population=variants,
        weights=[v.weight for v in variants],
        k=1,
    )[0]
    return selected.text.strip()


def choose_layout(preset: PosePreset) -> PoseLayout:
    layouts = [layout for layout in preset.layouts if layout.required or layout.variants]
    if not layouts:
        raise RuntimeError(f"У пресета «{preset.name}» нет доступных layouts.")
    return random.choices(layouts, weights=[layout.weight for layout in layouts], k=1)[0]


def build_pose_prompt(
    preset: PosePreset,
    layout: PoseLayout,
) -> tuple[str, dict[str, str]]:
    prompt_parts = [part.strip() for part in layout.required if part.strip()]
    selected_variants: dict[str, str] = {}
    chance = (
        layout.optional_group_chance
        if layout.optional_group_chance is not None
        else preset.optional_group_chance
    )

    groups = list(layout.variants.items())
    random.shuffle(groups)

    for group_name, variants in groups:
        if random.random() > chance:
            continue
        selected = weighted_variant_choice(variants)
        if not selected:
            continue
        selected_variants[group_name] = selected
        prompt_parts.append(selected)

    return ", ".join(dict.fromkeys(prompt_parts)), selected_variants


def choose_automatic_pose(context: str, width: int, height: int) -> PosePreset:
    presets = list_pose_presets()
    if not presets:
        raise RuntimeError("В библиотеке нет доступных пресетов поз.")

    orientation = detect_orientation(width, height)
    scored = [(p, context_score(p, context, orientation)) for p in presets]
    usable = [(p, s) for p, s in scored if s > -100]

    if not usable:
        usable = [(p, p.selection_weight) for p in presets]

    usable.sort(key=lambda item: item[1], reverse=True)
    top = usable[: min(5, len(usable))]
    minimum = min(score for _, score in top)
    weights = [max((score - minimum) + p.selection_weight, 0.1) for p, score in top]

    return random.choices([p for p, _ in top], weights=weights, k=1)[0]


def resolve_pose(
    pose_mode: str,
    manual_pose: str,
    pose_preset_id: str,
    environment: str,
    width: int,
    height: int,
) -> dict:
    pose_mode = pose_mode.strip().lower()
    manual_pose = manual_pose.strip()
    pose_preset_id = pose_preset_id.strip()
    
    # Пользователь сознательно не задаёт позу.
    if pose_mode == "free":
        return {
            "source": "free",
            "preset_id": None,
            "preset_name": None,
            "layout_id": None,
            "layout_name": None,
            "prompt": "",
            "selected_variants": {},
        }

    if pose_mode == "manual":
        if not manual_pose:
            raise RuntimeError(
                "Выбран ручной режим позы, "
                "но текстовое описание не заполнено."
            )

        return {
            "source": "manual",
            "preset_id": None,
            "preset_name": None,
            "layout_id": None,
            "layout_name": None,
            "prompt": manual_pose,
            "selected_variants": {},
        }

    if pose_mode == "preset":
        if not pose_preset_id:
            raise RuntimeError(
                "Выбран режим пресета, "
                "но пресет позы не указан."
            )

        preset = get_pose_preset(pose_preset_id)

        if preset is None:
            raise RuntimeError(
                f"Пресет позы «{pose_preset_id}» не найден."
            )

        source = "preset"

    elif pose_mode == "automatic":
        preset = choose_automatic_pose(
            context=environment,
            width=width,
            height=height,
        )
        source = "automatic"

    else:
        raise RuntimeError(
            f"Неизвестный режим позы: {pose_mode}"
        )

    layout = choose_layout(preset)

    pose_prompt, variants = build_pose_prompt(
        preset=preset,
        layout=layout,
    )

    return {
        "source": source,
        "preset_id": preset.id,
        "preset_name": preset.name,
        "layout_id": layout.id,
        "layout_name": layout.name or layout.id,
        "prompt": pose_prompt,
        "selected_variants": variants,
    }
