import json
from pathlib import Path


STYLES_FILE = Path("presets/styles.json")


def load_styles() -> dict:
    if not STYLES_FILE.exists():
        return {}

    try:
        content = STYLES_FILE.read_text(encoding="utf-8")
        data = json.loads(content)

        if not isinstance(data, dict):
            return {}

        return data
    except (json.JSONDecodeError, OSError):
        return {}


def get_all_styles() -> dict:
    return load_styles()


def get_style(style_id: str) -> dict | None:
    styles = load_styles()
    return styles.get(style_id)


def get_style_prompt(style_id: str) -> str:
    style = get_style(style_id)

    if not style:
        return ""

    prompt = style.get("prompt", "")

    if not isinstance(prompt, str):
        return ""

    return prompt.strip()
    
def get_style_negative_remove(style_id: str) -> list[str]:
    style = get_style(style_id)

    if not style:
        return []

    negative_remove = style.get("negative_remove", [])

    if not isinstance(negative_remove, list):
        return []

    return [
        item.strip()
        for item in negative_remove
        if isinstance(item, str) and item.strip()
    ]
    
def get_style_positive_remove(style_id: str) -> list[str]:
    style = get_style(style_id)

    if not style:
        return []

    positive_remove = style.get("positive_remove", [])

    if not isinstance(positive_remove, list):
        return []

    return [
        item.strip()
        for item in positive_remove
        if isinstance(item, str) and item.strip()
    ]
    
def get_style_negative_prompt(style_id: str) -> str:
    style = get_style(style_id)

    if not style:
        return ""

    prompt = style.get("negative_prompt", "")

    if not isinstance(prompt, str):
        return ""

    return prompt.strip()