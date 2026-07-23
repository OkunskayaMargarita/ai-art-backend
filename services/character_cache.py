import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


CACHE_DIRECTORY = Path("cache/characters")
CACHE_LIFETIME_DAYS = 30


def normalize_cache_key(value: str) -> str:
    """
    Judy Hopps -> judy_hopps
    judy_hopps -> judy_hopps
    """
    normalized = value.strip().lower()
    normalized = re.sub(r"\s+", "_", normalized)
    normalized = re.sub(r"[^a-z0-9_()\-]+", "", normalized)

    return normalized or "unknown_character"


def get_cache_path(character_name: str) -> Path:
    key = normalize_cache_key(character_name)
    return CACHE_DIRECTORY / f"{key}.json"


def load_character_cache(
    character_name: str,
    max_age_days: int = CACHE_LIFETIME_DAYS,
) -> dict[str, Any] | None:
    cache_path = get_cache_path(character_name)

    if not cache_path.exists():
        return None

    try:
        cache_data = json.loads(
            cache_path.read_text(encoding="utf-8"),
        )
    except (OSError, json.JSONDecodeError):
        return None

    cached_at_text = cache_data.get("cached_at")

    if not cached_at_text:
        return None

    try:
        cached_at = datetime.fromisoformat(cached_at_text)
    except ValueError:
        return None

    if cached_at.tzinfo is None:
        cached_at = cached_at.replace(tzinfo=timezone.utc)

    expires_at = cached_at + timedelta(days=max_age_days)

    if datetime.now(timezone.utc) > expires_at:
        return None

    result = cache_data.get("result")

    return result if isinstance(result, dict) else None


def save_character_cache(
    character_name: str,
    result: dict[str, Any],
) -> Path:
    CACHE_DIRECTORY.mkdir(parents=True, exist_ok=True)

    cache_path = get_cache_path(character_name)
    temporary_path = cache_path.with_suffix(".tmp")

    cache_data = {
        "cached_at": datetime.now(timezone.utc).isoformat(),
        "input": character_name,
        "result": result,
    }

    temporary_path.write_text(
        json.dumps(
            cache_data,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    temporary_path.replace(cache_path)

    return cache_path


def delete_character_cache(character_name: str) -> bool:
    cache_path = get_cache_path(character_name)

    if not cache_path.exists():
        return False

    cache_path.unlink()
    return True


def clear_character_cache() -> int:
    if not CACHE_DIRECTORY.exists():
        return 0

    deleted_count = 0

    for cache_file in CACHE_DIRECTORY.glob("*.json"):
        cache_file.unlink()
        deleted_count += 1

    return deleted_count