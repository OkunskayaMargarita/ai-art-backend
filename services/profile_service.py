import json
from pathlib import Path

from models.profile import ProfileData


PROFILES_FILE = Path("presets/profiles.json")


def ensure_profiles_file() -> None:
    PROFILES_FILE.parent.mkdir(parents=True, exist_ok=True)

    if not PROFILES_FILE.exists():
        PROFILES_FILE.write_text(
            json.dumps({}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def load_profiles() -> dict:
    ensure_profiles_file()

    try:
        content = PROFILES_FILE.read_text(encoding="utf-8")
        data = json.loads(content)

        if not isinstance(data, dict):
            return {}

        return data
    except (json.JSONDecodeError, OSError):
        return {}


def save_profiles(profiles: dict) -> None:
    ensure_profiles_file()

    temporary_file = PROFILES_FILE.with_suffix(".tmp")

    temporary_file.write_text(
        json.dumps(profiles, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    temporary_file.replace(PROFILES_FILE)


def get_all_profiles() -> dict:
    return load_profiles()


def get_profile(name: str) -> dict | None:
    profiles = load_profiles()
    return profiles.get(name)


def create_profile(name: str, settings: ProfileData) -> dict:
    profiles = load_profiles()

    if name in profiles:
        raise ValueError("Профиль с таким именем уже существует.")

    profiles[name] = settings.model_dump()
    save_profiles(profiles)

    return profiles[name]


def update_profile(
    current_name: str,
    settings: ProfileData,
) -> dict:
    profiles = load_profiles()

    if current_name not in profiles:
        raise KeyError("Профиль не найден.")

    profiles[current_name] = settings.model_dump()
    save_profiles(profiles)

    return profiles[current_name]


def delete_profile(name: str) -> None:
    profiles = load_profiles()

    if name not in profiles:
        raise KeyError("Профиль не найден.")

    del profiles[name]
    save_profiles(profiles)