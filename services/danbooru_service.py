import re
from collections import Counter
from services.character_cache import (
    load_character_cache,
    save_character_cache,
)

import requests


DANBOORU_API_URL = "https://danbooru.donmai.us"

REQUEST_HEADERS = {
    "User-Agent": "AIArt18/0.1",
}


# Теги, которые не описывают постоянную внешность персонажа.
# Их не нужно автоматически добавлять в prompt.
BLOCKED_TAGS = {
    # Количество персонажей
    "1girl",
    "1boy",
    "solo",
    "multiple_girls",
    "multiple_boys",

    # Композиция и камера
    "looking_at_viewer",
    "looking_back",
    "upper_body",
    "full_body",
    "portrait",
    "close-up",
    "cowboy_shot",
    "from_side",
    "from_above",
    "from_below",

    # Фон и освещение
    "simple_background",
    "white_background",
    "black_background",
    "outdoors",
    "indoors",
    "day",
    "night",
    "shadow",
    "light_rays",

    # Эмоции
    "smile",
    "grin",
    "blush",
    "open_mouth",
    "closed_mouth",
    "serious",
    "tears",

    # Действия и позы
    "standing",
    "sitting",
    "lying",
    "kneeling",
    "walking",
    "running",
    "arms_up",
    "hands_on_hips",

    # Качество и служебные теги
    "highres",
    "absurdres",
    "official_art",
    "traditional_media",
    "translation_request",
    "commentary_request",
    "web_address",
    "signature",
    "watermark",
    "text",
}

META_TAGS = {
    "english_text",
    "japanese_text",
    "chinese_text",
    "korean_text",
    "text",
    "watermark",
    "signature",
    "artist_name",
    "username",
    "web_address",
    "speech_bubble",
    "thought_bubble",
    "comic",
    "manga",
    "translation_request",
    "commentary_request",
}

BODY_TRANSIENT_TAGS = {
    "breasts",
    "large_breasts",
    "medium_breasts",
    "small_breasts",
    "cleavage",
    "navel",
    "midriff",
    "bare_shoulders",
    "bare_legs",
    "thighs",
}

CLOTHING_WORDS = {
    "shirt",
    "t-shirt",
    "dress",
    "skirt",
    "pants",
    "shorts",
    "jacket",
    "coat",
    "hoodie",
    "sweater",
    "uniform",
    "police_uniform",
    "school_uniform",
    "maid",
    "maid_outfit",
    "swimsuit",
    "bikini",
    "underwear",
    "panties",
    "bra",
    "boots",
    "shoes",
    "gloves",
    "hat",
    "cap",
    "tie",
    "necktie",
    "collar",
}

SPECIES_WORDS = {
    "furry",
    "anthro",
    "kemono",
    "animal_girl",
    "animal_boy",
    "rabbit_girl",
    "rabbit_boy",
    "fox_girl",
    "fox_boy",
    "cat_girl",
    "cat_boy",
    "dog_girl",
    "dog_boy",
    "furry_female",
    "furry_male",
}

SPECIES_SUFFIXES = (
    "_ears",
    "_tail",
    "_horns",
    "_wings",
    "_girl",
    "_boy",
)

APPEARANCE_SUFFIXES = (
    "_eyes",
    "_hair",
    "_fur",
    "_skin",
    "_scales",
)

APPEARANCE_WORDS = {
    "long_hair",
    "short_hair",
    "very_long_hair",
    "twintails",
    "ponytail",
    "braid",
    "freckles",
    "fangs",
    "facial_mark",
    "heterochromia",
}

GENERIC_SPECIES_TAGS = {
    "animal_ears",
    "tail",
    "paws",
    "claws",
    "fangs",
}


def normalize_character_query(value: str) -> str:

    normalized = value.strip().lower()
    normalized = re.sub(r"\s+", "_", normalized)
    return normalized


def search_character_tags(
    character_name: str,
    limit: int = 10,
) -> list[dict]:
    query = normalize_character_query(character_name)

    if not query:
        return []

    try:
        response = requests.get(
            f"{DANBOORU_API_URL}/tags.json",
            params={
                "search[name_matches]": f"*{query}*",
                "search[category]": 4,
                "search[hide_empty]": "yes",
                "search[order]": "count",
                "limit": limit,
            },
            headers=REQUEST_HEADERS,
            timeout=20,
        )
    except requests.ConnectionError as exc:
        raise RuntimeError(
            "Не удалось подключиться к Danbooru."
        ) from exc
    except requests.Timeout as exc:
        raise RuntimeError(
            "Danbooru слишком долго не отвечает."
        ) from exc

    if not response.ok:
        raise RuntimeError(
            f"Danbooru вернул HTTP {response.status_code}: "
            f"{response.text}"
        )

    tags = response.json()

    return [
        {
            "name": tag.get("name", ""),
            "post_count": tag.get("post_count", 0),
            "category": tag.get("category"),
        }
        for tag in tags
    ]


def resolve_character_tag(character_name: str) -> dict:
    normalized = normalize_character_query(character_name)
    results = search_character_tags(character_name)

    if not results:
        return {
            "found": False,
            "input": character_name,
            "resolved_tag": normalized,
            "candidates": [],
        }

    exact_match = next(
        (
            tag
            for tag in results
            if tag["name"] == normalized
        ),
        None,
    )

    best_match = exact_match or results[0]

    return {
        "found": True,
        "input": character_name,
        "resolved_tag": best_match["name"],
        "post_count": best_match["post_count"],
        "candidates": results,
    }


def fetch_character_posts(
    character_tag: str,
    limit: int = 100,
) -> list[dict]:
    try:
        response = requests.get(
            f"{DANBOORU_API_URL}/posts.json",
            params={
                "tags": character_tag,
                "limit": limit,
            },
            headers=REQUEST_HEADERS,
            timeout=30,
        )
    except requests.ConnectionError as exc:
        raise RuntimeError(
            "Не удалось получить изображения персонажа с Danbooru."
        ) from exc
    except requests.Timeout as exc:
        raise RuntimeError(
            "Danbooru слишком долго загружает данные персонажа."
        ) from exc

    if not response.ok:
        raise RuntimeError(
            f"Danbooru вернул HTTP {response.status_code}: "
            f"{response.text}"
        )

    posts = response.json()

    if not isinstance(posts, list):
        return []

    return posts

def filter_single_character_posts(
    posts: list[dict],
    character_tag: str,
) -> list[dict]:
    """
    Оставляет только посты, где искомый персонаж является
    единственным character-тегом.

    Это предотвращает попадание признаков других персонажей
    в автоматически собранное описание.
    """
    filtered_posts = []

    for post in posts:
        character_tags_string = post.get(
            "tag_string_character",
            "",
        )

        character_tags = character_tags_string.split()

        if character_tags == [character_tag]:
            filtered_posts.append(post)

    return filtered_posts

def extract_frequent_character_tags(
    posts: list[dict],
    minimum_frequency: float = 0.35,
    maximum_tags: int = 12,
) -> list[dict]:
    """
    Выделяет теги, которые повторяются на значительной части постов.

    minimum_frequency=0.35 означает:
    тег должен присутствовать минимум в 35% найденных публикаций.
    """
    if not posts:
        return []

    tag_counter = Counter()
    valid_post_count = 0

    for post in posts:
        tag_string = post.get("tag_string_general", "")

        if not tag_string:
            continue

        valid_post_count += 1
        post_tags = set(tag_string.split())

        for tag in post_tags:
            if tag in BLOCKED_TAGS:
                continue

            if len(tag) < 2:
                continue

            tag_counter[tag] += 1

    if valid_post_count == 0:
        return []

    minimum_count = max(
        2,
        round(valid_post_count * minimum_frequency),
    )

    frequent_tags = []

    for tag, count in tag_counter.most_common():
        if count < minimum_count:
            continue

        frequent_tags.append(
            {
                "tag": tag,
                "count": count,
                "frequency": round(
                    count / valid_post_count,
                    3,
                ),
            }
        )

        if len(frequent_tags) >= maximum_tags:
            break

    return frequent_tags


def classify_character_tag(tag: str) -> str:
    """
    Возвращает категорию тега.

    Возможные категории:
    - species
    - appearance
    - clothes
    - body
    - meta
    - other
    """
    if tag in META_TAGS:
        return "meta"

    if tag in BODY_TRANSIENT_TAGS:
        return "body"

    if tag in CLOTHING_WORDS:
        return "clothes"

    if tag in SPECIES_WORDS:
        return "species"

    if tag in GENERIC_SPECIES_TAGS:
        return "species"

    if tag.endswith(SPECIES_SUFFIXES):
        return "species"

    if tag in APPEARANCE_WORDS:
        return "appearance"

    if tag.endswith(APPEARANCE_SUFFIXES):
        return "appearance"

    return "other"


def categorize_character_tags(
    frequent_tags: list[dict],
) -> dict[str, list[dict]]:
    groups = {
        "species": [],
        "appearance": [],
        "clothes": [],
        "body": [],
        "meta": [],
        "other": [],
    }

    for item in frequent_tags:
        category = classify_character_tag(item["tag"])

        groups[category].append(item)

    return groups

def enrich_character(
    character_name: str,
    post_limit: int = 100,
    maximum_tags: int = 20,
) -> dict:
    
    cached_result = load_character_cache(character_name)

    if cached_result is not None:
        return {
            **cached_result,
            "from_cache": True,
        }
    
    resolved = resolve_character_tag(character_name)

    if not resolved["found"]:
        result = {
            **resolved,
            "original_sample_size": 0,
            "sample_size": 0,
            "tag_groups": {
                "species": [],
                "appearance": [],
                "clothes": [],
                "body": [],
                "meta": [],
                "other": [],
            },
            "identity_tags": [],
            "enriched_prompt": resolved["resolved_tag"],
            "from_cache": False,
        }

        save_character_cache(
            character_name=character_name,
            result=result,
        )

        return result

    character_tag = resolved["resolved_tag"]

    posts = fetch_character_posts(
        character_tag=character_tag,
        limit=post_limit,
    )

    single_character_posts = filter_single_character_posts(
        posts=posts,
        character_tag=character_tag,
    )

    frequent_tags = extract_frequent_character_tags(
        posts=single_character_posts,
        maximum_tags=maximum_tags,
    )

    tag_groups = categorize_character_tags(frequent_tags)

    # В автоматический prompt входят только устойчивые признаки:
    # вид/раса и внешность.
    identity_items = [
        *tag_groups["species"],
        *tag_groups["appearance"],
    ]

    identity_tags = [
        item["tag"]
        for item in identity_items
    ]

    prompt_parts = [
        character_tag,
        *identity_tags,
    ]

    result = {
        **resolved,
        "original_sample_size": len(posts),
        "sample_size": len(single_character_posts),
        "tag_groups": tag_groups,
        "identity_tags": identity_tags,
        "enriched_prompt": ", ".join(prompt_parts),
        "from_cache": False,
    }

    save_character_cache(
        character_name=character_name,
        result=result,
    )

    return result