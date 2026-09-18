from collections import Counter

from services.danbooru_service import (
    enrich_character,
    fetch_character_posts,
    filter_single_character_posts,
)


GENERIC_TAG_REPLACEMENTS = {
    # Уши
    "animal_ears": {
        "rabbit_ears",
        "fox_ears",
        "cat_ears",
        "dog_ears",
        "wolf_ears",
    },

    # Хвост
    "tail": {
        "rabbit_tail",
        "fox_tail",
        "cat_tail",
        "dog_tail",
        "wolf_tail",
        "striped_tail",
    },

    # Если внешний вид хвоста уже описан конкретнее,
    # общий вид хвоста можно убрать.
    "cat_tail": {
        "striped_tail",
    },

    # Общий тип персонажа
    "animal_girl": {
        "rabbit_girl",
        "fox_girl",
        "cat_girl",
        "dog_girl",
        "wolf_girl",
    },

    # Общий цвет кожи ничего не добавляет,
    # если конкретный цвет уже известен.
    "colored_skin": {
        "yellow_skin",
        "blue_skin",
        "green_skin",
        "red_skin",
        "purple_skin",
        "pink_skin",
        "orange_skin",
        "black_skin",
        "white_skin",
        "grey_skin",
        "gray_skin",
    },
}


def remove_generic_duplicates(
    tags: list[str],
) -> list[str]:
    """
    Удаляет общие теги, если уже есть более точные.

    rabbit_ears + animal_ears -> rabbit_ears
    rabbit_tail + tail -> rabbit_tail
    """
    tag_set = set(tags)

    for generic_tag, specific_tags in (
        GENERIC_TAG_REPLACEMENTS.items()
    ):
        if (
            generic_tag in tag_set
            and tag_set & specific_tags
        ):
            tag_set.remove(generic_tag)

    return [
        tag
        for tag in tags
        if tag in tag_set
    ]
    
def remove_furry_skin_duplicates(
    tags: list[str],
) -> list[str]:
    """
    Для furry-персонажей предпочитает конкретное описание
    цвета шерсти аналогичному описанию цвета кожи.

    furry + yellow_skin + yellow_fur
    -> furry + yellow_fur
    """
    tag_set = set(tags)

    if "furry" not in tag_set:
        return tags

    fur_colors = {
        tag.removesuffix("_fur")
        for tag in tag_set
        if tag.endswith("_fur")
    }

    return [
        tag
        for tag in tags
        if not (
            tag.endswith("_skin")
            and tag.removesuffix("_skin") in fur_colors
        )
    ]
    
def remove_redundant_gender_tags(
    tags: list[str],
) -> list[str]:
    """
    Удаляет общий furry, если уже присутствует
    более конкретный furry_female или furry_male.

    furry + furry_female -> furry_female
    furry + furry_male -> furry_male
    """
    tag_set = set(tags)

    if (
        "furry" in tag_set
        and (
            "furry_female" in tag_set
            or "furry_male" in tag_set
        )
    ):
        return [
            tag
            for tag in tags
            if tag != "furry"
        ]

    return tags


def extract_copyright_tags(
    posts: list[dict],
    maximum_tags: int = 3,
) -> list[str]:
    """
    Получает наиболее частые copyright/franchise-теги
    из одиночных публикаций персонажа.
    """
    counter = Counter()
    valid_posts = 0

    for post in posts:
        copyright_string = post.get(
            "tag_string_copyright",
            "",
        )

        if not copyright_string:
            continue

        valid_posts += 1

        for tag in set(copyright_string.split()):
            counter[tag] += 1

    if valid_posts == 0:
        return []

    result = []

    for tag, count in counter.most_common():
        frequency = count / valid_posts

        # Copyright-тег должен встречаться минимум
        # в половине отобранных публикаций.
        if frequency < 0.5:
            continue

        result.append(tag)

        if len(result) >= maximum_tags:
            break

    return result


def build_character_description(
    resolved_tag: str,
    identity_tags: list[str],
    copyright_tags: list[str],
) -> list[str]:
    """
    Собирает итоговое описание идентичности персонажа.
    """
    cleaned_identity_tags = clean_identity_tags(
        identity_tags
    )

    prompt_parts = [
        resolved_tag,
        *copyright_tags,
        *cleaned_identity_tags,
    ]

    # Удаляем повторы, сохраняя порядок.
    unique_parts = []
    seen = set()

    for part in prompt_parts:
        normalized = part.strip()

        if not normalized or normalized in seen:
            continue

        seen.add(normalized)
        unique_parts.append(normalized)

    return unique_parts
    
def clean_identity_tags(
    tags: list[str],
) -> list[str]:
    """
    Выполняет всю семантическую очистку identity-тегов.
    """
    cleaned_tags = remove_generic_duplicates(tags)
    cleaned_tags = remove_furry_skin_duplicates(cleaned_tags)
    cleaned_tags = remove_redundant_gender_tags(cleaned_tags)

    return cleaned_tags


def enhance_character_prompt(
    character_name: str,
    post_limit: int = 100,
    maximum_identity_tags: int = 20,
) -> dict:
    """
    Строит уточнённое описание персонажа для prompt.
    """
    character_data = enrich_character(
        character_name=character_name,
        post_limit=post_limit,
        maximum_tags=maximum_identity_tags,
    )

    resolved_tag = character_data.get(
        "resolved_tag",
        character_name,
    )

    if not character_data.get("found"):
        return {
            "found": False,
            "input": character_name,
            "resolved_tag": resolved_tag,
            "copyright_tags": [],
            "identity_tags": [],
            "cleaned_identity_tags": [],
            "prompt_parts": [resolved_tag],
            "enhanced_prompt": resolved_tag,
            "character_data": character_data,
        }

    posts = fetch_character_posts(
        character_tag=resolved_tag,
        limit=post_limit,
    )

    single_character_posts = filter_single_character_posts(
        posts=posts,
        character_tag=resolved_tag,
    )

    copyright_tags = extract_copyright_tags(
        posts=single_character_posts,
    )

    identity_tags = character_data.get(
        "identity_tags",
        [],
    )

    cleaned_identity_tags = clean_identity_tags(
        identity_tags
    )

    prompt_parts = build_character_description(
        resolved_tag=resolved_tag,
        identity_tags=identity_tags,
        copyright_tags=copyright_tags,
    )

    return {
        "found": True,
        "input": character_name,
        "resolved_tag": resolved_tag,
        "copyright_tags": copyright_tags,
        "identity_tags": identity_tags,
        "cleaned_identity_tags": cleaned_identity_tags,
        "prompt_parts": prompt_parts,
        "enhanced_prompt": ", ".join(prompt_parts),
        "character_data": character_data,
    }