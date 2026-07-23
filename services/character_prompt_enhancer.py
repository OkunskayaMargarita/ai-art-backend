from collections import Counter

from services.danbooru_service import (
    enrich_character,
    fetch_character_posts,
    filter_single_character_posts,
)


GENERIC_TAG_REPLACEMENTS = {
    "animal_ears": {
        "rabbit_ears",
        "fox_ears",
        "cat_ears",
        "dog_ears",
        "wolf_ears",
    },
    "tail": {
        "rabbit_tail",
        "fox_tail",
        "cat_tail",
        "dog_tail",
        "wolf_tail",
    },
    "animal_girl": {
        "rabbit_girl",
        "fox_girl",
        "cat_girl",
        "dog_girl",
        "wolf_girl",
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
    cleaned_identity_tags = remove_generic_duplicates(
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

    cleaned_identity_tags = remove_generic_duplicates(
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