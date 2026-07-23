import base64
from dataclasses import dataclass
from typing import Any

import requests

from services.danbooru_service import (
    fetch_character_posts,
    filter_single_character_posts,
    resolve_character_tag,
)


REQUEST_HEADERS = {
    "User-Agent": "AIArt18/0.1",
}

MAX_REFERENCE_SIZE_BYTES = 15 * 1024 * 1024

EXCLUDED_REFERENCE_TAGS = {
    "english_text",
    "japanese_text",
    "chinese_text",
    "korean_text",
    "text",
    "watermark",
    "signature",
    "speech_bubble",
    "thought_bubble",
    "comic",
    "manga",
    "multiple_views",
    "character_sheet",
}


@dataclass
class ReferenceCandidate:
    post_id: int
    image_url: str
    preview_url: str
    width: int
    height: int
    score: int
    category: str
    tags: set[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "post_id": self.post_id,
            "image_url": self.image_url,
            "preview_url": self.preview_url,
            "width": self.width,
            "height": self.height,
            "score": self.score,
            "category": self.category,
        }


def get_post_tags(post: dict) -> set[str]:
    return set(
        post.get("tag_string_general", "").split()
    )


def determine_reference_category(tags: set[str]) -> str:
    if {
        "profile",
        "from_side",
        "side_view",
    } & tags:
        return "side"

    if {
        "full_body",
        "standing",
    } & tags:
        return "full_body"

    if {
        "portrait",
        "close-up",
        "upper_body",
        "face",
    } & tags:
        return "face"

    return "general"


def is_suitable_reference(post: dict) -> bool:
    image_url = (
        post.get("large_file_url")
        or post.get("file_url")
    )

    if not image_url:
        return False

    width = int(post.get("image_width") or 0)
    height = int(post.get("image_height") or 0)

    if width < 512 or height < 512:
        return False

    tags = get_post_tags(post)

    if tags & EXCLUDED_REFERENCE_TAGS:
        return False

    return True


def post_to_candidate(post: dict) -> ReferenceCandidate:
    tags = get_post_tags(post)

    image_url = (
        post.get("large_file_url")
        or post.get("file_url")
    )

    preview_url = (
        post.get("preview_file_url")
        or post.get("large_file_url")
        or post.get("file_url")
    )

    return ReferenceCandidate(
        post_id=int(post.get("id") or 0),
        image_url=image_url,
        preview_url=preview_url,
        width=int(post.get("image_width") or 0),
        height=int(post.get("image_height") or 0),
        score=int(post.get("score") or 0),
        category=determine_reference_category(tags),
        tags=tags,
    )


def select_diverse_references(
    candidates: list[ReferenceCandidate],
    maximum_references: int = 3,
) -> list[ReferenceCandidate]:
    sorted_candidates = sorted(
        candidates,
        key=lambda item: (
            item.score,
            item.width * item.height,
        ),
        reverse=True,
    )

    selected: list[ReferenceCandidate] = []
    used_categories: set[str] = set()

    preferred_categories = [
        "face",
        "full_body",
        "side",
        "general",
    ]

    for category in preferred_categories:
        candidate = next(
            (
                item
                for item in sorted_candidates
                if item.category == category
                and item.post_id
                not in {
                    selected_item.post_id
                    for selected_item in selected
                }
            ),
            None,
        )

        if candidate is None:
            continue

        selected.append(candidate)
        used_categories.add(category)

        if len(selected) >= maximum_references:
            return selected

    for candidate in sorted_candidates:
        if candidate.post_id in {
            item.post_id
            for item in selected
        }:
            continue

        selected.append(candidate)

        if len(selected) >= maximum_references:
            break

    return selected


def find_character_references(
    character_name: str,
    post_limit: int = 100,
    maximum_references: int = 3,
) -> dict:
    resolved = resolve_character_tag(character_name)

    if not resolved.get("found"):
        return {
            **resolved,
            "references": [],
        }

    character_tag = resolved["resolved_tag"]

    posts = fetch_character_posts(
        character_tag=character_tag,
        limit=post_limit,
    )

    single_character_posts = filter_single_character_posts(
        posts=posts,
        character_tag=character_tag,
    )

    suitable_posts = [
        post
        for post in single_character_posts
        if is_suitable_reference(post)
    ]

    candidates = [
        post_to_candidate(post)
        for post in suitable_posts
    ]

    selected = select_diverse_references(
        candidates=candidates,
        maximum_references=maximum_references,
    )

    return {
        **resolved,
        "downloaded_posts": len(posts),
        "single_character_posts": len(
            single_character_posts
        ),
        "suitable_posts": len(suitable_posts),
        "references": [
            item.to_dict()
            for item in selected
        ],
    }


def download_reference_as_base64(
    image_url: str,
) -> str:
    """
    Загружает изображение только в оперативную память.

    Файл не сохраняется на диск.
    """
    try:
        response = requests.get(
            image_url,
            headers=REQUEST_HEADERS,
            timeout=30,
            stream=True,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(
            "Не удалось загрузить изображение-референс."
        ) from exc

    content_length = int(
        response.headers.get("content-length", 0)
        or 0
    )

    if content_length > MAX_REFERENCE_SIZE_BYTES:
        raise RuntimeError(
            "Изображение-референс слишком большое."
        )

    image_bytes = response.content

    if len(image_bytes) > MAX_REFERENCE_SIZE_BYTES:
        raise RuntimeError(
            "Изображение-референс слишком большое."
        )

    encoded = base64.b64encode(
        image_bytes
    ).decode("utf-8")

    return encoded