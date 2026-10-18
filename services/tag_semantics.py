"""Conservative reduction using a local, data-driven tag implication graph."""
import gzip
import json
import logging
import re
from functools import lru_cache
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parents[1] / "presets" / "tag_semantics.json.gz"
logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def load_tag_semantics() -> tuple[dict[str, set[str]], dict[str, str]]:
    try:
        with gzip.open(DATA_PATH, "rt", encoding="utf-8") as file:
            data = json.load(file)
        graph = {}
        for specific, general in data["implications"]:
            graph.setdefault(specific, set()).add(general)
        return graph, data["aliases"]
    except (OSError, ValueError, KeyError, TypeError):
        logger.warning("Tag semantics data unavailable; keeping tags without semantic reduction.")
        return {}, {}


def remove_implied_tags(tags: list[str]) -> list[str]:
    """Keep maximal facts, not substrings; leave weights and prose untouched.

    A -> B means A entails B, not that any phrase containing B entails A.
    Thus an object and an action involving its category can both be necessary.
    """
    graph, aliases = load_tag_semantics()

    def canonical(tag):
        key = "_".join(tag.casefold().split())
        if not re.fullmatch(r"[a-z0-9][a-z0-9_'\-]*", key):
            return None
        visited = set()
        while key in aliases and key not in visited:
            visited.add(key)
            key = aliases[key]
        return key

    keys = [canonical(tag) for tag in tags]
    closures = {}
    for key in set(keys) - {None}:
        reached = set()
        pending = list(graph.get(key, ()))
        while pending:
            parent = pending.pop()
            if parent not in reached:
                reached.add(parent)
                pending.extend(graph.get(parent, ()))
        closures[key] = reached

    result = []
    seen = set()
    for index, (tag, key) in enumerate(zip(tags, keys)):
        if key is None:
            result.append(tag)
            continue
        if key in seen:
            continue
        # In a malformed cyclic graph retain the first equivalent fact.
        redundant = any(other is not None and other != key
                        and key in closures[other]
                        and (other not in closures[key] or other_index < index)
                        for other_index, other in enumerate(keys))
        if not redundant:
            result.append(tag)
            seen.add(key)
    return result
