"""Lightweight taxonomy / findings index helpers.

The setup taxonomy is the shared vocabulary both the replay agent (tags) and the
research agent (findings) draw from. A tag is valid if it is a known slug or
starts with `propose:` (a freely-coined new pattern).
"""
from __future__ import annotations

import json

from .paths import LESSONS_INDEX


def load_lessons() -> list[dict]:
    if not LESSONS_INDEX.exists():
        return []
    return json.loads(LESSONS_INDEX.read_text())


def valid_slugs() -> set[str]:
    return {L["slug"] for L in load_lessons()}


def check_slug(slug: str, slugs: set[str], what: str) -> None:
    if slug in slugs or slug.startswith("propose:"):
        return
    raise SystemExit(
        f"{what} {slug!r} is not a known setup slug and is not prefixed 'propose:'. "
        f"Known slugs: {sorted(slugs)}"
    )


def taxonomy_text() -> str:
    lessons = load_lessons()
    if not lessons:
        return "(no lessons defined yet — use propose:<snake_slug> for any pattern)"
    return "\n".join(f"- {L['slug']}: {L.get('heuristic', L.get('title', ''))}" for L in lessons)
