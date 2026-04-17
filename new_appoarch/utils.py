from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def normalize_article_text(text: str) -> str:
    text = re.sub(r"<<IMAGE:\s*(.*?)\s*/IMAGE>>", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"<<TABLE:\s*(.*?)\s*/TABLE>>", " ", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def build_article_map(law_db: list[dict]) -> dict[tuple[str, str], str]:
    article_map: dict[tuple[str, str], str] = {}
    for law in law_db:
        law_id = str(law.get("id", ""))
        for article in law.get("articles", []):
            article_id = str(article.get("id", ""))
            title = str(article.get("title", "")).strip()
            text = normalize_article_text(str(article.get("text", "")))
            merged = f"{title}. {text}".strip(" .")
            article_map[(law_id, article_id)] = merged
    return article_map


def image_path_from_id(image_root: Path, image_id: str) -> Path:
    # Dataset currently uses .jpg for train/public/private image sets.
    return image_root / f"{image_id}.jpg"
