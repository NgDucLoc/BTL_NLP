from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
from ultralytics import YOLO

from config import get_paths
from utils import build_article_map, image_path_from_id, load_json, save_json


def _detect_first_box(detector: YOLO, image_path: Path, conf: float) -> tuple[int, int, int, int] | None:
    results = detector.predict(source=str(image_path), conf=conf, verbose=False)
    if not results:
        return None
    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return None

    b = boxes[0].xyxy[0].tolist()
    x1, y1, x2, y2 = map(int, b)
    return x1, y1, x2, y2


def build_samples(root: Path, detector_path: Path, out_json: Path, conf: float = 0.25) -> None:
    random.seed(42)
    paths = get_paths(root)

    train_data = load_json(paths.train_json)
    law_db = load_json(paths.law_json)
    article_map = build_article_map(law_db)

    all_articles = list(article_map.items())
    detector = YOLO(str(detector_path))

    out_json.parent.mkdir(parents=True, exist_ok=True)
    crops_dir = out_json.parent / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    samples = []
    for row in train_data:
        image_id = row["image_id"]
        qid = row["id"]
        question = row["question"]
        image_path = image_path_from_id(paths.train_images, image_id)
        if not image_path.exists():
            continue

        box = _detect_first_box(detector, image_path, conf=conf)
        if box is None:
            continue

        img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if img is None:
            continue

        h, w = img.shape[:2]
        x1, y1, x2, y2 = box
        x1 = max(0, min(w - 1, x1))
        y1 = max(0, min(h - 1, y1))
        x2 = max(x1 + 1, min(w, x2))
        y2 = max(y1 + 1, min(h, y2))
        crop = img[y1:y2, x1:x2]

        crop_name = f"{qid}_{image_id}.jpg"
        crop_path = crops_dir / crop_name
        cv2.imwrite(str(crop_path), crop)

        positives = []
        for ra in row.get("relevant_articles", []):
            key = (str(ra.get("law_id", "")), str(ra.get("article_id", "")))
            txt = article_map.get(key)
            if txt:
                positives.append({"law_id": key[0], "article_id": key[1], "text": txt})

        if not positives:
            continue

        pos = random.choice(positives)
        neg_key, neg_text = random.choice(all_articles)
        if neg_key == (pos["law_id"], pos["article_id"]):
            neg_key, neg_text = all_articles[(all_articles.index((neg_key, neg_text)) + 1) % len(all_articles)]

        samples.append(
            {
                "id": qid,
                "image_id": image_id,
                "question": question,
                "crop_path": str(crop_path.resolve()),
                "positive": pos,
                "negative": {"law_id": neg_key[0], "article_id": neg_key[1], "text": neg_text},
            }
        )

    save_json(out_json, samples)
    print(f"Saved {len(samples)} multimodal samples to {out_json}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build multimodal finetuning samples from detector outputs.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--detector", type=Path, required=True)
    parser.add_argument(
        "--out-json",
        type=Path,
        default=Path("new_appoarch/artifacts/phase2/multimodal_train_samples.json"),
    )
    parser.add_argument("--conf", type=float, default=0.25)
    args = parser.parse_args()

    out_json = args.out_json if args.out_json.is_absolute() else (args.root / args.out_json)
    detector = args.detector if args.detector.is_absolute() else (args.root / args.detector)

    build_samples(args.root, detector, out_json, conf=args.conf)


if __name__ == "__main__":
    main()
