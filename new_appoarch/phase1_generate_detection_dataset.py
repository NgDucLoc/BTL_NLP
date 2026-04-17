from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import numpy as np

from config import ensure_workspace_dirs, get_paths


def _to_yolo_xywh(box: tuple[int, int, int, int], w: int, h: int) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = box
    bw = x2 - x1
    bh = y2 - y1
    cx = x1 + bw / 2.0
    cy = y1 + bh / 2.0
    return cx / w, cy / h, bw / w, bh / h


def _paste_rgba(bg: np.ndarray, fg_rgba: np.ndarray, x: int, y: int) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    h, w = fg_rgba.shape[:2]
    h_bg, w_bg = bg.shape[:2]

    x = int(np.clip(x, 0, max(0, w_bg - w)))
    y = int(np.clip(y, 0, max(0, h_bg - h)))

    alpha = fg_rgba[:, :, 3:4] / 255.0
    fg_rgb = fg_rgba[:, :, :3]
    roi = bg[y : y + h, x : x + w].astype(np.float32)
    comp = fg_rgb.astype(np.float32) * alpha + roi * (1.0 - alpha)
    bg[y : y + h, x : x + w] = comp.astype(np.uint8)
    return bg, (x, y, x + w, y + h)


def _as_rgba(img_bgr: np.ndarray) -> np.ndarray:
    if img_bgr.shape[2] == 4:
        return img_bgr
    alpha = np.full((img_bgr.shape[0], img_bgr.shape[1], 1), 255, dtype=np.uint8)
    return np.concatenate([img_bgr, alpha], axis=-1)


def generate_synthetic_detection_set(
    root: Path,
    out_dir: Path,
    synth_per_real: int = 3,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> None:
    random.seed(seed)
    np.random.seed(seed)

    paths = get_paths(root)
    ensure_workspace_dirs(paths)

    real_images = sorted(paths.train_images.glob("*.jpg"))
    sign_images = sorted(list(paths.law_images.glob("*.png")) + list(paths.law_images.glob("*.jpg")))

    if not real_images:
        raise RuntimeError(f"No train images found at: {paths.train_images}")
    if not sign_images:
        raise RuntimeError(f"No sign images found at: {paths.law_images}")

    images_train = out_dir / "images" / "train"
    labels_train = out_dir / "labels" / "train"
    images_val = out_dir / "images" / "val"
    labels_val = out_dir / "labels" / "val"

    for p in [images_train, labels_train, images_val, labels_val]:
        p.mkdir(parents=True, exist_ok=True)

    all_samples = []
    for img_path in real_images:
        all_samples.append((img_path, False, 0))
        for i in range(synth_per_real):
            all_samples.append((img_path, True, i))

    random.shuffle(all_samples)
    n_val = int(len(all_samples) * val_ratio)
    val_set = set(range(n_val))

    for idx, (img_path, is_synth, copy_id) in enumerate(all_samples):
        bg = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if bg is None:
            continue

        h_bg, w_bg = bg.shape[:2]
        label_lines = []

        if is_synth:
            n_signs = random.randint(1, 3)
            for _ in range(n_signs):
                sign_path = random.choice(sign_images)
                sign = cv2.imread(str(sign_path), cv2.IMREAD_UNCHANGED)
                if sign is None:
                    continue
                sign = _as_rgba(sign)

                base = min(h_bg, w_bg)
                target_size = random.randint(max(32, base // 10), max(48, base // 4))
                scale = target_size / max(1, max(sign.shape[:2]))
                new_w = max(16, int(sign.shape[1] * scale))
                new_h = max(16, int(sign.shape[0] * scale))
                sign = cv2.resize(sign, (new_w, new_h), interpolation=cv2.INTER_AREA)

                x = random.randint(0, max(0, w_bg - new_w))
                y = random.randint(0, max(0, h_bg - new_h))

                bg, box = _paste_rgba(bg, sign, x, y)
                cx, cy, bw, bh = _to_yolo_xywh(box, w_bg, h_bg)
                label_lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        split = "val" if idx in val_set else "train"
        out_img_name = f"{img_path.stem}_s{copy_id}.jpg" if is_synth else f"{img_path.stem}_real.jpg"
        out_lbl_name = out_img_name.replace(".jpg", ".txt")

        out_img_path = (images_val if split == "val" else images_train) / out_img_name
        out_lbl_path = (labels_val if split == "val" else labels_train) / out_lbl_name

        cv2.imwrite(str(out_img_path), bg)
        out_lbl_path.write_text("\n".join(label_lines), encoding="utf-8")

    data_yaml = out_dir / "traffic_sign.yaml"
    data_yaml.write_text(
        "\n".join(
            [
                f"path: {out_dir.resolve()}",
                "train: images/train",
                "val: images/val",
                "names:",
                "  0: traffic_sign",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"Synthetic YOLO dataset generated at: {out_dir}")
    print(f"YOLO config: {data_yaml}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic detection data for YOLO phase 1.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("new_appoarch/artifacts/phase1/yolo_data"),
    )
    parser.add_argument("--synth-per-real", type=int, default=3)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = args.out_dir if args.out_dir.is_absolute() else (args.root / args.out_dir)
    generate_synthetic_detection_set(
        root=args.root,
        out_dir=out_dir,
        synth_per_real=args.synth_per_real,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
