from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def train_yolo(
    data_yaml: Path,
    out_dir: Path,
    model_name: str = "yolov8n.pt",
    epochs: int = 30,
    imgsz: int = 640,
    batch: int = 16,
    device: str = "0",
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(model_name)
    result = model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=str(out_dir),
        name="yolo_sign_detector",
    )

    best = Path(result.save_dir) / "weights" / "best.pt"
    print(f"Training finished. Best checkpoint: {best}")
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description="Train YOLO detector for traffic sign localization.")
    parser.add_argument("--data-yaml", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("new_appoarch/artifacts/phase1"))
    parser.add_argument("--model", type=str, default="yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", type=str, default="0")
    args = parser.parse_args()

    train_yolo(
        data_yaml=args.data_yaml,
        out_dir=args.out_dir,
        model_name=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
    )


if __name__ == "__main__":
    main()
