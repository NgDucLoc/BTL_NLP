from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def _run(cmd: list[str], cwd: Path) -> None:
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd), check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full 4-phase multimodal pipeline.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--llm", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--device", type=str, default="0")
    args = parser.parse_args()

    root = args.root.resolve()
    yolo_data = root / "new_appoarch" / "artifacts" / "phase1" / "yolo_data"
    data_yaml = yolo_data / "traffic_sign.yaml"
    phase1_dir = root / "new_appoarch" / "artifacts" / "phase1"
    detector_best = phase1_dir / "yolo_sign_detector" / "weights" / "best.pt"

    sample_json = root / "new_appoarch" / "artifacts" / "phase2" / "multimodal_train_samples.json"
    emb_dir = root / "new_appoarch" / "artifacts" / "phase2" / "clip_finetuned"
    vdb_dir = root / "new_appoarch" / "artifacts" / "phase3"
    pred_json = root / "new_appoarch" / "artifacts" / "phase4" / "public_predictions.json"

    _run(
        [
            "python",
            "new_appoarch/phase1_generate_detection_dataset.py",
            "--root",
            str(root),
            "--out-dir",
            str(yolo_data),
        ],
        root,
    )
    _run(
        [
            "python",
            "new_appoarch/phase1_train_yolo.py",
            "--data-yaml",
            str(data_yaml),
            "--out-dir",
            str(phase1_dir),
            "--device",
            args.device,
        ],
        root,
    )
    _run(
        [
            "python",
            "new_appoarch/phase2_build_multimodal_samples.py",
            "--root",
            str(root),
            "--detector",
            str(detector_best),
            "--out-json",
            str(sample_json),
        ],
        root,
    )
    _run(
        [
            "python",
            "new_appoarch/phase2_finetune_embedding.py",
            "--train-samples",
            str(sample_json),
            "--output-dir",
            str(emb_dir),
        ],
        root,
    )
    _run(
        [
            "python",
            "new_appoarch/phase3_build_vector_db.py",
            "--root",
            str(root),
            "--embedding-model-dir",
            str(emb_dir),
            "--out-dir",
            str(vdb_dir),
        ],
        root,
    )
    _run(
        [
            "python",
            "new_appoarch/phase4_rag_qa.py",
            "--root",
            str(root),
            "--split",
            "public",
            "--detector",
            str(detector_best),
            "--embedding-model-dir",
            str(emb_dir),
            "--vector-db-dir",
            str(vdb_dir),
            "--llm",
            args.llm,
            "--out-json",
            str(pred_json),
        ],
        root,
    )

    print("Pipeline done.")


if __name__ == "__main__":
    main()
