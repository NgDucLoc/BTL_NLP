from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class Paths:
    root: Path
    train_json: Path
    public_json: Path
    private_task1_json: Path
    private_task2_json: Path
    law_json: Path
    train_images: Path
    public_images: Path
    private_images: Path
    law_images: Path
    workspace: Path


def get_paths(root: str | Path) -> Paths:
    root = Path(root).resolve()
    dataset = root / "dataset"
    workspace = root / "new_appoarch"

    return Paths(
        root=root,
        train_json=dataset / "train data" / "vlsp_2025_train.json",
        public_json=dataset / "public_test data" / "vlsp_2025_public_test.json",
        private_task1_json=dataset
        / "private_test data (post submission)"
        / "submission_task1_no_labels.json",
        private_task2_json=dataset
        / "private_test data (post submission)"
        / "submission_task2_no_labels.json",
        law_json=dataset / "law_db" / "vlsp2025_law.json",
        train_images=dataset / "train data" / "train_images",
        public_images=dataset / "public_test data" / "public_test_images",
        private_images=dataset / "private_test data (post submission)" / "private_test_images",
        law_images=dataset / "law_db" / "images.fld",
        workspace=workspace,
    )


def ensure_workspace_dirs(paths: Paths) -> None:
    (paths.workspace / "artifacts").mkdir(parents=True, exist_ok=True)
    (paths.workspace / "artifacts" / "phase1").mkdir(parents=True, exist_ok=True)
    (paths.workspace / "artifacts" / "phase2").mkdir(parents=True, exist_ok=True)
    (paths.workspace / "artifacts" / "phase3").mkdir(parents=True, exist_ok=True)
    (paths.workspace / "artifacts" / "phase4").mkdir(parents=True, exist_ok=True)
