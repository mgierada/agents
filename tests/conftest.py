import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from PIL import Image


def write_image(path: Path, image: np.ndarray) -> Path:
    """Write an image, creating parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(path)
    return path


def write_label(path: Path, content: str) -> Path:
    """Write a YOLO label file, creating parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return path


@pytest.fixture
def sharp_image() -> np.ndarray:
    """High-frequency noise image: very high Laplacian variance, mid brightness."""
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, size=(64, 64), dtype=np.uint8)


@pytest.fixture
def flat_image() -> np.ndarray:
    """Uniform gray image: zero Laplacian variance."""
    return np.full((64, 64), 128, dtype=np.uint8)


GALLERY_ROWS: list[tuple[str, str, int, float, bool, int, str]] = [
    # crop_id, brand, short side px, knn_agreement, suspect, cluster, knn_pred
    ("vidA-0_0", "acme", 80, 1.0, False, 0, "acme"),
    ("vidA-0_1", "acme", 12, 0.1, True, 0, "globex"),
    ("vidB-3_0", "acme", 60, 0.9, False, 1, "acme"),
    ("vidC-1_0", "globex", 70, 1.0, False, 0, "globex"),
    ("vidC-1_1", "globex", 64, 0.2, True, -1, "acme"),
    ("vidD-0_0", "initech", 90, 0.95, False, 0, "initech"),
]
GALLERY_LABELS: dict[str, str] = {
    "vidA-0": "0 0.5 0.5 0.2 0.2 0\n1 0.2 0.2 0.05 0.05 0 0.9 [run_name=8baece93abc]\n",
    "vidB-3": "0 0.5 0.5 0.2 0.2 0\n",
    "vidC-1": "1 0.5 0.5 0.2 0.2 0\n1 0.95 0.5 0.2 0.2 0 0.8 [run_name=older_model]\n",
    "vidD-0": "2 0.5 0.5 0.3 0.3 0\n",
}
GALLERY_SUMMARY: dict[str, object] = {
    "best": "siglip2",
    "siglip2": {"top_confusions": ["acme -> globex: 5", "globex -> acme: 3"]},
}


@pytest.fixture
def gallery_dir(tmp_path: Path) -> Path:
    """A tiny logo-audit dir: analysis.parquet, summary.json, crops/ and labels."""
    root: Path = tmp_path / "gallery"
    rng = np.random.default_rng(0)
    rows: list[dict[str, object]] = []
    for crop_id, brand, side, agreement, suspect, cluster, knn_pred in GALLERY_ROWS:
        write_image(
            root / "crops" / f"{crop_id}.jpg",
            rng.integers(0, 256, (side, side * 2, 3), dtype=np.uint8),
        )
        rows.append(
            {
                "crop_id": crop_id,
                "crop_path": f"/remote/{crop_id}.jpg",
                "source_image": f"/remote/{crop_id.rsplit('_', 1)[0]}.jpg",
                "video_group": crop_id.split("-")[0],
                "brand": brand,
                "box_w": side * 2,
                "box_h": side,
                "area_fraction": 0.01,
                "knn_pred": knn_pred,
                "knn_vote_share": agreement,
                "knn_agreement": agreement,
                "knn_top_sim": 0.9,
                "suspect": suspect,
                "cluster": cluster,
            }
        )
    pl.DataFrame(rows).write_parquet(root / "analysis.parquet")
    (root / "summary.json").write_text(json.dumps(GALLERY_SUMMARY))
    for stem, text in GALLERY_LABELS.items():
        write_label(root / "labels_with_pseudo" / f"{stem}.txt", text)
    return root


@pytest.fixture
def target_list(tmp_path: Path) -> Path:
    """Target list file: acme, globex, initech, toyota."""
    path: Path = tmp_path / "targets.txt"
    path.write_text("acme\nglobex\ninitech\ntoyota\n")
    return path
