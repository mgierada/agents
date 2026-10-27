from pathlib import Path

import numpy as np

from cleanup.config import Settings
from cleanup.dataset import find_label, metadata_files, walk_files
from cleanup.graph import build_graph
from tests.conftest import write_image, write_label

IMAGE_EXTS: tuple[str, ...] = (".png", ".jpg")
DARWIN_JSON = '{"version": "2.0", "annotations": []}'


def make_darwin_batch(root: Path, good: np.ndarray, bad: np.ndarray) -> Path:
    """Darwin-style batch: images/, annotations/*.json, classes/*.txt."""
    write_image(root / "images/good.png", good)
    write_image(root / "images/bad.png", bad)
    write_label(root / "annotations/good.json", DARWIN_JSON)
    write_label(root / "annotations/bad.json", DARWIN_JSON)
    write_label(root / "classes/classes_bounding_box.txt", "Logo\n")
    return root


def test_find_label_prefers_yolo_then_darwin(tmp_path: Path) -> None:
    yolo: Path = write_label(tmp_path / "a/labels/train/x.txt", "")
    darwin: Path = write_label(tmp_path / "b/annotations/y.json", "{}")
    assert find_label(tmp_path / "a/images/train/x.jpg", tmp_path) == yolo
    assert find_label(tmp_path / "b/images/y.jpg", tmp_path) == darwin
    assert find_label(tmp_path / "c/images/z.jpg", tmp_path) is None
    assert find_label(tmp_path / "loose.jpg", tmp_path) is None


def test_walk_files_follows_symlinks_without_looping(tmp_path: Path) -> None:
    external: Path = tmp_path / "external"
    write_label(external / "f.txt", "")
    (external / "loop").symlink_to(external)
    data: Path = tmp_path / "data"
    data.mkdir()
    (data / "linked").symlink_to(external)

    assert list(walk_files(data)) == [data / "linked/f.txt"]


def test_metadata_files_skips_images_and_annotations(tmp_path: Path) -> None:
    write_label(tmp_path / "data.yaml", "")
    write_label(tmp_path / "ds/classes/c.txt", "")
    write_label(tmp_path / "ds/labels/x.txt", "")
    write_label(tmp_path / "ds/annotations/x.json", "")
    write_label(tmp_path / "ds/images/notes.txt", "")
    write_label(tmp_path / ".DS_Store", "")

    found: set[Path] = set(metadata_files(tmp_path, IMAGE_EXTS))

    assert found == {tmp_path / "data.yaml", tmp_path / "ds/classes/c.txt"}


def test_pipeline_mirrors_symlinked_darwin_dataset(
    tmp_path: Path, sharp_image: np.ndarray, flat_image: np.ndarray
) -> None:
    external: Path = make_darwin_batch(tmp_path / "external/batch-1", sharp_image, flat_image)
    raw: Path = tmp_path / "raw"
    raw.mkdir()
    (raw / "darwin_v7").symlink_to(external.parent)
    settings = Settings(data_dir=raw, output_dir=tmp_path / "clean", min_width=32, min_height=32)

    result = build_graph().invoke({"settings": settings, "records": []})

    assert result["summary"] == {"pass": 1, "fail": 1}
    batch: Path = tmp_path / "clean/darwin_v7/batch-1"
    assert (batch / "images/good.png").is_file()
    assert (batch / "annotations/good.json").read_text() == DARWIN_JSON
    assert (batch / "classes/classes_bounding_box.txt").is_file()
    assert not (batch / "images/bad.png").exists()
    assert not (batch / "annotations/bad.json").exists()
    assert not (tmp_path / "clean/darwin_v7").is_symlink()


def test_pipeline_yolo_layout_copies_metadata(tmp_path: Path, sharp_image: np.ndarray) -> None:
    raw: Path = tmp_path / "raw"
    write_image(raw / "images/val/a.png", sharp_image)
    write_label(raw / "labels/val/a.txt", "0 0.5 0.5 0.2 0.2\n")
    write_label(raw / "data.yaml", "names: [logo]\n")
    settings = Settings(data_dir=raw, output_dir=tmp_path / "clean", min_width=32, min_height=32)

    build_graph().invoke({"settings": settings, "records": []})

    clean: Path = tmp_path / "clean"
    assert (clean / "images/val/a.png").is_file()
    assert (clean / "labels/val/a.txt").is_file()
    assert (clean / "data.yaml").read_text() == "names: [logo]\n"
