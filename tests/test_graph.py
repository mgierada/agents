from pathlib import Path

import numpy as np

from cleanup.checks.image import blur_score
from cleanup.config import Settings
from cleanup.graph import build_graph
from cleanup.state import ImageRecord
from tests.conftest import write_image, write_label

GOOD_LABEL = "0 0.5 0.5 0.2 0.2\n"


def make_settings(tmp_path: Path, **overrides: object) -> Settings:
    """Settings pointing at tmp dirs, sized for the 64x64 test images."""
    base: dict[str, object] = {
        "data_dir": tmp_path / "raw",
        "output_dir": tmp_path / "clean",
        "min_width": 32,
        "min_height": 32,
    }
    return Settings(**(base | overrides))  # type: ignore[arg-type]


def test_pipeline_copies_only_passing_images(
    tmp_path: Path, sharp_image: np.ndarray, flat_image: np.ndarray
) -> None:
    raw: Path = tmp_path / "raw"
    write_image(raw / "images/train/good.png", sharp_image)
    write_label(raw / "labels/train/good.txt", GOOD_LABEL)
    write_image(raw / "images/train/blurry.png", flat_image)
    write_image(raw / "images/train/bad_label.png", sharp_image)
    write_label(raw / "labels/train/bad_label.txt", "0 0.5\n")
    write_image(raw / "images/val/tiny.png", sharp_image[:16, :16])

    settings: Settings = make_settings(tmp_path)
    result = build_graph().invoke({"settings": settings, "records": []})

    assert result["summary"] == {"pass": 1, "fail": 3}
    clean: Path = tmp_path / "clean"
    assert (clean / "images/train/good.png").exists()
    assert (clean / "labels/train/good.txt").read_text() == GOOD_LABEL
    assert not (clean / "images/train/blurry.png").exists()
    assert (clean / "report.json").exists()


def test_dry_run_writes_report_only(tmp_path: Path, sharp_image: np.ndarray) -> None:
    write_image(tmp_path / "raw/images/a.png", sharp_image)
    settings: Settings = make_settings(tmp_path, dry_run=True)

    build_graph().invoke({"settings": settings, "records": []})

    assert (tmp_path / "clean/report.json").exists()
    assert not (tmp_path / "clean/images").exists()


def test_borderline_images_go_through_judge(tmp_path: Path, sharp_image: np.ndarray) -> None:
    write_image(tmp_path / "raw/images/a.png", sharp_image)
    # Place the threshold just below the image's score so it lands in the borderline band.
    settings: Settings = make_settings(
        tmp_path, blur_threshold=blur_score(sharp_image) / 1.1, borderline_margin=0.2
    )
    seen: list[ImageRecord] = []

    def drop_all(record: ImageRecord) -> bool:
        seen.append(record)
        return False

    result = build_graph(judge=drop_all).invoke({"settings": settings, "records": []})

    assert [r.rel_path for r in seen] == [Path("a.png")]
    assert result["summary"] == {"fail": 1}
    assert result["records"][0].reasons == ("borderline_blur", "judge:drop")
