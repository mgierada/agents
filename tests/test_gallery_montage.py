from pathlib import Path

import numpy as np
from PIL import Image

from cleanup.gallery.montage import review_sheet
from tests.conftest import write_image


def crops(tmp_path: Path, n: int, prefix: str) -> list[Path]:
    rng = np.random.default_rng(0)
    return [
        write_image(
            tmp_path / f"{prefix}{i}.jpg", rng.integers(0, 256, (40, 80, 3), dtype=np.uint8)
        )
        for i in range(n)
    ]


def test_review_sheet_stacks_reference_section_above_review(tmp_path: Path) -> None:
    with_refs: Image.Image = review_sheet(
        crops(tmp_path, 5, "c"), crops(tmp_path, 4, "r"), "acme", tile_px=50, cols=4
    )
    without: Image.Image = review_sheet(crops(tmp_path, 5, "c"), [], "acme", tile_px=50, cols=4)

    assert with_refs.height > without.height
    assert without.width >= 4 * 50
