from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from cleanup.gallery.judge import MontageVerdict
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


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("mostly_not_a_logo", "not_a_logo"),  # seen live from Haiku 5.5
        ("Not a logo", "not_a_logo"),
        ("nearly_all_consistent", "consistent"),
        ("unreadable", "unreadable"),
        ("brand_match", "mixed"),  # a crop label: no montage meaning, keep the crops anyway
    ],
)
def test_montage_summary_near_misses_keep_crop_verdicts(raw: str, expected: str) -> None:
    verdict = MontageVerdict.model_validate(
        {"cluster_verdict": raw, "cluster_suggested_brand": "", "tiles": []}
    )
    assert verdict.cluster_verdict == expected
