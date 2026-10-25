from pathlib import Path

import numpy as np
import pytest

from cleanup.checks.image import (
    blur_score,
    brightness_issues,
    grade_blur,
    resolution_issues,
)
from cleanup.checks.labels import label_issues, line_issues


def test_blur_score_separates_sharp_and_flat(
    sharp_image: np.ndarray, flat_image: np.ndarray
) -> None:
    assert blur_score(flat_image) == 0.0
    assert blur_score(sharp_image) > 1000.0


@pytest.mark.parametrize(
    ("score", "expected"),
    [(50.0, "fail"), (110.0, "borderline"), (200.0, "pass")],
)
def test_grade_blur(score: float, expected: str) -> None:
    assert grade_blur(score, threshold=100.0, margin=0.2) == expected


def test_resolution_issues() -> None:
    assert resolution_issues(100, 400, 320, 320) == ["low_resolution:100x400"]
    assert resolution_issues(640, 480, 320, 320) == []


@pytest.mark.parametrize(
    ("value", "expected"),
    [(10.0, ["too_dark"]), (250.0, ["too_bright"]), (128.0, [])],
)
def test_brightness_issues(value: float, expected: list[str]) -> None:
    assert brightness_issues(value, 30.0, 225.0) == expected


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        ("0 0.5 0.5 0.2 0.2", []),
        ("0 0.5 0.5", ["malformed_line:1"]),
        ("a 0.5 0.5 0.2 0.2", ["malformed_line:1"]),
        ("-1 0.5 0.5 0.2 0.2", ["negative_class:1"]),
        ("0 1.5 0.5 0.2 0.2", ["out_of_bounds:1"]),
        ("0 0.5 0.5 0.001 0.001", ["tiny_box:1"]),
    ],
)
def test_line_issues(line: str, expected: list[str]) -> None:
    assert line_issues(line, 1, min_box_area=1e-4) == expected


def test_label_issues_missing_file(tmp_path: Path) -> None:
    missing: Path = tmp_path / "nope.txt"
    assert label_issues(missing, 1e-4, allow_empty=True) == []
    assert label_issues(missing, 1e-4, allow_empty=False) == ["missing_label"]


def test_label_issues_reports_line_numbers(tmp_path: Path) -> None:
    path: Path = tmp_path / "x.txt"
    path.write_text("0 0.5 0.5 0.2 0.2\n\n0 2 0.5 0.2 0.2\n")
    assert label_issues(path, 1e-4, allow_empty=True) == ["out_of_bounds:2"]
