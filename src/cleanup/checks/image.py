"""Image-level quality checks."""

from pathlib import Path

import cv2
import numpy as np

from cleanup.state import Verdict


def load_gray(path: Path) -> np.ndarray | None:
    """Load an image as grayscale. Returns None if the file is unreadable."""
    return cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)


def blur_score(gray: np.ndarray) -> float:
    """Variance of the Laplacian. Lower means blurrier."""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def brightness(gray: np.ndarray) -> float:
    """Mean pixel intensity in [0, 255]."""
    return float(gray.mean())


def grade_blur(score: float, threshold: float, margin: float) -> Verdict:
    """Grade a blur score: fail below threshold, borderline within margin above it."""
    if score < threshold:
        return "fail"
    if score < threshold * (1 + margin):
        return "borderline"
    return "pass"


def resolution_issues(width: int, height: int, min_width: int, min_height: int) -> list[str]:
    """Return reasons if the image is smaller than the minimum size."""
    if width < min_width or height < min_height:
        return [f"low_resolution:{width}x{height}"]
    return []


def brightness_issues(value: float, minimum: float, maximum: float) -> list[str]:
    """Return reasons if the image is under- or over-exposed."""
    if value < minimum:
        return ["too_dark"]
    if value > maximum:
        return ["too_bright"]
    return []
