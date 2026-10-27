from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from cleanup.state import Verdict


def open_image(path: Path) -> Image.Image:
    """Open an image with EXIF orientation applied. Raises OSError if unreadable."""
    with Image.open(fp=path) as image:
        image.load()
        return ImageOps.exif_transpose(image)


def load_gray(path: Path) -> np.ndarray | None:
    """Load an image as a grayscale array. Returns None if the file is unreadable."""
    try:
        return np.asarray(open_image(path).convert(mode="L"))
    except OSError:
        return None


def blur_score(gray: np.ndarray) -> float:
    """Variance of the 4-neighbour Laplacian (interior pixels). Lower means blurrier."""
    if min(gray.shape[:2]) < 3:
        return 0.0
    g = gray.astype(np.float64)
    laplacian = g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:] - 4 * g[1:-1, 1:-1]
    return float(laplacian.var())


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


def brightness_issues(value: float, min: float, max: float) -> list[str]:
    """Return reasons if the image is under- or over-exposed."""
    if value < min:
        return ["too_dark"]
    if value > max:
        return ["too_bright"]
    return []
