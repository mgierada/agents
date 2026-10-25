from pathlib import Path

import numpy as np
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
