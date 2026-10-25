"""Runtime settings for the cleanup pipeline."""

from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Paths and quality thresholds. Override via `CLEANUP_*` env vars or `.env`."""

    model_config = SettingsConfigDict(env_prefix="CLEANUP_", env_file=".env", extra="ignore")

    data_dir: Path = Path("data/raw")
    output_dir: Path = Path("data/clean")
    image_exts: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

    min_width: int = 320
    min_height: int = 320
    blur_threshold: float = 100.0
    # Images whose blur score is within this fraction above the threshold are "borderline".
    borderline_margin: float = 0.2
    brightness_min: float = 30.0
    brightness_max: float = 225.0

    # Minimum normalized box area (w * h) before a box counts as degenerate.
    min_box_area: float = 1e-4
    allow_empty_labels: bool = True

    # Fallback decision for borderline images when no judge is configured.
    borderline_policy: Literal["keep", "drop"] = "keep"
    dry_run: bool = False
