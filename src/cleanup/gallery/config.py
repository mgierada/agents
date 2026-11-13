"""Settings of the gallery audit and the on-disk layout of one audit run."""

import os
from dataclasses import dataclass
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

from cleanup.llm import Effort

ANALYSIS_NAME = "analysis.parquet"
SUMMARY_NAME = "summary.json"
CROPS_DIR_NAME = "crops"
CACHE_DIR_NAME = "_cache"
# Brands behind the regression-test identification failures; reviewed first.
FOCUS_BRANDS: tuple[str, ...] = (
    "guaranteed_rate",
    "bmo_harris_bank",
    "algorand",
    "benjamin_moore",
    "charlotte's_web",
)
# Flags that make a crop unusable for the gallery. The rest only inform review order.
# extreme_aspect is soft: wide wordmarks (e.g. guaranteed_rate, 17% of its crops) are real.
# conflicting_label is soft: the judge decides which of the two brands is right.
HARD_FLAGS: tuple[str, ...] = (
    "unreadable",
    "small",
    "empty",
    "low_blur",
    "out_of_bounds",
    "duplicate",
)


class GallerySettings(BaseSettings):
    """Gallery audit settings. Override via `CLEANUP_GALLERY_*` env vars or `.env`."""

    model_config = SettingsConfigDict(
        env_prefix="CLEANUP_GALLERY_", env_file=".env", extra="ignore"
    )

    # Logo-audit output dir: analysis.parquet, summary.json, crops/, labels dir.
    gallery_dir: Path = Path(".")
    # Where runs are written; defaults to <gallery_dir>/agent_audit.
    runs_dir: Path | None = None
    labels_dir_name: str = "labels_with_pseudo"
    # One brand per line (line number = class id). None: the gallery's own brands.
    target_list: Path | None = None
    # JSON {"old_name": "new_name"}: LEGACY_LOGO_ALIASES and SPELLING_CORRECTIONS merged.
    aliases: Path | None = None
    # Pseudo-label lines whose run_name starts with this are the distillation teacher's.
    teacher_run_prefix: str = "8baece"
    workers: int = os.cpu_count() or 4

    # Deterministic flags. Blur/contrast are measured on the saved (<=256px) crop.
    min_side_px: int = 20
    # On the 8k-crop sample: blur<10 -> 5.7% of crops, mean kNN agreement 0.37 vs 0.74 above 200.
    blur_min: float = 10.0
    blur_brand_quantile: float = 0.05
    # Brands with fewer crops in the run get no brand-relative blur flag.
    blur_brand_min_crops: int = 20
    empty_std: float = 4.0
    contrast_min: float = 12.0
    brightness_min: float = 20.0
    brightness_max: float = 235.0
    max_aspect: float = 6.0
    duplicate_iou: float = 0.9
    hard_flags: tuple[str, ...] = HARD_FLAGS

    # VLM review.
    judge: str = "claude:claude-haiku-5-5"
    judge_effort: Effort = "low"
    montage_size: int = 16
    montage_cols: int = 4
    tile_px: int = 160
    reference_count: int = 4
    reference_min_agreement: float = 0.9
    reference_min_side_px: int = 64
    review_workers: int = 4

    @property
    def audit_root(self) -> Path:
        """Directory holding all runs of this gallery."""
        return self.runs_dir or self.gallery_dir / "agent_audit"

    @property
    def crops_dir(self) -> Path:
        """Directory of the saved crops, `<crop_id>.jpg`."""
        return self.gallery_dir / CROPS_DIR_NAME

    @property
    def labels_dir(self) -> Path:
        """YOLO label files of the batch the gallery was cropped from."""
        return self.gallery_dir / self.labels_dir_name

    @property
    def cache_dir(self) -> Path:
        """Per-gallery cache of crop statistics and parsed label boxes, shared by runs."""
        return self.audit_root / CACHE_DIR_NAME

    def run_paths(self, run: str) -> "RunPaths":
        """Paths of the named run."""
        return RunPaths(root=self.audit_root / run)


@dataclass(frozen=True)
class RunPaths:
    """Files of one audit run. Inputs are never modified; every output lives here."""

    root: Path

    @property
    def run_json(self) -> Path:
        """Run metadata: subset, settings, created_at."""
        return self.root / "run.json"

    @property
    def crops(self) -> Path:
        """The run's working set: analysis rows of the selected crops."""
        return self.root / "crops.parquet"

    @property
    def flags(self) -> Path:
        """Deterministic flags per crop."""
        return self.root / "flags.parquet"

    @property
    def priority(self) -> Path:
        """Brand review order."""
        return self.root / "priority.csv"

    @property
    def references(self) -> Path:
        """Reference crop ids per brand shown to the judge; editable by hand."""
        return self.root / "references.json"

    @property
    def judgements(self) -> Path:
        """Append-only per-crop judge verdicts."""
        return self.root / "judgements.jsonl"

    @property
    def units(self) -> Path:
        """Append-only per-montage judge calls: cluster verdict, usage, errors."""
        return self.root / "units.jsonl"

    @property
    def review_log(self) -> Path:
        """Append-only log of review invocations: judge, scope, units, token usage."""
        return self.root / "review_log.jsonl"

    @property
    def montages(self) -> Path:
        """Montages exactly as the judge saw them."""
        return self.root / "montages"

    @property
    def events(self) -> Path:
        """Append-only human labels and overrides from the UI."""
        return self.root / "events.jsonl"

    @property
    def decisions(self) -> Path:
        """Final per-crop decision: before/after brand, decided_by."""
        return self.root / "decisions.parquet"

    @property
    def brand_changes(self) -> Path:
        """Per-brand before/after counts."""
        return self.root / "brand_changes.csv"

    @property
    def new_classes(self) -> Path:
        """Brands the judge saw that are not on the target list."""
        return self.root / "new_class_candidates.csv"

    @property
    def summary(self) -> Path:
        """Decision counts and judge usage."""
        return self.root / "summary.json"

    @property
    def calibration(self) -> Path:
        """Judge and flag agreement with human labels."""
        return self.root / "calibration.json"
