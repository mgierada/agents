import polars as pl

from cleanup.gallery.config import GallerySettings
from cleanup.gallery.flags import add_flags
from cleanup.gallery.priority import brand_priority, parse_confusions
from cleanup.gallery.subset import Subset, select_crops

SETTINGS = GallerySettings(blur_brand_min_crops=3)


def flag_table(**overrides: list[object]) -> pl.DataFrame:
    """Three clean crops of one brand, with column overrides."""
    base: dict[str, list[object]] = {
        "crop_id": ["a", "b", "c"],
        "brand": ["x", "x", "x"],
        "box_w": [100, 100, 100],
        "box_h": [50, 50, 50],
        "readable": [True, True, True],
        "blur": [500.0, 600.0, 700.0],
        "contrast": [40.0, 40.0, 40.0],
        "brightness": [120.0, 120.0, 120.0],
        "out_of_bounds": [False, False, False],
        "duplicate_of": [None, None, None],
        "conflicts_with": [None, None, None],
        "label_found": [True, True, True],
    }
    return pl.DataFrame(base | overrides)


def flags_of(table: pl.DataFrame) -> list[list[str]]:
    return add_flags(table, SETTINGS)["flags"].to_list()


def test_clean_crops_get_no_flags() -> None:
    flagged: pl.DataFrame = add_flags(flag_table(blur=[700.0, 700.0, 700.0]), SETTINGS)
    assert flagged["flags"].to_list() == [[], [], []]
    assert not flagged["gallery_unusable"].any()


def test_hard_flags_make_crop_unusable() -> None:
    table: pl.DataFrame = flag_table(box_h=[50, 10, 50], blur=[500.0, 600.0, 5.0])
    flagged: pl.DataFrame = add_flags(table, SETTINGS)

    assert "small" in flagged["flags"][1]
    assert "low_blur" in flagged["flags"][2]
    assert flagged["gallery_unusable"].to_list() == [False, True, True]


def test_soft_flags_do_not_remove_crops() -> None:
    table: pl.DataFrame = flag_table(
        box_w=[100, 900, 100],
        conflicts_with=[None, None, "a"],
        brightness=[10.0, 120.0, 120.0],
        blur=[700.0] * 3,
    )
    flagged: pl.DataFrame = add_flags(table, SETTINGS)

    assert flagged["flags"].to_list() == [["too_dark"], ["extreme_aspect"], ["conflicting_label"]]
    assert not flagged["gallery_unusable"].any()


def test_missing_label_info_is_not_flagged() -> None:
    table: pl.DataFrame = flag_table(blur=[700.0] * 3).with_columns(
        pl.lit(None, dtype=pl.Boolean).alias("out_of_bounds"),
        pl.lit(None, dtype=pl.Boolean).alias("label_found"),
    )
    assert flags_of(table) == [[], [], []]


def test_brand_relative_blur_needs_enough_crops() -> None:
    table: pl.DataFrame = flag_table(blur=[50.0, 600.0, 700.0])
    assert flags_of(table)[0] == ["low_blur_rel"]
    assert flags_of(table.head(2))[0] == []


ANALYSIS = pl.DataFrame(
    {
        "crop_id": [f"c{i}" for i in range(12)],
        "brand": ["a"] * 4 + ["b"] * 4 + ["c"] * 4,
        "suspect": [True, False, False, False] + [False] * 8,
        "knn_agreement": [0.2, 1.0, 1.0, 1.0] + [1.0] * 8,
    }
)
CONFUSIONS: pl.DataFrame = parse_confusions(
    {"best": "m", "m": {"top_confusions": ["a -> b: 4", "b -> a: 2", "bad line"]}}
)


def test_parse_confusions() -> None:
    assert CONFUSIONS.to_dicts() == [
        {"brand": "a", "partner": "b", "count": 4},
        {"brand": "b", "partner": "a", "count": 2},
    ]


def test_brand_priority_puts_focus_then_confused_brands_first() -> None:
    priority: pl.DataFrame = brand_priority(ANALYSIS, CONFUSIONS, focus=["c"])

    assert priority["brand"].to_list() == ["c", "a", "b"]
    assert priority["rank"].to_list() == [1, 2, 3]
    assert priority.filter(pl.col("brand") == "a")["confusion_count"].item() == 6


def test_select_crops_filters_and_limits() -> None:
    priority: pl.DataFrame = brand_priority(ANALYSIS, CONFUSIONS, focus=[])

    top: pl.DataFrame = select_crops(ANALYSIS, Subset(top_brands=1), priority, set())
    suspects: pl.DataFrame = select_crops(
        ANALYSIS, Subset(brands=("a", "b"), suspects_only=True), priority, set()
    )
    limited: pl.DataFrame = select_crops(ANALYSIS, Subset(limit=5), priority, set())

    assert set(top["brand"]) == {"a"}
    assert suspects["crop_id"].to_list() == ["c0"]
    assert limited.height == 5


def test_calibration_sample_is_stratified() -> None:
    priority: pl.DataFrame = brand_priority(ANALYSIS, CONFUSIONS, focus=[])
    sample: pl.DataFrame = select_crops(
        ANALYSIS, Subset(calibration_sample=6), priority, confusion_brands={"a", "b"}
    )

    counts: dict[str, int] = dict(sample["stratum"].value_counts().iter_rows())
    assert sample.height == 6
    assert sample["crop_id"].n_unique() == 6
    # one suspect exists, so the random stratum is topped up
    assert counts == {"suspect": 1, "confusion_pair": 2, "random": 3}
