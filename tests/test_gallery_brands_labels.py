import json
from pathlib import Path

from cleanup.gallery.brands import BrandCatalog, load_catalog, normalise
from cleanup.gallery.labels import LabelBox, Origin, find_overlaps, label_rows, parse_line

CATALOG = BrandCatalog(
    targets=frozenset({"discount_tire", "u.s._bank", "benjamin_moore", "bmw"}),
    aliases={"enjamin_moore": "benjamin_moore"},
)


def box(crop_id: str, class_id: int, cx: float, origin: Origin = "original") -> LabelBox:
    return LabelBox(crop_id, class_id, cx, 0.5, 0.2, 0.2, origin, None, None)


def test_normalise() -> None:
    assert normalise("  Discount Tire ") == "discount_tire"
    assert normalise("Coca-Cola") == "coca_cola"


def test_resolve_exact_alias_fuzzy_and_unknown() -> None:
    assert CATALOG.resolve("Discount Tire") == "discount_tire"
    assert CATALOG.resolve("enjamin moore") == "benjamin_moore"
    assert CATALOG.resolve("u.s. bank") == "u.s._bank"
    assert CATALOG.resolve("Discount Tires") == "discount_tire"
    assert CATALOG.resolve("bmo") is None
    assert CATALOG.resolve("YASUDA") is None
    assert CATALOG.resolve("  ") is None


def test_load_catalog_applies_aliases_to_target_list(tmp_path: Path) -> None:
    targets: Path = tmp_path / "t.txt"
    targets.write_text("enjamin_moore\nacme\n\n")
    aliases: Path = tmp_path / "a.json"
    aliases.write_text(json.dumps({"enjamin_moore": "benjamin_moore"}))

    catalog: BrandCatalog = load_catalog(targets, aliases, fallback=["ignored"])

    assert catalog.targets == {"benjamin_moore", "acme"}


def test_load_catalog_falls_back_to_gallery_brands() -> None:
    assert load_catalog(None, None, fallback=["Acme"]).targets == {"acme"}


def test_parse_line_tags_origin_and_confidence() -> None:
    original: LabelBox = parse_line("s_0", "3 0.5 0.5 0.1 0.1 0", "8baece")
    teacher: LabelBox = parse_line("s_1", "3 0.5 0.5 0.1 0.1 0 0.91 [run_name=8baece93]", "8baece")
    pseudo: LabelBox = parse_line("s_2", "3 0.5 0.5 0.1 0.1 0 0.7 [run_name=train_v8]", "8baece")

    assert (original.origin, original.confidence) == ("original", None)
    assert (teacher.origin, teacher.confidence) == ("teacher", 0.91)
    assert (pseudo.origin, pseudo.run_name) == ("pseudo", "train_v8")


def test_out_of_bounds() -> None:
    assert not box("a", 0, 0.5).out_of_bounds
    assert box("a", 0, 0.95).out_of_bounds


def test_find_overlaps_separates_duplicates_from_conflicts() -> None:
    boxes: list[LabelBox] = [
        box("s_0", 1, 0.5, origin="pseudo"),
        box("s_1", 1, 0.5),  # same class, original: kept, the pseudo one is the duplicate
        box("s_2", 2, 0.5),  # same place, other class: conflict
        box("s_3", 1, 0.1),  # elsewhere
    ]
    duplicates, conflicts = find_overlaps(boxes, iou_threshold=0.9)

    assert duplicates == {"s_0": "s_1"}
    assert conflicts == {"s_2": "s_1", "s_1": "s_2", "s_0": "s_2"}


def test_label_rows_numbers_non_empty_lines(tmp_path: Path) -> None:
    path: Path = tmp_path / "vid-3.txt"
    path.write_text("0 0.5 0.5 0.2 0.2 0\n\n1 0.1 0.1 0.1 0.1 0\n")

    rows: list[dict[str, object]] = label_rows(path, "8baece", duplicate_iou=0.9)

    assert [r["crop_id"] for r in rows] == ["vid-3_0", "vid-3_1"]
    assert label_rows(tmp_path / "missing.txt", "8baece", 0.9) == []
