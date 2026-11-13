"""init -> review -> resolve -> calibrate -> UI API on a tiny synthetic gallery, with a fake
judge (no model calls)."""

import re
from pathlib import Path
from typing import Any

import polars as pl
import pytest
from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableLambda

import cleanup.gallery.review as review
from cleanup.gallery import jsonl
from cleanup.gallery.brands import BrandCatalog
from cleanup.gallery.calibrate import calibrate_run
from cleanup.gallery.config import GallerySettings, RunPaths
from cleanup.gallery.judge import MontageVerdict, TileLabel, TileVerdict
from cleanup.gallery.resolve import decide, resolve_run
from cleanup.gallery.run import init_run
from cleanup.gallery.server import RunState, api_items, api_label
from cleanup.gallery.subset import Subset

# Fake judge answers per brand: tile 1 verdict, then brand_match for the rest.
ANSWERS: dict[str, tuple[TileLabel, str]] = {
    "acme": ("different_brand", "Toyota"),
    "globex": ("different_brand", "Yasuda"),
    "initech": ("unrecognisable", ""),
}


def search(pattern: str, text: str) -> str:
    match: re.Match[str] | None = re.search(pattern, text)
    assert match is not None
    return match.group(1)


def fake_judge(messages: list[BaseMessage]) -> MontageVerdict:
    blocks: list[Any] = list(messages[1].content)
    text: str = "\n".join(b["text"] for b in blocks if b["type"] == "text")
    brand: str = search(r"Labelled brand: (\S+)", text)
    n: int = int(search(r"Judge the (\d+) numbered", text))
    first, suggested = ANSWERS[brand]
    tiles: list[TileVerdict] = [
        TileVerdict(
            tile=i,
            verdict=first if i == 1 else "brand_match",
            visible_text="",
            suggested_brand=suggested if i == 1 else "",
            confidence="high",
        )
        for i in range(1, n + 1)
    ] + [
        TileVerdict(
            tile=99, verdict="not_a_logo", visible_text="", suggested_brand="", confidence="high"
        )
    ]
    return MontageVerdict(cluster_verdict="mixed", cluster_suggested_brand="", tiles=tiles)


@pytest.fixture
def settings(
    gallery_dir: Path, target_list: Path, monkeypatch: pytest.MonkeyPatch
) -> GallerySettings:
    monkeypatch.setattr(review, "build_chain", lambda **_: RunnableLambda(fake_judge))
    return GallerySettings(
        gallery_dir=gallery_dir,
        target_list=target_list,
        judge="fake:judge",
        workers=1,
        min_side_px=20,
    )


@pytest.fixture
def reviewed(settings: GallerySettings) -> RunPaths:
    paths: RunPaths = init_run(settings, "r1", Subset())
    review.review_run(settings, "r1", review.ReviewScope())
    return paths


def test_init_writes_run_with_flags_and_origins(settings: GallerySettings) -> None:
    paths: RunPaths = init_run(settings, "r1", Subset(brands=("acme",)))

    flags: pl.DataFrame = pl.read_parquet(paths.flags).sort("crop_id")
    assert flags["crop_id"].to_list() == ["vidA-0_0", "vidA-0_1", "vidB-3_0"]
    assert flags["origin"].to_list() == ["original", "teacher", "original"]
    assert flags["gallery_unusable"].to_list() == [False, True, False]  # 12px crop is small
    assert paths.priority.exists() and paths.run_json.exists()
    with pytest.raises(FileExistsError):
        init_run(settings, "r1", Subset())


def test_review_skips_flagged_logs_everything_and_resumes(
    settings: GallerySettings, reviewed: RunPaths
) -> None:
    judged: list[dict[str, Any]] = jsonl.read(reviewed.judgements)

    assert "vidA-0_1" not in {r["crop_id"] for r in judged}  # hard-flagged: not sent
    assert all(1 <= r["tile"] <= 16 for r in judged)  # bogus tile 99 ignored
    assert (reviewed.montages / f"{judged[0]['unit_id']}.jpg").exists()
    assert reviewed.references.exists()
    assert len(jsonl.read(reviewed.review_log)) == 1

    assert review.review_run(settings, "r1", review.ReviewScope()) == {}
    assert len(jsonl.read(reviewed.judgements)) == len(judged)


def test_resolve_writes_before_after(settings: GallerySettings, reviewed: RunPaths) -> None:
    decisions: pl.DataFrame = resolve_run(settings, "r1", judge=None, min_confidence="low")
    by_id: dict[str, dict[str, Any]] = {r["crop_id"]: r for r in decisions.to_dicts()}

    assert by_id["vidA-0_1"]["decision"] == "drop_from_gallery"
    assert by_id["vidA-0_1"]["decided_by"] == "rule:small"
    relabelled: list[dict[str, Any]] = [r for r in by_id.values() if r["decision"] == "relabel"]
    # acme has two embedding clusters -> two montages, tile 1 of each relabelled
    assert [(r["brand_before"], r["brand_after"]) for r in relabelled] == [("acme", "toyota")] * 2

    changes: pl.DataFrame = pl.read_csv(reviewed.brand_changes)
    toyota: dict[str, Any] = changes.filter(pl.col("brand") == "toyota").row(0, named=True)
    assert (toyota["before"], toyota["relabelled_in"], toyota["after"]) == (0, 2, 2)
    candidates: pl.DataFrame = pl.read_csv(reviewed.new_classes)
    assert candidates["candidate"].to_list() == ["yasuda"]


def test_human_label_overrides_and_calibrates(
    settings: GallerySettings, reviewed: RunPaths
) -> None:
    state = RunState(settings=settings, run="r1")
    judged: dict[str, dict[str, Any]] = jsonl.latest_by(jsonl.read(reviewed.judgements), "crop_id")
    crop_id, row = next((c, r) for c, r in judged.items() if r["verdict"] == "different_brand")

    api_label(state, {"crop_id": crop_id, "verdict": "brand_match"})
    event: dict[str, Any] = api_label(state, {"crop_id": crop_id, "verdict": "not_a_logo"})

    assert event["previous_verdict"] == "brand_match"
    assert len(jsonl.read(reviewed.events)) == 2
    decisions: pl.DataFrame = resolve_run(settings, "r1", judge=None, min_confidence="low")
    assert decisions.filter(pl.col("crop_id") == crop_id)["decided_by"].item() == "human"

    report: dict[str, Any] = calibrate_run(settings, "r1")
    per_verdict: dict[str, Any] = report["judges"]["fake:judge"]["per_verdict"]
    assert per_verdict["different_brand"]["precision"] == 0.0
    assert row["brand"] in {"acme", "globex"}


def test_ui_api_filters_and_validates(settings: GallerySettings, reviewed: RunPaths) -> None:
    resolve_run(settings, "r1", judge=None, min_confidence="low")
    state = RunState(settings=settings, run="r1")

    changed: dict[str, Any] = api_items(state, {"changed": "1"})
    relabels: dict[str, Any] = api_items(state, {"decision": "relabel"})

    assert all(i["decision"] not in ("keep", "pending") for i in changed["items"])
    assert relabels["total"] == 2
    assert {i["brand_after"] for i in relabels["items"]} == {"toyota"}
    with pytest.raises(ValueError, match="unknown crop_id"):
        api_label(state, {"crop_id": "../x", "verdict": "brand_match"})
    with pytest.raises(ValueError, match="suggested_brand"):
        api_label(state, {"crop_id": "vidA-0_0", "verdict": "different_brand"})


CATALOG = BrandCatalog(targets=frozenset({"acme", "toyota"}), aliases={})
CROP: dict[str, Any] = {"crop_id": "c", "brand": "acme", "flags": [], "gallery_unusable": False}


@pytest.mark.parametrize(
    ("verdict", "expected"),
    [
        (None, ("pending", "acme")),
        ({"verdict": "brand_match", "confidence": "low"}, ("pending", "acme")),
        (
            {"verdict": "different_brand", "suggested_brand": "ACME", "confidence": "high"},
            ("keep", "acme"),
        ),
        (
            {"verdict": "different_brand", "suggested_brand": "Toyota", "confidence": "high"},
            ("relabel", "toyota"),
        ),
        (
            {"verdict": "different_brand", "suggested_brand": "Rewe", "confidence": "high"},
            ("drop_from_gallery", None),
        ),
        ({"verdict": "not_a_logo", "confidence": "medium"}, ("not_a_logo", None)),
    ],
)
def test_decide(verdict: dict[str, Any] | None, expected: tuple[str, str | None]) -> None:
    row: dict[str, Any] | None = verdict and verdict | {"judge": "j"}
    result: dict[str, Any] = decide(CROP, row, None, CATALOG, min_rank=1)
    assert (result["decision"], result["brand_after"]) == expected


def test_rejudge_reviews_again_and_newest_verdict_wins(
    settings: GallerySettings, reviewed: RunPaths
) -> None:
    first: int = len(jsonl.read(reviewed.judgements))

    review.review_run(settings, "r1", review.ReviewScope(rejudge=True))

    rows: list[dict[str, Any]] = jsonl.read(reviewed.judgements)
    assert len(rows) == 2 * first
    latest: dict[str, dict[str, Any]] = jsonl.latest_by(rows, "crop_id")
    assert all(r["ts"] >= rows[0]["ts"] for r in latest.values())


def test_judge_sees_one_sheet_with_references_and_crops(
    settings: GallerySettings, reviewed: RunPaths, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[list[Any]] = []

    def spy(messages: list[BaseMessage]) -> MontageVerdict:
        seen.append(list(messages[1].content))
        return fake_judge(messages)

    monkeypatch.setattr(review, "build_chain", lambda **_: RunnableLambda(spy))
    review.review_run(settings, "r1", review.ReviewScope(brands=("acme",), rejudge=True))

    images: list[Any] = [b for b in seen[0] if b["type"] == "image_url"]
    assert len(images) == 1
    assert not list(reviewed.montages.glob("*_refs.jpg"))
