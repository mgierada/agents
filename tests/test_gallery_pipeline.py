"""init -> review -> resolve -> calibrate -> UI API on a tiny synthetic gallery, with a fake
judge (no model calls)."""

import json
import re
from pathlib import Path
from typing import Any

import polars as pl
import pytest
from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableLambda

import cleanup.gallery.review as review
import cleanup.llm as llm
from cleanup.gallery import jsonl
from cleanup.gallery.brands import BrandCatalog
from cleanup.gallery.calibrate import calibrate_run
from cleanup.gallery.config import GallerySettings, RunPaths
from cleanup.gallery.judge import MontageVerdict, TileLabel, TileVerdict
from cleanup.gallery.resolve import decide, resolve_run
from cleanup.gallery.run import init_run
from cleanup.gallery.server import (
    RunState,
    api_accept_relabels,
    api_items,
    api_label,
    api_meta,
    api_progress,
    api_rejudge,
    api_spend,
)
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
    # One judge's different_brand only suggests: out of acme, not yet into toyota.
    suggested: list[dict[str, Any]] = [
        r for r in by_id.values() if r["decision"] == "suggested_relabel"
    ]
    assert [(r["brand_before"], r["brand_after"], r["relabel_target"]) for r in suggested] == [
        ("acme", None, "toyota")
    ] * 2
    assert not pl.read_csv(reviewed.brand_changes).filter(pl.col("brand") == "toyota").height

    decisions = resolve_run(
        settings, "r1", judge=None, min_confidence="low", trust_judge_relabels=True
    )
    by_id = {r["crop_id"]: r for r in decisions.to_dicts()}
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
    suggested: dict[str, Any] = api_items(state, {"decision": "suggested_relabel"})

    assert all(i["decision"] not in ("keep", "pending") for i in changed["items"])
    assert suggested["total"] == 2
    assert {i["relabel_target"] for i in suggested["items"]} == {"toyota"}
    with pytest.raises(ValueError, match="unknown crop_id"):
        api_label(state, {"crop_id": "../x", "verdict": "brand_match"})
    with pytest.raises(ValueError, match="suggested_brand"):
        api_label(state, {"crop_id": "vidA-0_0", "verdict": "different_brand"})


def test_accept_relabels_confirms_filtered_suggestions(
    settings: GallerySettings, reviewed: RunPaths
) -> None:
    resolve_run(settings, "r1", judge=None, min_confidence="low")
    state = RunState(settings=settings, run="r1")

    none: dict[str, Any] = api_accept_relabels(state, {"filters": {"brand": "globex"}})
    some: dict[str, Any] = api_accept_relabels(
        state, {"filters": {"brand": "acme", "decision": "suggested_relabel"}}
    )

    assert none["accepted"] == 0
    assert some["accepted"] == 2
    relabels: dict[str, Any] = api_items(state, {"decision": "relabel"})
    assert {(i["brand_after"], i["decided_by"]) for i in relabels["items"]} == {("toyota", "human")}
    events: list[dict[str, Any]] = jsonl.read(reviewed.events)
    assert [e["note"] for e in events] == ["accepted (bulk)"] * 2


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
            ("suggested_relabel", None),
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


TOYOTA: dict[str, Any] = {
    "verdict": "different_brand",
    "suggested_brand": "Toyota",
    "confidence": "high",
    "judge": "j",
}


@pytest.mark.parametrize(
    ("opinions", "trust", "expected"),
    [
        ([], False, "suggested_relabel"),
        ([], True, "relabel"),
        ([TOYOTA | {"judge": "j2", "suggested_brand": "TOYOTA"}], False, "relabel"),
        ([TOYOTA | {"judge": "j2", "suggested_brand": "acme"}], False, "suggested_relabel"),
        ([TOYOTA], False, "suggested_relabel"),  # the same judge does not second itself
    ],
)
def test_relabel_needs_second_judge_or_trust(
    opinions: list[dict[str, Any]], trust: bool, expected: str
) -> None:
    result: dict[str, Any] = decide(
        CROP, TOYOTA, None, CATALOG, min_rank=0, opinions=opinions, trust_judge_relabels=trust
    )
    assert result["decision"] == expected
    assert result["relabel_target"] == "toyota"


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


def test_ui_api_spend_counts_review_calls(settings: GallerySettings, reviewed: RunPaths) -> None:
    spend: dict[str, Any] = api_spend(RunState(settings=settings, run="r1"))

    # The fake judge reports no billed requests: $0, not "unpriced".
    assert (spend["run_usd"], spend["unpriced_judges"]) == (0, [])
    assert [r["judge"] for r in spend["recent"]] == ["fake:judge"]


INVALID_ANSWER: dict[str, Any] = {
    "cluster_verdict": "brand_match",  # tile label in the cluster field
    "cluster_suggested_brand": "",
    "tiles": [],
}


def review_with(settings: GallerySettings, monkeypatch: pytest.MonkeyPatch, judge: Any) -> RunPaths:
    """Review run r1 (acme only) with `judge` as the chain."""
    monkeypatch.setattr(review, "build_chain", lambda **_: RunnableLambda(judge))
    paths: RunPaths = init_run(settings, "r1", Subset(brands=("acme",)))
    review.review_run(settings, "r1", review.ReviewScope())
    return paths


def test_invalid_answer_is_retried(
    settings: GallerySettings, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[int] = []

    def flaky(messages: list[BaseMessage]) -> Any:
        calls.append(1)
        return INVALID_ANSWER if len(calls) % 2 else fake_judge(messages)

    paths: RunPaths = review_with(settings, monkeypatch, flaky)

    units: list[dict[str, Any]] = jsonl.read(paths.units)
    assert len(calls) == 2 * len(units)  # each unit: one invalid answer, one retry
    assert not any("error" in u for u in units)


def test_retries_stop_at_the_limit(
    settings: GallerySettings, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[int] = []

    def invalid(_messages: list[BaseMessage]) -> Any:
        calls.append(1)
        return INVALID_ANSWER

    paths: RunPaths = review_with(
        settings.model_copy(update={"judge_retries": 1}), monkeypatch, invalid
    )

    units: list[dict[str, Any]] = jsonl.read(paths.units)
    assert len(calls) == 2 * len(units)
    assert all("cluster_verdict" in u["error"] for u in units)


def test_auth_errors_are_not_retried(
    settings: GallerySettings, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[int] = []

    def denied(_messages: list[BaseMessage]) -> Any:
        calls.append(1)
        raise RuntimeError("401 signature expired")

    paths: RunPaths = review_with(settings, monkeypatch, denied)

    assert len(calls) == len(jsonl.read(paths.units))


def test_plot_writes_calibration_charts(settings: GallerySettings, reviewed: RunPaths) -> None:
    from cleanup.gallery.plots import plot_run

    state = RunState(settings=settings, run="r1")
    for crop_id in jsonl.latest_by(jsonl.read(reviewed.judgements), "crop_id"):
        api_label(state, {"crop_id": crop_id, "verdict": "brand_match"})
    calibrate_run(settings, "r1")

    written: list[Path] = plot_run(settings, "r1")

    assert sorted(p.name for p in written) == [
        "accuracy.png",
        "confusion.png",
        "flags.png",
        "precision_recall.png",
    ]
    assert all(p.stat().st_size > 0 for p in written)


def test_review_writes_progress_for_the_ui(settings: GallerySettings, reviewed: RunPaths) -> None:
    progress: dict[str, Any] = api_progress(RunState(settings=settings, run="r1"))

    assert progress["status"] == "finished"
    assert progress["done_units"] == progress["total_units"] == len(jsonl.read(reviewed.units))
    assert progress["done_crops"] == progress["total_crops"]
    assert progress["errors"] == 0


def test_interrupted_review_is_marked_stopped(
    settings: GallerySettings, monkeypatch: pytest.MonkeyPatch
) -> None:
    def interrupted(_messages: list[BaseMessage]) -> Any:
        raise KeyboardInterrupt

    monkeypatch.setattr(review, "build_chain", lambda **_: RunnableLambda(interrupted))
    paths: RunPaths = init_run(settings, "r1", Subset(brands=("acme",)))
    monkeypatch.setattr(review.UnitReviewer, "__call__", lambda self, unit: interrupted([]))

    with pytest.raises(KeyboardInterrupt):
        review.review_run(settings, "r1", review.ReviewScope())

    assert json.loads(paths.progress.read_text())["status"] == "stopped"


def test_ui_gets_refresh_interval_from_settings(
    settings: GallerySettings, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CLEANUP_GALLERY_UI_REFRESH_SECONDS", "3")
    configured = GallerySettings(**settings.model_dump(exclude={"ui_refresh_seconds"}))
    init_run(configured, "r1", Subset())
    state = RunState(settings=configured, run="r1")

    assert api_meta(state)["refresh_seconds"] == 3
    assert api_progress(state) == {"status": "none"}


def test_review_filter_shows_pending_queued_and_judging(
    settings: GallerySettings, reviewed: RunPaths
) -> None:
    state = RunState(settings=settings, run="r1")
    judged: list[str] = sorted(jsonl.latest_by(jsonl.read(reviewed.judgements), "crop_id"))
    unjudged: set[str] = set(state.items["crop_id"]) - set(judged)
    queued, judging, finished = judged[0], judged[1], judged[2]

    # Simulate a running --rejudge over three crops: one finished, one in a worker, one queued.
    tracker = review.ProgressTracker(
        path=reviewed.progress, judge="fake:judge", total_units=3, total_crops=3
    )
    tracker.write_queue(
        reviewed.review_queue,
        [queued, judging, finished],
        units_offset=reviewed.units.stat().st_size,
    )
    tracker.start(review.ReviewUnit("u-judging", "acme", 0, (judging,)))
    jsonl.append(
        reviewed.units,
        [{"unit_id": "u-done", "judge": "fake:judge", "ts": jsonl.now(), "crop_ids": [finished]}],
    )

    def ids(review_value: str) -> set[str]:
        page: dict[str, Any] = api_items(state, {"review": review_value, "limit": "500"})
        return {i["crop_id"] for i in page["items"]}

    assert ids("judging") == {judging}
    assert ids("in_review") == {queued, judging}
    assert ids("pending") == unjudged | {queued, judging}
    assert ids("not_judged") == unjudged
    assert finished in ids("judged") and queued not in ids("judged")
    statuses: dict[str, str | None] = {
        i["crop_id"]: i["review_status"] for i in api_items(state, {"limit": "500"})["items"]
    }
    assert (statuses[judging], statuses[queued], statuses[finished]) == ("judging", "queued", None)

    tracker.finish("finished")
    assert ids("in_review") == set()


def test_failed_units_record_their_crops(
    settings: GallerySettings, monkeypatch: pytest.MonkeyPatch
) -> None:
    def denied(_messages: list[BaseMessage]) -> Any:
        raise RuntimeError("401")

    paths: RunPaths = review_with(settings, monkeypatch, denied)

    assert all(u["crop_ids"] for u in jsonl.read(paths.units))


def test_second_opinion_from_ui_is_logged_and_confirms_relabel(
    settings: GallerySettings, reviewed: RunPaths
) -> None:
    resolve_run(settings, "r1", judge=None, min_confidence="low")
    state = RunState(settings=settings, run="r1")
    crop_id: str = api_items(state, {"decision": "suggested_relabel"})["items"][0]["crop_id"]

    result: dict[str, Any] = api_rejudge(state, {"crop_id": crop_id, "judge": "fake:second"})

    assert result["verdict"]["judge"] == "fake:second"
    assert result["item"]["judge"] == "fake:second"  # card shows the newest verdict
    assert jsonl.read(reviewed.units)[-1]["source"] == "ui"
    assert jsonl.read(reviewed.review_log)[-1]["scope"] == {"source": "ui", "crop_id": crop_id}
    assert "fake:second" in api_meta(state)["judges"]
    # Two judges named the same brand: the suggestion becomes a relabel.
    decisions: pl.DataFrame = resolve_run(settings, "r1", judge=None, min_confidence="low")
    row: dict[str, Any] = decisions.filter(pl.col("crop_id") == crop_id).row(0, named=True)
    assert (row["decision"], row["brand_after"]) == ("relabel", "toyota")


def test_second_opinion_errors_are_reported(
    settings: GallerySettings, reviewed: RunPaths, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = RunState(settings=settings, run="r1")
    crop_id: str = state.items["crop_id"][0]
    judged_before: int = len(jsonl.read(reviewed.judgements))

    def denied(_messages: list[BaseMessage]) -> Any:
        raise RuntimeError("401 signature expired")

    monkeypatch.setattr(review, "build_chain", lambda **_: RunnableLambda(denied))
    with pytest.raises(ValueError, match="401 signature expired"):
        api_rejudge(state, {"crop_id": crop_id, "judge": "fake:broken"})
    monkeypatch.setattr(review, "build_chain", llm.build_chain)
    with pytest.raises(ValueError, match="not supported"):
        api_rejudge(state, {"crop_id": crop_id, "judge": "nope:model"})
    with pytest.raises(ValueError, match="unknown crop_id"):
        api_rejudge(state, {"crop_id": "missing", "judge": "fake:second"})

    assert len(jsonl.read(reviewed.judgements)) == judged_before
