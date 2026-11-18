from typing import Any

import polars as pl
import pytest

from cleanup.gallery.brand_stats import brand_stats


def items() -> pl.DataFrame:
    """acme: 4 judged (1 match, 2 -> toyota, 1 not a logo), 1 unjudged; globex: 2 matches."""
    rows: list[tuple[str, str, str | None, str | None, str | None, str, str | None]] = [
        # crop_id, brand, verdict, suggested, relabel_target, decision, brand_after
        ("a1", "acme", "brand_match", "", None, "keep", "acme"),
        ("a2", "acme", "different_brand", "Toyota", "toyota", "suggested_relabel", None),
        ("a3", "acme", "different_brand", "toyota ", "toyota", "relabel", "toyota"),
        ("a4", "acme", "not_a_logo", "", None, "not_a_logo", None),
        ("a5", "acme", None, None, None, "pending", "acme"),
        ("g1", "globex", "brand_match", "", None, "keep", "globex"),
        ("g2", "globex", "brand_match", "", None, "keep", "globex"),
    ]
    columns: list[str] = [
        "crop_id",
        "brand",
        "judge_verdict",
        "judge_suggested_brand",
        "relabel_target",
        "decision",
        "brand_after",
    ]
    return pl.DataFrame(rows, schema=columns, orient="row")


def test_brand_stats_ranks_brands_by_disagreement_and_lists_confusions() -> None:
    human: dict[str, dict[str, Any]] = {
        "a1": {"verdict": "brand_match"},
        "a4": {"verdict": "unrecognisable"},
        "a2": {"verdict": "different_brand", "note": "accepted (bulk)"},
    }

    stats: dict[str, Any] = brand_stats(items(), human, top=5, min_judged=1)

    overall: dict[str, Any] = stats["overall"]
    assert (overall["crops"], overall["judged"], overall["agree"], overall["disagree"]) == (
        7,
        6,
        3,
        3,
    )
    assert overall["human_vs_vlm"] == {"compared": 2, "agree": 1}  # bulk accept excluded
    assert overall["bulk_accepted"] == 1
    acme, globex = stats["brands"]
    assert (acme["brand"], acme["judged"], acme["disagree_rate"]) == ("acme", 4, 0.75)
    assert (acme["mislabel_rate"], acme["relabelled_out"]) == (0.5, 2)
    assert (acme["top_suggested"], acme["top_suggested_n"]) == ("toyota", 2)
    assert (acme["after"], acme["delta"]) == (2, -3)  # a1 + pending a5
    assert globex["disagree_rate"] == 0.0
    assert stats["pairs"] == [
        {"brand": "acme", "suggested": "toyota", "n": 2, "judged": 4, "share_of_judged": 0.5}
    ]


def test_brand_stats_min_judged_and_top_limit_the_ranking() -> None:
    stats: dict[str, Any] = brand_stats(items(), {}, sort="mislabelled", top=1, min_judged=3)

    assert [b["brand"] for b in stats["brands"]] == ["acme"]
    assert (stats["n_brands"], stats["n_ranked"]) == (2, 1)


def test_brand_stats_works_before_any_resolve() -> None:
    unresolved: pl.DataFrame = items().drop("decision", "brand_after", "relabel_target")

    stats: dict[str, Any] = brand_stats(unresolved, {}, min_judged=0)

    assert stats["overall"]["decisions"] == {}
    assert {b["brand"]: b["relabelled_out"] for b in stats["brands"]} == {"acme": 0, "globex": 0}
    assert stats["pairs"][0]["suggested"] == "toyota"


def test_brand_stats_rejects_unknown_sort() -> None:
    with pytest.raises(ValueError, match="sort"):
        brand_stats(items(), {}, sort="nope")


def test_brand_stats_asc_order_puts_most_agreement_first() -> None:
    stats: dict[str, Any] = brand_stats(items(), {}, order="asc", min_judged=1)

    assert [b["brand"] for b in stats["brands"]] == ["globex", "acme"]


def test_brand_stats_ranks_by_each_verdict_share() -> None:
    agree: dict[str, Any] = brand_stats(items(), {}, sort="brand_match_rate", min_judged=1)
    not_logo: dict[str, Any] = brand_stats(items(), {}, sort="not_a_logo_rate", min_judged=1)

    assert [(b["brand"], b["brand_match_rate"]) for b in agree["brands"]] == [
        ("globex", 1.0),
        ("acme", 0.25),
    ]
    assert not_logo["brands"][0]["brand"] == "acme"
    assert set(agree["verdict_sorts"].values()) <= set(agree["sorts"])


def test_brand_stats_offset_pages_through_the_ranking() -> None:
    first: dict[str, Any] = brand_stats(items(), {}, top=1, min_judged=1)
    more: dict[str, Any] = brand_stats(items(), {}, top=1, min_judged=1, offset=1)
    past: dict[str, Any] = brand_stats(items(), {}, top=1, min_judged=1, offset=2)

    assert [b["brand"] for b in first["brands"] + more["brands"]] == ["acme", "globex"]
    assert past["brands"] == []
    with pytest.raises(ValueError, match="order"):
        brand_stats(items(), {}, order="middle")
