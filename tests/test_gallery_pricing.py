import json
from pathlib import Path
from typing import Any

import pytest

from cleanup.gallery.pricing import estimate_usd, spend_report

USAGE: dict[str, int] = {
    "requests": 2,
    "input_tokens": 1_000_000,
    "output_tokens": 1_000_000,
    "cache_read_input_tokens": 1_000_000,
    "cache_creation_input_tokens": 1_000_000,
}


@pytest.mark.parametrize(
    ("judge", "expected"),
    [
        # Anthropic usage: input_tokens already excludes cached tokens.
        ("claude:claude-haiku-5-5", 0.10 + 0.50 + 0.01 + 0.125),
        # LangChain usage: input_tokens includes cache reads/writes, so none are left uncached.
        ("bedrock-converse:global.anthropic.claude-haiku-5-5", 0.50 + 0.01 + 0.125),
        ("ollama:qwen2.5vl:7b", 0.0),
        ("openai:unknown-model", None),
    ],
)
def test_estimate_usd(judge: str, expected: float | None) -> None:
    usd: float | None = estimate_usd(judge, USAGE)
    assert usd == (None if expected is None else pytest.approx(expected))


def write_log(root: Path, run: str, entries: list[dict[str, Any]]) -> None:
    (root / run).mkdir(parents=True)
    (root / run / "review_log.jsonl").write_text("".join(json.dumps(e) + "\n" for e in entries))


def entry(ts: str, judge: str, input_tokens: int) -> dict[str, Any]:
    usage: dict[str, int] = {"requests": 1, "input_tokens": input_tokens, "output_tokens": 0}
    return {"ts": ts, "judge": judge, "units": 1, "crops": 1, "usage": usage}


def test_spend_report_totals_per_run_and_gallery(tmp_path: Path) -> None:
    write_log(
        tmp_path,
        "a",
        [
            entry("2026-11-13T10:00:00+00:00", "claude:claude-haiku-5-5", 1_000_000),
            entry("2026-11-13T11:00:00+00:00", "openai:mystery", 5),
            entry("2026-11-13T11:30:00+00:00", "bedrock:failed-model", 0) | {"usage": {}},
        ],
    )
    write_log(
        tmp_path, "b", [entry("2026-11-13T12:00:00+00:00", "claude:claude-opus-5-5", 1_000_000)]
    )

    report: dict[str, Any] = spend_report(tmp_path, "a", recent=5)

    assert report["run_usd"] == pytest.approx(0.10)
    assert report["all_usd"] == pytest.approx(4.10)
    assert report["unpriced_judges"] == ["openai:mystery"]
    assert [r["judge"] for r in report["recent"]] == [
        "bedrock:failed-model",
        "openai:mystery",
        "claude:claude-haiku-5-5",
    ]
    assert report["recent"][0]["usd"] == 0.0  # no successful request, nothing billed
    assert report["recent"][2]["usd_per_request"] == pytest.approx(0.10)
    assert any(p["judge"] == "claude:claude-haiku-5-5" for p in report["prices"])
