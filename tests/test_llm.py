import importlib.util
from types import SimpleNamespace
from typing import Any

import anthropic
import pytest
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableLambda

import cleanup.llm as llm
from cleanup.judge import JudgeVerdict
from cleanup.llm import ClaudeStructured, bedrock_model_id, build_chain, parse_spec, to_anthropic


class FakeMessages:
    """Stands in for `client.messages`; returns a canned response and records the call."""

    def __init__(self, response: Any) -> None:
        self.response = response
        self.calls: list[dict[str, Any]] = []

    def parse(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return self.response


def fake_response(parsed: Any, stop_reason: str = "end_turn") -> SimpleNamespace:
    usage = SimpleNamespace(
        input_tokens=100,
        output_tokens=20,
        cache_read_input_tokens=50,
        cache_creation_input_tokens=None,
    )
    return SimpleNamespace(
        parsed_output=parsed, stop_reason=stop_reason, stop_details=None, usage=usage
    )


def make_claude(response: Any) -> tuple[ClaudeStructured, FakeMessages]:
    messages = FakeMessages(response)
    client: Any = SimpleNamespace(messages=messages)
    return ClaudeStructured("claude-opus-5-5", JudgeVerdict, "low", client=client), messages


MESSAGES: list[BaseMessage] = [
    SystemMessage(content="be strict"),
    HumanMessage(
        content=[
            {"type": "text", "text": "keep?"},
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,QUJD"}},
        ]
    ),
]


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        ("claude:claude-opus-5-5", ("claude", "claude-opus-5-5")),
        ("bedrock:claude-haiku-5-5", ("bedrock", "claude-haiku-5-5")),
        ("openai:gpt-5", ("openai", "gpt-5")),
        ("ollama:qwen2.5vl:7b", ("ollama", "qwen2.5vl:7b")),
    ],
)
def test_parse_spec(spec: str, expected: tuple[str, str]) -> None:
    assert parse_spec(spec) == expected


def test_to_anthropic_converts_system_text_and_image() -> None:
    system, content = to_anthropic(MESSAGES)
    assert system == [{"type": "text", "text": "be strict", "cache_control": {"type": "ephemeral"}}]
    assert content == [
        {"type": "text", "text": "keep?"},
        {
            "type": "image",
            "source": {"type": "base64", "media_type": "image/jpeg", "data": "QUJD"},
        },
    ]


def test_claude_structured_returns_parsed_output_and_tallies_usage() -> None:
    verdict = JudgeVerdict(keep=True, reason="clear logo")
    claude, messages = make_claude(fake_response(verdict))

    assert claude(MESSAGES) == verdict
    claude(MESSAGES)

    call: dict[str, Any] = messages.calls[0]
    assert call["output_format"] is JudgeVerdict
    assert call["output_config"] == {"effort": "low"}
    assert claude.usage["requests"] == 2
    assert claude.usage["input_tokens"] == 200
    assert claude.usage["cache_read_input_tokens"] == 100


def test_claude_structured_raises_on_refusal() -> None:
    claude, _ = make_claude(fake_response(None, stop_reason="refusal"))
    with pytest.raises(RuntimeError, match="refused"):
        claude(MESSAGES)


def test_claude_structured_raises_without_output() -> None:
    claude, _ = make_claude(fake_response(None, stop_reason="max_tokens"))
    with pytest.raises(ValueError, match="max_tokens"):
        claude(MESSAGES)


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("claude-opus-5-5", "anthropic.claude-opus-5-5"),
        ("anthropic.claude-opus-5-5", "anthropic.claude-opus-5-5"),
        ("us.anthropic.claude-opus-5-5", "us.anthropic.claude-opus-5-5"),
    ],
)
def test_bedrock_model_id(model: str, expected: str) -> None:
    assert bedrock_model_id(model) == expected


def test_build_chain_bedrock_uses_mantle_client(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AWS_REGION", "us-east-1")
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "test-key")

    chain = build_chain("bedrock:claude-opus-5-5", JudgeVerdict)

    assert isinstance(chain, RunnableLambda)
    claude = chain.func
    assert isinstance(claude, ClaudeStructured)
    assert claude.model == "anthropic.claude-opus-5-5"
    assert isinstance(claude.client, anthropic.AnthropicBedrockMantle)


def test_bedrock_without_api_key_needs_boto3(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AWS_REGION", "us-east-1")
    for name in llm.BEDROCK_API_KEY_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(importlib.util, "find_spec", lambda _name: None)

    with pytest.raises(RuntimeError, match="boto3"):
        build_chain("bedrock:claude-opus-5-5", JudgeVerdict)


def openai_result(parsed: Any, refusal: str | None = None) -> dict[str, Any]:
    raw = SimpleNamespace(
        usage_metadata={
            "input_tokens": 300,
            "output_tokens": 40,
            "input_token_details": {"cache_read": 100},
        },
        additional_kwargs={"refusal": refusal} if refusal else {},
    )
    return {"raw": raw, "parsed": parsed, "parsing_error": None}


def test_openai_structured_returns_parsed_and_tallies_usage() -> None:
    verdict = JudgeVerdict(keep=False, reason="blurry")
    seen: list[list[BaseMessage]] = []

    def fake(messages: list[BaseMessage]) -> dict[str, Any]:
        seen.append(messages)
        return openai_result(verdict)

    judge = llm.OpenAIStructured(structured=RunnableLambda(fake))

    assert judge(MESSAGES) == verdict
    assert seen == [MESSAGES]
    assert judge.usage == {
        "requests": 1,
        "input_tokens": 300,
        "output_tokens": 40,
        "cache_read_input_tokens": 100,
    }


def test_openai_structured_raises_on_refusal_and_missing_output() -> None:
    refused = llm.OpenAIStructured(
        structured=RunnableLambda(lambda _m: openai_result(None, refusal="no"))
    )
    empty = llm.OpenAIStructured(structured=RunnableLambda(lambda _m: openai_result(None)))

    with pytest.raises(RuntimeError, match="refused"):
        refused(MESSAGES)
    with pytest.raises(ValueError, match="no structured output"):
        empty(MESSAGES)


def test_openai_without_package_explains_install(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(importlib.util, "find_spec", lambda _name: None)
    with pytest.raises(RuntimeError, match="langchain-openai"):
        build_chain("openai:gpt-5", JudgeVerdict)
