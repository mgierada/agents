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
        (
            "bedrock-converse:qwen.qwen3-vl-235b-a22b",
            ("bedrock-converse", "qwen.qwen3-vl-235b-a22b"),
        ),
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
    monkeypatch.delenv("AWS_PROFILE", raising=False)

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


def test_bedrock_profile_is_passed_explicitly_over_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}
    monkeypatch.setenv("AWS_PROFILE", "zoomph")
    monkeypatch.setenv("AWS_REGION", "eu-west-1")
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "test-key")
    monkeypatch.setattr(importlib.util, "find_spec", lambda _name: object())
    monkeypatch.setattr(anthropic, "AnthropicBedrockMantle", lambda **kw: seen.update(kw))

    llm.bedrock_client()

    assert seen == {"aws_profile": "zoomph", "aws_region": "eu-west-1"}


def test_bedrock_profile_needs_boto3_even_with_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AWS_PROFILE", "zoomph")
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "test-key")
    monkeypatch.setattr(importlib.util, "find_spec", lambda _name: None)

    with pytest.raises(RuntimeError, match="boto3"):
        llm.bedrock_client()


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

    judge = llm.LangChainStructured("openai", structured=RunnableLambda(fake))

    assert judge(MESSAGES) == verdict
    assert seen == [MESSAGES]
    assert judge.usage == {
        "requests": 1,
        "input_tokens": 300,
        "output_tokens": 40,
        "cache_read_input_tokens": 100,
        "cache_creation_input_tokens": 0,
    }


def test_openai_structured_raises_on_refusal_and_missing_output() -> None:
    refused = llm.LangChainStructured(
        "openai", structured=RunnableLambda(lambda _m: openai_result(None, refusal="no"))
    )
    empty = llm.LangChainStructured(
        "openai", structured=RunnableLambda(lambda _m: openai_result(None))
    )

    with pytest.raises(RuntimeError, match="refused"):
        refused(MESSAGES)
    with pytest.raises(ValueError, match="no structured output"):
        empty(MESSAGES)


def test_openai_without_package_explains_install(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(importlib.util, "find_spec", lambda _name: None)
    with pytest.raises(RuntimeError, match="langchain-openai"):
        build_chain("openai:gpt-5", JudgeVerdict)


class FakeConverse:
    """Stands in for ChatBedrockConverse; records constructor args and the messages sent."""

    seen: dict[str, Any] = {}

    def __init__(self, **kwargs: Any) -> None:
        FakeConverse.seen = dict(kwargs)

    def with_structured_output(self, **kwargs: Any) -> RunnableLambda:
        FakeConverse.seen["structured"] = kwargs
        return RunnableLambda(self.respond)

    def respond(self, messages: list[BaseMessage]) -> dict[str, Any]:
        FakeConverse.seen["messages"] = messages
        return openai_result(JudgeVerdict(keep=True, reason="ok"))


@pytest.fixture
def converse(monkeypatch: pytest.MonkeyPatch) -> type[FakeConverse]:
    monkeypatch.setenv("AWS_PROFILE", "zoomph")
    monkeypatch.setenv("AWS_REGION", "us-east-1")
    monkeypatch.setattr("langchain_aws.ChatBedrockConverse", FakeConverse)
    return FakeConverse


def test_build_chain_bedrock_converse_uses_profile_and_region(
    converse: type[FakeConverse],
) -> None:
    chain = build_chain("bedrock-converse:qwen.qwen3-vl-235b-a22b", JudgeVerdict)

    assert chain.invoke(MESSAGES) == JudgeVerdict(keep=True, reason="ok")
    seen: dict[str, Any] = converse.seen
    assert seen["model"] == "qwen.qwen3-vl-235b-a22b"
    assert (seen["credentials_profile_name"], seen["region_name"]) == ("zoomph", "us-east-1")
    assert seen["structured"] == {"schema": JudgeVerdict, "include_raw": True}
    assert seen["messages"] == MESSAGES  # no cache point: Qwen rejects it
    assert seen["temperature"] == 0


def test_bedrock_converse_caches_system_prompt_for_claude(converse: type[FakeConverse]) -> None:
    chain = build_chain("bedrock-converse:global.anthropic.claude-haiku-5-5", JudgeVerdict)

    chain.invoke(MESSAGES)

    system, human = converse.seen["messages"]
    assert system.content == [{"type": "text", "text": "be strict"}, llm.CONVERSE_CACHE_POINT]
    assert human == MESSAGES[1]
    assert converse.seen["temperature"] is None
    assert MESSAGES[0].content == "be strict"  # input not mutated


def test_langchain_structured_tallies_cache_writes() -> None:
    result: dict[str, Any] = openai_result(JudgeVerdict(keep=True, reason="ok"))
    result["raw"].usage_metadata["input_token_details"] = {
        "cache_read": 0,
        "cache_creation": 0,
        "ephemeral_5m_input_tokens": 900,
    }
    judge = llm.LangChainStructured(
        "bedrock-converse", structured=RunnableLambda(lambda _m: result)
    )

    judge(MESSAGES)

    assert judge.usage["cache_creation_input_tokens"] == 900
