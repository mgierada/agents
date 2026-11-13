"""VLM backends behind one interface: LangChain messages in, a validated pydantic model out.

A judge spec selects the backend:
    claude:<model>    Anthropic API (ANTHROPIC_API_KEY or an `ant auth login` profile)
    bedrock:<model>   Claude on Amazon Bedrock (AWS_REGION plus AWS credentials or a Bedrock
                      API key in AWS_BEARER_TOKEN_BEDROCK); `claude-*` ids get the `anthropic.`
                      prefix Bedrock uses, anything else (inference profile, ARN) is passed as is
    openai:<model>    OpenAI (OPENAI_API_KEY; OPENAI_BASE_URL for OpenAI-compatible servers);
                      needs `uv add langchain-openai`
    ollama:<model>    local Ollama
"""

import importlib.util
import os
import threading
from collections import Counter
from typing import Any, Literal, cast

import anthropic
from anthropic.types import (
    ContentBlockParam,
    ImageBlockParam,
    MessageParam,
    ParsedMessage,
    TextBlockParam,
)
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import Runnable, RunnableLambda
from langchain_ollama import ChatOllama
from pydantic import BaseModel

from cleanup.errors import UnknownProviderError

Provider = Literal["claude", "bedrock", "openai", "ollama"]
Effort = Literal["low", "medium", "high"]
ImageMediaType = Literal["image/jpeg", "image/png", "image/gif", "image/webp"]

PROVIDERS: tuple[Provider, ...] = ("claude", "bedrock", "openai", "ollama")
BEDROCK_MODEL_PREFIX = "anthropic."
# Bedrock API-key env vars; without one the client signs requests with SigV4 via boto3.
BEDROCK_API_KEY_ENV_VARS: tuple[str, ...] = ("AWS_BEARER_TOKEN_BEDROCK", "ANTHROPIC_AWS_API_KEY")
DATA_URL_PREFIX = "data:"
CLAUDE_MAX_TOKENS = 16000


def parse_spec(spec: str) -> tuple[Provider, str]:
    """Split provider-model string into subcomponents `claude:claude-opus-5-5`
    into (provider, model).

    Example:
        `claude:claude-opus-5-5` -> ("claude, claude-opus-5-5")

    """
    provider, _, model = spec.partition(":")
    if provider not in PROVIDERS:
        raise UnknownProviderError(f"Provider {provider} is not supported.")
    return provider, model


def build_chain(
    spec: str, schema: type[BaseModel], effort: Effort = "low"
) -> Runnable[list[BaseMessage], Any]:
    """Chain mapping LangChain messages to an instance of `schema` for the given judge spec."""
    provider, model = parse_spec(spec)
    if provider == "claude":
        return RunnableLambda(ClaudeStructured(model=model, schema=schema, effort=effort))
    if provider == "bedrock":
        return RunnableLambda(
            ClaudeStructured(
                model=bedrock_model_id(model),
                schema=schema,
                effort=effort,
                client=bedrock_client(),
            )
        )
    if provider == "openai":
        return RunnableLambda(OpenAIStructured(structured=openai_structured(model, schema)))
    return ChatOllama(model=model, temperature=0).with_structured_output(schema=schema)


def openai_structured(model: str, schema: type[BaseModel]) -> Runnable[list[BaseMessage], Any]:
    """ChatOpenAI with strict JSON-schema output, returning raw message and parsed model.

    Imported lazily: langchain-openai is only needed for `openai:` judges. No temperature or
    reasoning effort is set, since reasoning and non-reasoning models accept different ones.
    """
    if importlib.util.find_spec("langchain_openai") is None:
        raise RuntimeError("openai judges need langchain-openai: `uv add langchain-openai`")
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(model=model).with_structured_output(
        schema=schema, method="json_schema", strict=True, include_raw=True
    )


def bedrock_model_id(model: str) -> str:
    """`claude-opus-5-5` -> `anthropic.claude-opus-5-5`; other ids (profiles, ARNs) unchanged."""
    return BEDROCK_MODEL_PREFIX + model if model.startswith("claude-") else model


def bedrock_client() -> anthropic.AnthropicBedrockMantle:
    """Bedrock client from the environment (AWS_REGION, AWS_PROFILE / keys, or API key).

    Fails fast when SigV4 signing is needed but boto3 is missing, instead of failing on every
    request of a run.
    """
    has_api_key: bool = any(os.environ.get(name) for name in BEDROCK_API_KEY_ENV_VARS)
    if not has_api_key and importlib.util.find_spec("boto3") is None:
        raise RuntimeError(
            "bedrock with AWS credentials needs boto3: `uv add 'anthropic[bedrock]'`, "
            "or set AWS_BEARER_TOKEN_BEDROCK to a Bedrock API key"
        )
    return anthropic.AnthropicBedrockMantle()


class UsageTally:
    """Thread-safe running token-usage tally of a judge backend."""

    def __init__(self) -> None:
        """Start an empty tally."""
        self.usage: Counter[str] = Counter()
        self._lock = threading.Lock()

    def record(self, **counts: int) -> None:
        """Add one request's token counts."""
        with self._lock:
            self.usage.update(requests=1, **counts)


class OpenAIStructured(UsageTally):
    """Calls an OpenAI chat model with structured output and tallies token usage."""

    def __init__(self, structured: Runnable[list[BaseMessage], dict[str, Any]]) -> None:
        """`structured` is a chat model's `with_structured_output(..., include_raw=True)`."""
        super().__init__()
        self.structured = structured

    def __call__(self, messages: list[BaseMessage]) -> BaseModel:
        """Send the messages; raise on refusal or when no valid output came back."""
        result: dict[str, Any] = self.structured.invoke(messages)
        raw: Any = result["raw"]
        usage: dict[str, Any] = getattr(raw, "usage_metadata", None) or {}
        self.record(
            input_tokens=usage.get("input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0),
            cache_read_input_tokens=usage.get("input_token_details", {}).get("cache_read", 0),
        )
        refusal: str | None = getattr(raw, "additional_kwargs", {}).get("refusal")
        if refusal:
            raise RuntimeError(f"openai refused: {refusal}")
        if result["parsed"] is None:
            raise ValueError(f"no structured output: {result.get('parsing_error')!r}")
        return result["parsed"]


class ClaudeStructured(UsageTally):
    """Calls Claude with structured output and keeps a running token-usage tally."""

    def __init__(
        self,
        model: str,
        schema: type[BaseModel],
        effort: Effort,
        client: anthropic.Anthropic | anthropic.AnthropicBedrockMantle | None = None,
    ) -> None:
        """`client` defaults to `anthropic.Anthropic()` (ANTHROPIC_API_KEY or `ant` profile)."""
        super().__init__()
        self.model = model
        self.schema = schema
        self.effort = effort
        self.client = client or anthropic.Anthropic()

    def __call__(self, messages: list[BaseMessage]) -> BaseModel:
        """Send the messages; raise on refusal or when no valid output came back."""
        system, content = to_anthropic(messages)
        user: MessageParam = {"role": "user", "content": content}
        response: ParsedMessage[Any] = self.client.messages.parse(
            model=self.model,
            max_tokens=CLAUDE_MAX_TOKENS,
            system=system,
            messages=[user],
            output_format=self.schema,
            output_config={"effort": self.effort},
        )
        self.record_usage(response)
        if response.stop_reason == "refusal":
            raise RuntimeError(f"claude refused: {response.stop_details}")
        if response.parsed_output is None:
            raise ValueError(f"no structured output, stop_reason={response.stop_reason}")
        return response.parsed_output

    def record_usage(self, response: ParsedMessage[Any]) -> None:
        """Add the response's token counts to the tally (thread-safe)."""
        usage = response.usage
        self.record(
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cache_read_input_tokens=usage.cache_read_input_tokens or 0,
            cache_creation_input_tokens=usage.cache_creation_input_tokens or 0,
        )


def to_anthropic(
    messages: list[BaseMessage],
) -> tuple[list[TextBlockParam], list[ContentBlockParam]]:
    """Convert system + human LangChain messages to Anthropic system and user content blocks.

    The system prompt is marked cacheable: it is identical across a run, so later requests
    read it from the prompt cache.
    """
    system: list[TextBlockParam] = [
        {"type": "text", "text": str(m.content), "cache_control": {"type": "ephemeral"}}
        for m in messages
        if isinstance(m, SystemMessage)
    ]
    content: list[ContentBlockParam] = [
        to_anthropic_block(block)
        for m in messages
        if isinstance(m, HumanMessage)
        for block in (m.content if isinstance(m.content, list) else [m.content])
    ]
    return system, content


def to_anthropic_block(block: str | dict[str, Any]) -> ContentBlockParam:
    """One LangChain content block (text or base64 image_url) as an Anthropic block."""
    if isinstance(block, str):
        return {"type": "text", "text": block}
    if block["type"] == "text":
        return {"type": "text", "text": block["text"]}
    url: str = block["image_url"]["url"]
    header, _, data = url.removeprefix(DATA_URL_PREFIX).partition(",")
    media_type = cast(ImageMediaType, header.split(";")[0])
    image: ImageBlockParam = {
        "type": "image",
        "source": {"type": "base64", "media_type": media_type, "data": data},
    }
    return image
