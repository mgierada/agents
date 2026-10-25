"""Pluggable judges that resolve borderline images to keep/drop."""

from typing import Literal, Protocol

from cleanup.state import ImageRecord


class Judge(Protocol):
    """Decides whether a borderline image should be kept."""

    def __call__(self, record: ImageRecord) -> bool: ...


def policy_judge(policy: Literal["keep", "drop"]) -> Judge:
    """Judge that applies a fixed policy to every borderline image."""
    keep: bool = policy == "keep"
    return lambda _record: keep


# TODO: add an LLM/VLM judge, e.g. built with `langchain.chat_models.init_chat_model`,
# sending the base64 image and asking for a structured keep/drop verdict.
