from typing import Literal, Protocol

from pydantic import BaseModel, Field

from cleanup.state import ImageRecord


class JudgeVerdict(BaseModel):
    """Keep/drop decision for one image."""

    keep: bool = Field(description="True if the image is usable as detector training data")
    reason: str = Field(description="One short sentence explaining the decision")


class Judge(Protocol):
    """Decides whether an image should be kept."""

    def __call__(self, record: ImageRecord) -> JudgeVerdict: ...


def policy_judge(policy: Literal["keep", "drop"]) -> Judge:
    """Judge that applies a fixed policy to every image."""
    verdict = JudgeVerdict(keep=policy == "keep", reason=f"policy:{policy}")
    return lambda _record: verdict
