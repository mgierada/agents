import base64
import io
from pathlib import Path
from typing import Any

import structlog
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import Runnable
from langchain_ollama import ChatOllama
from PIL import Image

from cleanup.checks.image import open_image
from cleanup.judge import JudgeVerdict
from cleanup.state import ImageRecord

log = structlog.get_logger()

JPEG_QUALITY = 90

SYSTEM_PROMPT = """You are a logo quality expert. You review images for a object-detection 
    training dataset. The training images contains logos of vairous brands. Drop an image if 
    the logo is too blurry, badly exposed, heavily occluded and tiny Drop the image if it corrupted 
    and mostly empty, or otherwise unlikely to help a detector learn logo objects. Keep it if 
    the scene and logos are reasonably visible. Answer with a keep|drop decision and one 
    sentence reason.
    """
USER_PROMPT = "Should this image be kept in the training set?"


class VlmJudge:
    """Judge that asks a vision-language model whether to keep an image."""

    def __init__(
        self,
        chain: Runnable[list[BaseMessage], Any],
        fallback_keep: bool,
        max_edge: int = 1024,
    ) -> None:
        """`chain` maps messages to a JudgeVerdict, e.g. `llm.with_structured_output(...)`."""
        self.chain = chain
        self.fallback_keep = fallback_keep
        self.max_edge = max_edge

    def __call__(self, record: ImageRecord) -> JudgeVerdict:
        """Judge one image. Errors fall back to `fallback_keep` instead of aborting the run."""
        try:
            result = self.chain.invoke(input=build_messages(record.image_path, self.max_edge))
            return JudgeVerdict.model_validate(result)
        except Exception as exc:  # noqa: BLE001 - one bad image or model hiccup must not abort
            log.warning("judge_error", image=str(record.image_path), error=repr(exc))
            return JudgeVerdict(keep=self.fallback_keep, reason=f"judge_error:{type(exc).__name__}")


def ollama_judge(model: str, fallback_keep: bool, max_edge: int = 1024) -> VlmJudge:
    """Build a VlmJudge backed by a local Ollama model (honours OLLAMA_HOST)."""
    vlm = ChatOllama(model=model, temperature=0)
    return VlmJudge(
        chain=vlm.with_structured_output(schema=JudgeVerdict),
        fallback_keep=fallback_keep,
        max_edge=max_edge,
    )


def build_messages(image_path: Path, max_edge: int) -> list[BaseMessage]:
    """Build the system + user messages with the image inlined as a base64 data URL."""
    data: str = encode_image(image_path, max_edge)
    return [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(
            content=[
                {"type": "text", "text": USER_PROMPT},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{data}"}},
            ]
        ),
    ]


def encode_image(path: Path, max_edge: int) -> str:
    """Load an image, downscale so its longest edge is <= max_edge, return base64 JPEG."""
    image: Image.Image = open_image(path).convert(mode="RGB")
    image.thumbnail(size=(max_edge, max_edge), resample=Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    image.save(fp=buffer, format="JPEG", quality=JPEG_QUALITY)
    return base64.b64encode(s=buffer.getvalue()).decode(encoding="ascii")
