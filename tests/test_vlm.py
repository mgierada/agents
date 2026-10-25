import base64
import io
from pathlib import Path

import numpy as np
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.runnables import RunnableLambda
from PIL import Image

from cleanup.judge import JudgeVerdict
from cleanup.state import ImageRecord
from cleanup.vlm import VlmJudge, build_messages, encode_image
from tests.conftest import write_image


def make_record(path: Path) -> ImageRecord:
    return ImageRecord(image_path=path, label_path=path.with_suffix(".txt"), rel_path=path)


def decode(data: str) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(base64.b64decode(data))))


def test_encode_image_downscales_longest_edge(tmp_path: Path) -> None:
    path: Path = write_image(tmp_path / "big.png", np.zeros((200, 400, 3), dtype=np.uint8))
    assert decode(encode_image(path, max_edge=100)).shape[:2] == (50, 100)


def test_encode_image_keeps_small_images(tmp_path: Path) -> None:
    path: Path = write_image(tmp_path / "small.png", np.zeros((40, 60, 3), dtype=np.uint8))
    assert decode(encode_image(path, max_edge=100)).shape[:2] == (40, 60)


def test_build_messages_inlines_image(tmp_path: Path, sharp_image: np.ndarray) -> None:
    path: Path = write_image(tmp_path / "a.png", sharp_image)
    human = build_messages(path, max_edge=64)[-1]
    assert isinstance(human, HumanMessage)
    image_block = human.content[-1]
    assert isinstance(image_block, dict)
    assert image_block["image_url"]["url"].startswith("data:image/jpeg;base64,")


def test_vlm_judge_returns_model_verdict(tmp_path: Path, sharp_image: np.ndarray) -> None:
    path: Path = write_image(tmp_path / "a.png", sharp_image)
    seen: list[list[BaseMessage]] = []

    def fake_model(messages: list[BaseMessage]) -> JudgeVerdict:
        seen.append(messages)
        return JudgeVerdict(keep=False, reason="too dark")

    judge = VlmJudge(RunnableLambda(fake_model), fallback_keep=True)

    assert judge(make_record(path)) == JudgeVerdict(keep=False, reason="too dark")
    assert len(seen) == 1


def test_vlm_judge_falls_back_on_error(tmp_path: Path) -> None:
    judge = VlmJudge(RunnableLambda(lambda _m: None), fallback_keep=True)
    verdict: JudgeVerdict = judge(make_record(tmp_path / "missing.png"))
    assert verdict == JudgeVerdict(keep=True, reason="judge_error:FileNotFoundError")
