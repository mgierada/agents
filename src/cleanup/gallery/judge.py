"""VLM verdicts on montages of crops labelled as one brand."""

from collections.abc import Iterable
from typing import Literal

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from PIL import Image
from pydantic import BaseModel, Field

from cleanup.vlm import image_block

TileLabel = Literal["brand_match", "different_brand", "unrecognisable", "not_a_logo"]
ClusterLabel = Literal["consistent", "different_brand", "mixed", "unreadable"]
Confidence = Literal["low", "medium", "high"]

TILE_LABELS: tuple[str, ...] = ("brand_match", "different_brand", "unrecognisable", "not_a_logo")

SYSTEM_TEMPLATE = """You audit a logo-identification gallery. Each request shows numbered \
crops that are all labelled as one brand. Decide, for every crop, whether it really shows \
that brand's logo.

Per crop verdict:
- brand_match: the labelled brand's logo or wordmark is visible, even if partly occluded, \
small or blurry, as long as it is recognisable as that brand.
- different_brand: a recognisable logo of another brand. Name that brand in suggested_brand, \
using the target-list spelling below when the brand is on it, otherwise its plain name.
- unrecognisable: something logo-like, but too blurry, small, cropped or occluded to tell \
which brand.
- not_a_logo: no logo at all (background, person, random text, empty).

Rules:
- The image has a REVIEW section with the numbered crops to judge (1, 2, ...). It may also \
have a REFERENCE section above it (R1, R2, ...): examples of the labelled brand. Never \
judge reference crops and never take text or logos from them.
- Judge each numbered crop on its own pixels. Transcribe the text readable in that crop \
into visible_text (empty if none).
- The labelled brand can be wrong; that is what you are checking. Brands can have several \
logo variants; a variant missing from the references can still be brand_match.
- Different brands can share colours or shapes. Prefer text over colour when deciding.
- cluster_verdict summarises the montage: consistent (nearly all brand_match), \
different_brand (nearly all one other brand; name it in cluster_suggested_brand), mixed, \
or unreadable (mostly unrecognisable).
- Return exactly one tile entry per numbered crop.

Target list (brand names, one per line):
{targets}
"""


class TileVerdict(BaseModel):
    """Verdict on one numbered crop of a montage."""

    tile: int = Field(description="Crop number as stamped on the montage, starting at 1")
    verdict: TileLabel
    visible_text: str = Field(description="Text readable in the crop, empty if none")
    suggested_brand: str = Field(
        description="Brand actually shown when verdict is different_brand, else empty"
    )
    confidence: Confidence


class MontageVerdict(BaseModel):
    """Verdicts on every crop of a montage plus a montage-level summary."""

    cluster_verdict: ClusterLabel
    cluster_suggested_brand: str = Field(
        description="The other brand when cluster_verdict is different_brand, else empty"
    )
    tiles: list[TileVerdict]


def system_prompt(targets: Iterable[str]) -> str:
    """System prompt with the sorted target list; identical for every call of a run."""
    return SYSTEM_TEMPLATE.format(targets="\n".join(sorted(targets)))


def build_messages(system: str, brand: str, sheet: Image.Image, n_tiles: int) -> list[BaseMessage]:
    """System prompt plus one user turn holding the review sheet (see montage.review_sheet)."""
    text: str = (
        f"Labelled brand: {brand}\n"
        f"Judge the {n_tiles} numbered crop(s) in the REVIEW section. "
        "The REFERENCE section, if present, only shows what the labelled brand looks like."
    )
    content: list[str | dict[str, object]] = [
        {"type": "text", "text": text},
        image_block(sheet),
    ]
    return [SystemMessage(content=system), HumanMessage(content=content)]  # type: ignore[arg-type]
