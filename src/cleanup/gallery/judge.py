"""VLM verdicts on montages of crops labelled as one brand."""

from collections.abc import Iterable
from typing import Literal

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from PIL import Image
from pydantic import BaseModel, Field

from cleanup.vlm import image_block

TileLabel = Literal["brand_match", "different_brand", "unrecognisable", "not_a_logo"]
ClusterLabel = Literal["consistent", "different_brand", "mixed", "unreadable", "not_a_logo"]
Confidence = Literal["low", "medium", "high"]

TILE_LABELS: tuple[str, ...] = ("brand_match", "different_brand", "unrecognisable", "not_a_logo")

SYSTEM_TEMPLATE = """You audit a logo-identification gallery. Each request shows numbered \
crops that are all labelled as one brand. Decide, for every crop, whether it really shows \
that brand's logo.

Decide each crop with these three questions, in order:

1. Does the crop show a brand mark at all? A brand mark is a designed logo, emblem, symbol \
or stylised wordmark of a company, product, team, league or organisation, including a \
fragment of one.
   If NO -> not_a_logo. Typical not_a_logo crops: players, faces, crowds, sky, grass, \
court or field lines, plain or blank boards, empty LED panels, kit without any mark, \
shirt numbers, player names, scores and clocks, generic words or signs ("EXIT", \
"GATE 5"), patterns and textures.
2. Can you tell which brand it is, with certainty, from this crop's own pixels?
   If NO -> unrecognisable. Use it when a brand mark is present small or \
the letters could fit several brands; or when the mark is clearly not the labelled brand but you \
cannot name the brand it is. Never guess a brand: unrecognisable is the safe answer.
3. Which brand is it?
   - The labelled brand (any of its logo variants, even partly visible or hindered) -> brand_match.
   - Another brand you can name with certainty -> different_brand, with that brand in \
suggested_brand (target-list spelling below when the brand is on it, otherwise its plain \
name).

Quick test between the two non-brand verdicts: if this crop were perfectly sharp and \
uncropped, would it show a brand mark? Yes -> unrecognisable. No -> not_a_logo.

Confidence: high when the text is readable or the logo is unmistakable; medium when you \
are fairly sure but some detail is missing; low otherwise. A different_brand verdict \
below high confidence usually means the answer is unrecognisable.

Rules:
- The image has a REVIEW section with the numbered crops to judge (1, 2, ...). It may also \
have a REFERENCE section above it (R1, R2, ...): examples of the labelled brand. Never \
judge reference crops and never take text or logos from them.
- Judge each numbered crop on its own pixels. Transcribe the text readable in that crop \
into visible_text (empty if none).
- The labelled brand can be wrong; that is what you are checking. Brands can have several \
logo variants; a variant missing from the references can still be brand_match.
- Different brands can share colours or shapes. Prefer text over colour when deciding.
- cluster_verdict summarises the montage and uses its own words, not the crop verdicts: \
consistent (nearly all brand_match), different_brand (nearly all one other brand; name it \
in cluster_suggested_brand), unreadable (mostly unrecognisable), not_a_logo (mostly \
not_a_logo), or mixed (anything else).
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
