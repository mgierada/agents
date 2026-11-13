"""Numbered crop montages: the unit a VLM reviews in one call.

References and crops under review go into one image with labelled sections, so the judge
cannot mix them up. Some backends (Ollama) drop the order of text and images in a message,
so two separate images could not be told apart.
"""

import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from cleanup.checks.image import open_image

BACKGROUND = (255, 255, 255)
GAP_PX = 4
LABEL_FONT_PX = 18
LABEL_BOX = (0, 0, 0)
LABEL_TEXT = (255, 255, 0)
BANNER_FONT_PX = 20
BANNER_PAD_PX = 6
REFERENCE_BANNER = (90, 90, 90)
REVIEW_BANNER = (20, 70, 180)
BANNER_TEXT = (255, 255, 255)
DIVIDER_PX = 12


def review_sheet(
    crops: list[Path], references: list[Path], brand: str, tile_px: int, cols: int
) -> Image.Image:
    """One image: an optional REFERENCE section (R1, R2, ...; not judged) above a divider,
    then the REVIEW section with the crops to judge, numbered from 1."""
    review: Image.Image = montage(crops, tile_px, cols)
    sections: list[Image.Image] = []
    if references:
        refs: Image.Image = montage(references, tile_px, len(references), prefix="R")
        sections += [
            banner(f"REFERENCE: examples of {brand} - do not judge", refs.width, REFERENCE_BANNER),
            refs,
            Image.new("RGB", (1, DIVIDER_PX), BACKGROUND),
        ]
    sections += [
        banner(f"REVIEW: judge {numbering(len(crops))}", review.width, REVIEW_BANNER),
        review,
    ]
    return stack(sections)


def numbering(n: int) -> str:
    """'crop 1' or 'crops 1-n'."""
    return "crop 1" if n == 1 else f"crops 1-{n}"


def banner(text: str, min_width: int, color: tuple[int, int, int]) -> Image.Image:
    """Solid bar with white text, at least `min_width` wide."""
    font = ImageFont.load_default(size=BANNER_FONT_PX)
    left, top, right, bottom = (
        round(v) for v in ImageDraw.Draw(Image.new("RGB", (1, 1))).textbbox((0, 0), text, font=font)
    )
    width: int = max(min_width, right - left + 2 * BANNER_PAD_PX)
    bar = Image.new("RGB", (width, bottom - top + 2 * BANNER_PAD_PX), color)
    ImageDraw.Draw(bar).text(
        (BANNER_PAD_PX, BANNER_PAD_PX - top), text, fill=BANNER_TEXT, font=font
    )
    return bar


def stack(images: list[Image.Image]) -> Image.Image:
    """Stack images vertically, left-aligned on a white canvas."""
    canvas = Image.new(
        "RGB", (max(i.width for i in images), sum(i.height for i in images)), BACKGROUND
    )
    y: int = 0
    for image in images:
        canvas.paste(image, (0, y))
        y += image.height
    return canvas


def montage(paths: list[Path], tile_px: int, cols: int, prefix: str = "") -> Image.Image:
    """Grid of letterboxed tiles, each stamped `<prefix><n>` with n starting at 1."""
    cols = max(1, min(cols, len(paths)))
    rows: int = math.ceil(len(paths) / cols)
    step: int = tile_px + GAP_PX
    canvas = Image.new("RGB", (cols * step - GAP_PX, rows * step - GAP_PX), BACKGROUND)
    for i, path in enumerate(paths):
        tile: Image.Image = letterbox(open_image(path).convert("RGB"), tile_px)
        stamp(tile, f"{prefix}{i + 1}")
        canvas.paste(tile, ((i % cols) * step, (i // cols) * step))
    return canvas


def letterbox(image: Image.Image, size: int) -> Image.Image:
    """Fit an image into a size x size square (up- or downscaled), padded with white."""
    scale: float = size / max(image.size)
    resized: Image.Image = image.resize(
        (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
        resample=Image.Resampling.LANCZOS,
    )
    tile = Image.new("RGB", (size, size), BACKGROUND)
    tile.paste(resized, ((size - resized.width) // 2, (size - resized.height) // 2))
    return tile


def stamp(tile: Image.Image, text: str) -> None:
    """Draw a high-contrast label in the tile's top-left corner (in place)."""
    draw = ImageDraw.Draw(tile)
    font = ImageFont.load_default(size=LABEL_FONT_PX)
    left, top, right, bottom = draw.textbbox((2, 0), text, font=font)
    draw.rectangle((0, 0, right + 2, bottom + 2), fill=LABEL_BOX)
    draw.text((2, 0), text, fill=LABEL_TEXT, font=font)
