"""Target-list brand names: loading, alias mapping and resolving free-text judge suggestions.

Decisions are keyed by brand name, never by class id: the gallery batch was built with an
older target list, so ids would not line up with the list a decision is applied to.
"""

import difflib
import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

# difflib ratio a suggestion needs to match a target name it does not equal exactly.
FUZZY_CUTOFF = 0.92
SEPARATORS = re.compile(r"[\s\-]+")


def normalise(name: str) -> str:
    """Target-list style name: 'Discount Tire' -> 'discount_tire'."""
    return SEPARATORS.sub("_", name.strip().lower())


@dataclass(frozen=True)
class BrandCatalog:
    """Target brand names plus the alias map applied to every incoming name."""

    targets: frozenset[str]
    aliases: Mapping[str, str]

    def canonical(self, name: str) -> str:
        """Normalised name with aliases (legacy names, spelling corrections) applied."""
        normal: str = normalise(name)
        return self.aliases.get(normal, normal)

    def resolve(self, suggested: str) -> str | None:
        """Target brand a free-text suggestion refers to, or None if it is not on the list."""
        if not suggested.strip():
            return None
        name: str = self.canonical(suggested)
        if name in self.targets:
            return name
        close: list[str] = difflib.get_close_matches(
            word=name, possibilities=self.targets, n=1, cutoff=FUZZY_CUTOFF
        )
        return close[0] if close else None


def load_catalog(
    target_list: Path | None, aliases_path: Path | None, fallback: Iterable[str]
) -> BrandCatalog:
    """Catalog from a target list (one brand per line) and an alias JSON.

    Without a target list the `fallback` names (the gallery's own brands) are the targets.
    """
    aliases: dict[str, str] = load_aliases(aliases_path)
    names: Iterable[str] = read_target_list(target_list) if target_list else fallback
    targets: frozenset[str] = frozenset(aliases.get(n, n) for n in map(normalise, names))
    return BrandCatalog(targets=targets, aliases=aliases)


def read_target_list(path: Path) -> list[str]:
    """Non-empty lines of a target list file."""
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def load_aliases(path: Path | None) -> dict[str, str]:
    """Alias JSON as a normalised {old: new} map; empty when no file is given."""
    if path is None:
        return {}
    raw: dict[str, str] = json.loads(path.read_text())
    return {normalise(old): normalise(new) for old, new in raw.items()}
