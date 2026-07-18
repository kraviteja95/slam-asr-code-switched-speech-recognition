"""Text normalisation for Hindi-English code-switched transcripts.

WER numbers can swing wildly depending on how aggressively you normalise the
reference and hypothesis strings.  For the MUCS challenge the community
standard is:

1. Unicode NFC normalisation (so ``क + ि`` matches ``कि``).
2. Lower-case Roman characters; leave Devanagari unchanged.
3. Convert Devanagari digits to Roman digits (``१२३`` → ``123``).
4. Remove punctuation and control characters.
5. Collapse whitespace.
6. Optionally strip Hindi/English filler words (``uh``, ``um``, ``अं``).

The normaliser is *deterministic* and *idempotent* — running it twice
produces the same output as running it once.  This is essential for
reproducible WER reporting.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Iterable, List

# Devanagari digits 0-9  →  Roman digits 0-9
_DEVANAGARI_DIGIT_MAP = str.maketrans("०१२३४५६७८९", "0123456789")

# Characters that we treat as punctuation / non-linguistic decoration.
# We keep the ASCII hyphen because "e-mail" or "c-plus-plus" is a token
# in this technical corpus, but we replace it with a space so WER counts
# each half separately (matches the challenge convention).
_PUNCTUATION_RE = re.compile(
    r"[!\"#\$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~।॥\u2013\u2014\u2018\u2019\u201c\u201d]"
)

# Extra whitespace collapse
_WHITESPACE_RE = re.compile(r"\s+")

# Common filler tokens (Hindi + English)
_DEFAULT_FILLERS = {
    "uh", "um", "hmm", "mm", "mhm", "err", "ah", "eh",
    "अं", "आं", "ओं", "उम", "एर", "अरे",
}


@dataclass
class CodeSwitchTextNormalizer:
    """Idempotent text normaliser for MUCS Hindi-English transcripts.

    Parameters
    ----------
    lowercase_roman:
        Lower-case ASCII / Latin characters.  Devanagari has no case.
    map_devanagari_digits:
        Replace ``०–९`` with ``0–9``.
    remove_punctuation:
        Strip characters listed in :data:`_PUNCTUATION_RE`.
    remove_fillers:
        Drop tokens matching :attr:`fillers` (default: common HI + EN fillers).
    fillers:
        Iterable of tokens to remove (matched after tokenisation).  Set to
        ``None`` to use the default list; pass an empty set to disable.
    """

    lowercase_roman: bool = True
    map_devanagari_digits: bool = True
    remove_punctuation: bool = True
    remove_fillers: bool = False
    fillers: Iterable[str] | None = None

    def __post_init__(self) -> None:
        self._fillers = set(self.fillers) if self.fillers is not None else _DEFAULT_FILLERS

    # ---------------------------------------------------------------- API
    def __call__(self, text: str) -> str:
        """Normalise a single string."""
        return self.normalize(text)

    def normalize(self, text: str) -> str:
        """Return the canonicalised form of *text*."""
        if text is None:
            return ""
        # 1) Unicode canonical composition — combining marks + base characters.
        text = unicodedata.normalize("NFC", text)

        # 2) Roman lower-case (leaves Devanagari alone).
        if self.lowercase_roman:
            text = text.lower()

        # 3) Devanagari digits -> Roman digits.
        if self.map_devanagari_digits:
            text = text.translate(_DEVANAGARI_DIGIT_MAP)

        # 4) Punctuation stripping.
        if self.remove_punctuation:
            text = _PUNCTUATION_RE.sub(" ", text)

        # 5) Tokenise, optionally remove fillers.
        tokens = text.split()
        if self.remove_fillers and self._fillers:
            tokens = [t for t in tokens if t not in self._fillers]

        # 6) Collapse whitespace.
        return _WHITESPACE_RE.sub(" ", " ".join(tokens)).strip()

    def batch_normalize(self, texts: Iterable[str]) -> List[str]:
        """Normalise many strings; convenience wrapper."""
        return [self.normalize(t) for t in texts]


# ------------------------- Quick self-test -------------------------
if __name__ == "__main__":  # pragma: no cover
    norm = CodeSwitchTextNormalizer(remove_fillers=True)
    samples = [
        "Meet the GIMP के Spoken Tutorial में आपका स्वागत है।",
        "अध्याय १२३ - Introduction to Python?",
        "uh yeh tutorial hai... um okay",
    ]
    for s in samples:
        print(f"{s!r}\n  -> {norm(s)!r}")
