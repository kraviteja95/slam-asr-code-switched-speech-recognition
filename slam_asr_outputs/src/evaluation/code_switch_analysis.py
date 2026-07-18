"""Code-switching analytics for the qualitative section of the report.

* :func:`code_switch_stats` — counts how many code-switch boundaries a
  reference contains, useful when partitioning errors by "monolingual"
  vs "heavily code-switched" utterances.
* :func:`confusion_by_script` — cross-tabulates errors by reference /
  hypothesis script class so we can see e.g. how often the model emits
  Devanagari when the reference is English.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Dict, List, Tuple

DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")
ROMAN_RE = re.compile(r"[A-Za-z]")


def _script_of(word: str) -> str:
    if DEVANAGARI_RE.search(word):
        return "hi"
    if ROMAN_RE.search(word):
        return "en"
    return "other"


def code_switch_stats(text: str) -> Dict[str, float]:
    """Return code-switching descriptors for a single reference string.

    Keys
    ----
    n_words:
        Total token count.
    n_switches:
        Number of adjacent-token language changes.
    switch_rate:
        ``n_switches / max(n_words - 1, 1)``.
    """
    tokens = text.split()
    if len(tokens) < 2:
        return {"n_words": len(tokens), "n_switches": 0, "switch_rate": 0.0}
    scripts = [_script_of(t) for t in tokens]
    switches = sum(
        1 for a, b in zip(scripts[:-1], scripts[1:])
        if a in ("hi", "en") and b in ("hi", "en") and a != b
    )
    return {
        "n_words": len(tokens),
        "n_switches": switches,
        "switch_rate": switches / (len(tokens) - 1),
    }


def confusion_by_script(refs: List[str], hyps: List[str]) -> Dict[Tuple[str, str], int]:
    """Rough token-level confusion counts keyed by ``(ref_script, hyp_script)``.

    Uses a naive positional alignment — good enough for the report's error
    breakdown chart; use :func:`per_language_wer` for exact WER.
    """
    counts: Counter[Tuple[str, str]] = Counter()
    for ref, hyp in zip(refs, hyps):
        r_toks = ref.split()
        h_toks = hyp.split()
        for i in range(max(len(r_toks), len(h_toks))):
            r = r_toks[i] if i < len(r_toks) else ""
            h = h_toks[i] if i < len(h_toks) else ""
            counts[(_script_of(r), _script_of(h))] += 1
    return dict(counts)
