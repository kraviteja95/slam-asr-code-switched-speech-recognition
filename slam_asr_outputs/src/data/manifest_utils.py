"""Manifest loading, path rewriting, and dataset statistics.

The MUCS manifests are JSON-Lines files with one entry per utterance:

    {"audio_path": "/home/.../654444_0049FmqkWQMNW6Tc_0000.wav",
     "text": "meet the gimp के spoken tutorial में आपका स्वागत है",
     "speaker": "654444",
     "duration": 3.0,
     "cut_id": "654444_0049FmqkWQMNW6Tc_0000"}

The upstream manifests point to the dataset author's local machine, so we
provide `rewrite_manifest_paths` to relocate them to (a) the local repo
copy or (b) Kaggle-hosted paths.
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

# ---------------------------------------------------------------------------
# JSONL I/O
# ---------------------------------------------------------------------------


def load_manifest(path: str | Path) -> List[Dict[str, Any]]:
    """Load a JSONL manifest into a list of dictionaries.

    Parameters
    ----------
    path:
        Path to a ``.jsonl`` file.

    Returns
    -------
    list of dict
        One dictionary per line; malformed lines are skipped with a warning.
    """
    entries: List[Dict[str, Any]] = []
    path = Path(path)
    with path.open("r", encoding="utf-8") as fh:
        for i, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError as err:
                print(f"[manifest] skipping malformed line {i} in {path}: {err}")
    return entries


def save_manifest(entries: Iterable[Dict[str, Any]], path: str | Path) -> None:
    """Serialise an iterable of dicts back to JSONL."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for entry in entries:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Path rewriting
# ---------------------------------------------------------------------------


def _speaker_from_cut_id(cut_id: str) -> str:
    """Extract the leading speaker/session token from ``cut_id`` (before ``_``)."""
    return cut_id.split("_", 1)[0]


def rewrite_manifest_paths(
    entries: List[Dict[str, Any]],
    old_prefix: str,
    new_root: str,
    split: str,
    add_speaker_subdir: bool = True,
) -> List[Dict[str, Any]]:
    """Rewrite the ``audio_path`` field of every entry to a new root.

    The upstream MUCS manifests point at
    ``/home/puneets/suksham-mucs/mucs/data/processed/<split>/<file>.wav``.
    Our on-disk layout is
    ``<new_root>/data/processed/<split>/<speaker3>/<file>.wav`` — the extra
    ``<speaker3>`` sub-directory is the first three characters of the
    speaker id (see the local ``datasets/mucs/data/processed/train/``
    directory listing).

    Parameters
    ----------
    entries:
        List of manifest rows.
    old_prefix:
        Prefix in the current ``audio_path`` that must be replaced.
    new_root:
        Replacement root, e.g. ``datasets/mucs`` or
        ``/kaggle/input/mucs-2021-hindi-english-code-switched-speech/mucs``.
    split:
        Split name (``train`` / ``test`` / ``blindtest``) — used to insert
        the correct three-digit sub-directory.
    add_speaker_subdir:
        Whether to add the ``<first-three-digits-of-speaker>/`` intermediate
        directory. Set to ``False`` if audio is stored flat under the split.

    Returns
    -------
    list of dict
        Deep-copied entries with corrected ``audio_path`` fields.
    """
    fixed: List[Dict[str, Any]] = []
    for row in entries:
        row = dict(row)
        p = row.get("audio_path") or row.get("source") or ""
        if old_prefix and p.startswith(old_prefix):
            p = p.replace(old_prefix, new_root, 1)
        # Insert speaker three-digit sub-directory if not already present.
        if add_speaker_subdir:
            filename = Path(p).name
            spk3 = filename.split("_", 1)[0][:3]
            if f"/{split}/{spk3}/" not in p:
                # Replace .../processed/<split>/<file>.wav with .../processed/<split>/<spk3>/<file>.wav
                p = re.sub(
                    rf"/processed/{split}/(?!(?:\d{{3}})/)([^/]+\.wav)$",
                    rf"/processed/{split}/{spk3}/\1",
                    p,
                )
        if "audio_path" in row:
            row["audio_path"] = p
        if "source" in row:
            row["source"] = p
        fixed.append(row)
    return fixed


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")
ROMAN_RE = re.compile(r"[A-Za-z]")


def _script_of_word(word: str) -> str:
    """Return one of {``hi``, ``en``, ``mixed``, ``other``} for a token."""
    has_dev = bool(DEVANAGARI_RE.search(word))
    has_rom = bool(ROMAN_RE.search(word))
    if has_dev and has_rom:
        return "mixed"
    if has_dev:
        return "hi"
    if has_rom:
        return "en"
    return "other"


@dataclass
class ManifestStats:
    """Container for the human-readable statistics printed by ``manifest_statistics``."""

    n_utterances: int
    n_speakers: int
    total_duration_h: float
    mean_duration_s: float
    median_duration_s: float
    n_tokens: int
    n_unique_tokens: int
    frac_hindi_tokens: float
    frac_english_tokens: float
    frac_mixed_tokens: float
    frac_code_switched_utterances: float


def manifest_statistics(entries: List[Dict[str, Any]]) -> ManifestStats:
    """Compute descriptive statistics used in the EDA notebook and report.

    In particular, we report the fraction of utterances that are
    *code-switched* — i.e., contain at least one Hindi and at least one
    English word — which is the core motivator for this project.
    """
    speakers: set[str] = set()
    durations: List[float] = []
    all_tokens: List[str] = []
    hi_tokens = en_tokens = mx_tokens = 0
    n_cs_utts = 0

    for row in entries:
        text = row.get("text") or row.get("target") or ""
        text_nfc = unicodedata.normalize("NFC", text)
        tokens = text_nfc.split()
        all_tokens.extend(tokens)
        scripts = [_script_of_word(t) for t in tokens]
        n_hi = sum(1 for s in scripts if s == "hi")
        n_en = sum(1 for s in scripts if s == "en")
        n_mx = sum(1 for s in scripts if s == "mixed")
        hi_tokens += n_hi
        en_tokens += n_en
        mx_tokens += n_mx
        if n_hi > 0 and n_en > 0:
            n_cs_utts += 1
        speakers.add(str(row.get("speaker", "")))
        durations.append(float(row.get("duration", 0.0)))

    n_tokens = len(all_tokens)
    n_utts = len(entries)
    durations_sorted = sorted(durations)
    median = durations_sorted[len(durations_sorted) // 2] if durations_sorted else 0.0

    return ManifestStats(
        n_utterances=n_utts,
        n_speakers=len(speakers),
        total_duration_h=sum(durations) / 3600.0,
        mean_duration_s=sum(durations) / max(n_utts, 1),
        median_duration_s=median,
        n_tokens=n_tokens,
        n_unique_tokens=len(set(all_tokens)),
        frac_hindi_tokens=hi_tokens / max(n_tokens, 1),
        frac_english_tokens=en_tokens / max(n_tokens, 1),
        frac_mixed_tokens=mx_tokens / max(n_tokens, 1),
        frac_code_switched_utterances=n_cs_utts / max(n_utts, 1),
    )


# ---------------------------------------------------------------------------
# CLI: `python -m src.data.manifest_utils --manifest ... --stats`
# ---------------------------------------------------------------------------


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Inspect a MUCS manifest.")
    parser.add_argument("--manifest", required=True, help="Path to JSONL manifest.")
    parser.add_argument("--stats", action="store_true", help="Print dataset statistics.")
    parser.add_argument(
        "--rewrite-to", default=None,
        help="If provided, rewrite audio paths to this root and write output next to input.",
    )
    parser.add_argument("--old-prefix", default="/home/puneets/suksham-mucs/mucs")
    parser.add_argument("--split", default=None,
                        help="Split name (train/test/blindtest). Auto-detected from filename if omitted.")
    args = parser.parse_args()

    entries = load_manifest(args.manifest)
    print(f"Loaded {len(entries)} entries from {args.manifest}")

    if args.stats:
        stats = manifest_statistics(entries)
        for field, value in stats.__dict__.items():
            if isinstance(value, float):
                print(f"  {field:35s} = {value:.4f}")
            else:
                print(f"  {field:35s} = {value}")

    if args.rewrite_to:
        split = args.split or Path(args.manifest).parent.name
        rewritten = rewrite_manifest_paths(entries, args.old_prefix, args.rewrite_to, split)
        out = Path(args.manifest).with_suffix(".rewritten.jsonl")
        save_manifest(rewritten, out)
        print(f"Wrote rewritten manifest to {out}")


if __name__ == "__main__":  # pragma: no cover
    _cli()
