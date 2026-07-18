"""Word / character error rate + per-language and OOV metrics.

The MUCS challenge scores hypotheses with:

* Unicode NFC canonicalisation,
* Lower-casing of the Roman characters,
* Devanagari digit → Roman digit mapping,
* Stripping of punctuation.

We apply the exact same :class:`CodeSwitchTextNormalizer` to both
reference and hypothesis before invoking ``jiwer``.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from ..data.text_normalization import CodeSwitchTextNormalizer

DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")
ROMAN_RE = re.compile(r"[A-Za-z]")


# ---------------------------------------------------------------------------
# Core WER / CER
# ---------------------------------------------------------------------------


def _import_jiwer():
    try:
        import jiwer
    except ImportError as err:  # pragma: no cover
        raise ImportError(
            "jiwer is required for WER/CER; install with `pip install jiwer`."
        ) from err
    return jiwer


def compute_wer(refs: List[str], hyps: List[str],
                normalize: bool = True) -> float:
    """Word Error Rate ∈ ``[0, ∞)``."""
    jiwer = _import_jiwer()
    if normalize:
        norm = CodeSwitchTextNormalizer()
        refs = norm.batch_normalize(refs)
        hyps = norm.batch_normalize(hyps)
    # jiwer drops empty references — replace them with a single placeholder
    # so the sample still counts.
    refs = [r if r.strip() else "<empty>" for r in refs]
    hyps = [h if h.strip() else "<empty>" for h in hyps]
    return float(jiwer.wer(refs, hyps))


def compute_cer(refs: List[str], hyps: List[str],
                normalize: bool = True) -> float:
    """Character Error Rate — kinder to script-mixed transcripts than WER."""
    jiwer = _import_jiwer()
    if normalize:
        norm = CodeSwitchTextNormalizer()
        refs = norm.batch_normalize(refs)
        hyps = norm.batch_normalize(hyps)
    refs = [r if r.strip() else "<empty>" for r in refs]
    hyps = [h if h.strip() else "<empty>" for h in hyps]
    return float(jiwer.cer(refs, hyps))


def compute_wer_cer(refs: List[str], hyps: List[str],
                    normalize: bool = True) -> Dict[str, float]:
    """Return WER and CER in one dictionary — convenience."""
    return {
        "wer": compute_wer(refs, hyps, normalize=normalize),
        "cer": compute_cer(refs, hyps, normalize=normalize),
    }


# ---------------------------------------------------------------------------
# Per-language WER
# ---------------------------------------------------------------------------


def _script_of(word: str) -> str:
    """Return ``hi`` / ``en`` / ``other`` for a token."""
    if DEVANAGARI_RE.search(word):
        return "hi"
    if ROMAN_RE.search(word):
        return "en"
    return "other"


def per_language_wer(refs: List[str], hyps: List[str],
                     normalize: bool = True) -> Dict[str, float]:
    """Approximate per-language WER via token-level alignment.

    Uses ``jiwer`` to compute the raw edit alignment for each utterance,
    then buckets each reference word by script and asks: was it correctly
    predicted? This is the metric used in the MUCS baseline paper.
    """
    jiwer = _import_jiwer()
    if normalize:
        norm = CodeSwitchTextNormalizer()
        refs = norm.batch_normalize(refs)
        hyps = norm.batch_normalize(hyps)

    hi_errs = hi_words = 0
    en_errs = en_words = 0
    for ref, hyp in zip(refs, hyps):
        if not ref.strip():
            continue
        out = jiwer.process_words([ref], [hyp])
        # jiwer >= 3.0 exposes alignments[0]: list of AlignmentChunk objects
        ref_words = ref.split()
        hyp_words = hyp.split()
        try:
            chunks = out.alignments[0]
        except AttributeError:  # older jiwer
            chunks = []
        for chunk in chunks:
            op = chunk.type
            for r_idx in range(chunk.ref_start_idx, chunk.ref_end_idx):
                w = ref_words[r_idx]
                lang = _script_of(w)
                if lang == "hi":
                    hi_words += 1
                    if op != "equal":
                        hi_errs += 1
                elif lang == "en":
                    en_words += 1
                    if op != "equal":
                        en_errs += 1
    return {
        "hindi_wer": hi_errs / max(hi_words, 1),
        "english_wer": en_errs / max(en_words, 1),
        "n_hindi_words": hi_words,
        "n_english_words": en_words,
    }


# ---------------------------------------------------------------------------
# OOV rate
# ---------------------------------------------------------------------------


def oov_rate(train_refs: List[str], test_refs: List[str],
             normalize: bool = True) -> float:
    """Fraction of *test* word tokens that never appear in *train*."""
    if normalize:
        norm = CodeSwitchTextNormalizer()
        train_refs = norm.batch_normalize(train_refs)
        test_refs = norm.batch_normalize(test_refs)
    vocab = set()
    for r in train_refs:
        vocab.update(r.split())
    n = oov = 0
    for r in test_refs:
        for w in r.split():
            n += 1
            if w not in vocab:
                oov += 1
    return oov / max(n, 1)


# ---------------------------------------------------------------------------
# End-to-end evaluation over a predictions JSONL file
# ---------------------------------------------------------------------------


def evaluate_predictions_file(
    predictions_path: str | Path,
    train_manifest: Optional[str | Path] = None,
    normalize: bool = True,
    verbose: bool = True,
) -> Dict[str, float]:
    """Score a JSONL predictions file (fields ``reference`` and ``hypothesis``).

    Optionally computes OOV rate against ``train_manifest``.
    """
    refs: List[str] = []
    hyps: List[str] = []
    with open(predictions_path, "r", encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            refs.append(row.get("reference", ""))
            hyps.append(row.get("hypothesis", ""))

    metrics = compute_wer_cer(refs, hyps, normalize=normalize)
    metrics.update(per_language_wer(refs, hyps, normalize=normalize))

    if train_manifest is not None:
        from ..data.manifest_utils import load_manifest
        train_rows = load_manifest(train_manifest)
        train_refs = [r.get("text") or r.get("target") or "" for r in train_rows]
        metrics["oov_rate"] = oov_rate(train_refs, refs, normalize=normalize)

    if verbose:
        print(f"[eval] over {len(refs)} utterances:")
        for k, v in metrics.items():
            if isinstance(v, float):
                print(f"  {k:20s} = {v:.4f}")
            else:
                print(f"  {k:20s} = {v}")
    return metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Score predictions JSONL.")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--train-manifest", default=None,
                        help="If given, compute OOV rate against this manifest.")
    parser.add_argument("--no-normalize", action="store_true")
    args = parser.parse_args()
    evaluate_predictions_file(
        args.predictions,
        train_manifest=args.train_manifest,
        normalize=not args.no_normalize,
    )


if __name__ == "__main__":  # pragma: no cover
    _cli()
