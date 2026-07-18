"""PyTorch dataset & collators for MUCS Hindi-English code-switched ASR.

Three consumers are supported:

* :class:`MUCSDataset` — thin ``torch.utils.data.Dataset`` that lazily loads
  a waveform + reference transcript from a manifest row.  Optional on-the-fly
  duration filtering and text normalisation.

* :class:`WhisperCollator` — batches raw waveforms into 80-mel log-Mel
  features + tokenised targets using a HuggingFace
  ``WhisperProcessor``.  Used by the Whisper-only baseline (notebook 5).

* :class:`SlamASRCollator` — the collator that powers the full SLAM-ASR
  model.  It produces (a) Whisper input features, (b) tokenised LLM
  ``input_ids`` for the prompt + target, and (c) a mask that tells the
  training loop which positions must be replaced with projected audio
  embeddings at forward time.

All collators expect the config dictionary described in
``configs/slam_asr.yaml``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from .manifest_utils import load_manifest
from .text_normalization import CodeSwitchTextNormalizer

try:
    import soundfile as sf
except ImportError as _err:  # pragma: no cover
    sf = None
    _SF_IMPORT_ERROR = _err


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class MUCSDataset(Dataset):
    """PyTorch dataset over a MUCS manifest.

    Parameters
    ----------
    manifest_path:
        Path to a JSONL manifest.  Each row must have ``audio_path`` and
        ``text`` (or ``source`` / ``target``) fields.
    sample_rate:
        Expected sample rate.  If the audio file uses a different rate a
        cheap fallback resampler is used (librosa if available, else raise).
    min_duration_s, max_duration_s:
        Filter out utterances outside this range at load time.
    normalizer:
        Optional :class:`CodeSwitchTextNormalizer`.  Passing one guarantees
        the returned ``text`` matches the reference used at evaluation.
    audio_path_prefix:
        Optional prefix prepended to relative ``audio_path`` fields (useful
        when the manifest has been rewritten to relative paths).
    """

    def __init__(
        self,
        manifest_path: str | Path,
        sample_rate: int = 16000,
        min_duration_s: float = 0.5,
        max_duration_s: float = 30.0,
        normalizer: Optional[CodeSwitchTextNormalizer] = None,
        audio_path_prefix: Optional[str] = None,
    ) -> None:
        super().__init__()
        if sf is None:  # pragma: no cover
            raise ImportError(f"soundfile is required to load audio: {_SF_IMPORT_ERROR}")

        raw = load_manifest(manifest_path)
        # Filter by duration.
        self.entries: List[Dict[str, Any]] = [
            r for r in raw
            if min_duration_s <= float(r.get("duration", max_duration_s)) <= max_duration_s
        ]
        self.sample_rate = sample_rate
        self.normalizer = normalizer
        self.audio_path_prefix = audio_path_prefix

    def __len__(self) -> int:
        return len(self.entries)

    def _resolve_path(self, row: Dict[str, Any]) -> str:
        p = row.get("audio_path") or row.get("source") or ""
        if self.audio_path_prefix and not p.startswith("/") and not p.startswith(str(self.audio_path_prefix)):
            p = str(Path(self.audio_path_prefix) / p)
        return p

    def _load_audio(self, path: str) -> np.ndarray:
        """Load audio from disk, resample if needed, return mono float32 in [-1, 1]."""
        audio, sr = sf.read(path, dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != self.sample_rate:
            # Prefer librosa (high-quality kaiser); fall back to a plain linear
            # resample if librosa isn't available (very rare on Kaggle).
            try:  # pragma: no cover
                import librosa
                audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)
            except ImportError:
                idx = np.linspace(0, len(audio) - 1, int(len(audio) * self.sample_rate / sr))
                audio = np.interp(idx, np.arange(len(audio)), audio).astype("float32")
        return audio

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.entries[idx]
        audio = self._load_audio(self._resolve_path(row))
        text = row.get("text") or row.get("target") or ""
        if self.normalizer is not None:
            text = self.normalizer(text)
        return {
            "audio": audio,
            "text": text,
            "sample_rate": self.sample_rate,
            "speaker": row.get("speaker"),
            "cut_id": row.get("cut_id") or row.get("key"),
            "duration": float(row.get("duration", len(audio) / self.sample_rate)),
        }


# ---------------------------------------------------------------------------
# Whisper collator (encoder-decoder baseline in notebook 5)
# ---------------------------------------------------------------------------


@dataclass
class WhisperCollator:
    """Collate a batch of dataset items into Whisper training tensors.

    Parameters
    ----------
    processor:
        ``transformers.WhisperProcessor`` instance.  Provides the mel-filter
        bank + the byte-pair tokenizer.
    language:
        Whisper language code.  ``"hi"`` produces the best results for Hindi
        with embedded English.
    task:
        Either ``"transcribe"`` or ``"translate"``.
    """

    processor: Any
    language: str = "hi"
    task: str = "transcribe"

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        audios = [b["audio"] for b in batch]
        texts = [b["text"] for b in batch]

        # 80-mel features, padded to 30 s.
        inputs = self.processor.feature_extractor(
            audios,
            sampling_rate=self.processor.feature_extractor.sampling_rate,
            return_tensors="pt",
        )
        # Force the decoder into Hindi transcription mode.
        self.processor.tokenizer.set_prefix_tokens(language=self.language, task=self.task)
        labels = self.processor.tokenizer(
            texts, padding=True, return_tensors="pt"
        )["input_ids"]
        # Replace pad tokens with -100 so they are ignored by CrossEntropy.
        labels = labels.masked_fill(labels == self.processor.tokenizer.pad_token_id, -100)

        return {
            "input_features": inputs["input_features"],
            "labels": labels,
        }


# ---------------------------------------------------------------------------
# SLAM-ASR collator (Whisper encoder + LLM decoder)
# ---------------------------------------------------------------------------


@dataclass
class SlamASRCollator:
    """Collate a batch for the SLAM-ASR model.

    The output is a plain ``dict`` with:

    * ``input_features``  — Whisper log-Mel tensor ``(B, 80, T_mel)``.
    * ``input_ids``       — LLM token ids of the *prompt template*
      (``<AUDIO>`` positions filled with the LLM's ``pad`` id and replaced
      with projected audio embeddings inside the model).
    * ``attention_mask``  — corresponding mask.
    * ``labels``          — target token ids, ``-100`` everywhere except
      the assistant-response span so the LLM only learns to predict the
      transcript.
    * ``audio_placeholder_mask`` — boolean tensor of shape ``input_ids``
      that marks the positions of ``<AUDIO>``; consumed by the model's
      ``forward``.

    Parameters
    ----------
    whisper_processor:
        For extracting log-Mel features.
    llm_tokenizer:
        HuggingFace tokenizer of the LLM decoder.
    audio_placeholder_token:
        String inserted into the prompt where audio embeddings will be
        spliced in.  Any string is fine as long as it produces at least
        one token id.  We use the LLM's ``pad`` token by default so we
        don't need to grow the vocabulary.
    instruction:
        Natural-language instruction to prepend to every audio (matches
        ``configs/slam_asr.yaml::prompt.instruction``).
    n_audio_tokens:
        How many placeholder tokens to insert.  Should be an *upper bound*
        on the projector's output length; the model masks out unused
        positions at forward time.  A safe default for 30 s Whisper-base
        with a downsample-5 projector is ``30*50/5 = 300``.
    system_prompt:
        Optional system message that sets the assistant's role.
    """

    whisper_processor: Any
    llm_tokenizer: Any
    audio_placeholder_token: str = "<|audio|>"
    instruction: str = (
        "Transcribe the following Hindi-English code-switched speech. "
        "Output only the transcript in mixed Devanagari and Roman script."
    )
    n_audio_tokens: int = 300
    system_prompt: str = (
        "You are a bilingual Hindi-English speech recognition assistant."
    )

    # ---- internal helpers ---------------------------------------------------

    def _build_prompt_ids(self, target_text: str) -> Dict[str, torch.Tensor]:
        """Tokenise ``system + instruction + <AUDIO>*N + target`` for one sample.

        Returns dict of 1-D tensors: ``input_ids``, ``attention_mask``,
        ``labels``, ``audio_placeholder_mask``.
        """
        tok = self.llm_tokenizer
        # We use the chat template if the model provides one — this handles
        # Qwen / LLaMA-3 special tokens correctly.  Otherwise we fall back
        # to a plain concatenation.
        placeholder = self.audio_placeholder_token * self.n_audio_tokens

        if hasattr(tok, "apply_chat_template") and tok.chat_template:
            # Two-pass approach: first render prompt without target to know
            # where the target starts, then render with target.
            prompt_wo = tok.apply_chat_template(
                [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": self.instruction + "\n" + placeholder},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
            full_text = prompt_wo + target_text + (tok.eos_token or "")
        else:
            prompt_wo = (
                f"{self.system_prompt}\n"
                f"{self.instruction}\n{placeholder}\n"
                f"Transcript: "
            )
            full_text = prompt_wo + target_text + (tok.eos_token or "")

        # Tokenise both to figure out the split point.
        ids_wo = tok(prompt_wo, add_special_tokens=False)["input_ids"]
        ids_full = tok(full_text, add_special_tokens=False)["input_ids"]
        target_start = len(ids_wo)

        input_ids = torch.tensor(ids_full, dtype=torch.long)
        labels = input_ids.clone()
        labels[:target_start] = -100  # don't compute loss on prompt

        # Placeholder mask — where <AUDIO> tokens live.
        placeholder_ids = tok(self.audio_placeholder_token, add_special_tokens=False)["input_ids"]
        if len(placeholder_ids) == 1:
            audio_mask = input_ids == placeholder_ids[0]
        else:
            # Multi-token placeholder — walk the sequence and mark spans.
            audio_mask = torch.zeros_like(input_ids, dtype=torch.bool)
            plen = len(placeholder_ids)
            i = 0
            while i <= len(input_ids) - plen:
                if input_ids[i:i + plen].tolist() == placeholder_ids:
                    audio_mask[i:i + plen] = True
                    i += plen
                else:
                    i += 1

        return {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
            "labels": labels,
            "audio_placeholder_mask": audio_mask,
        }

    # ---- public entrypoint --------------------------------------------------

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # 1) Audio → log-Mel features (fixed length, padded to 30 s).
        audios = [b["audio"] for b in batch]
        feats = self.whisper_processor.feature_extractor(
            audios,
            sampling_rate=self.whisper_processor.feature_extractor.sampling_rate,
            return_tensors="pt",
        )["input_features"]

        # 2) Prompt/label token sequences (variable length → pad).
        per_sample = [self._build_prompt_ids(b["text"]) for b in batch]
        pad_id = self.llm_tokenizer.pad_token_id or self.llm_tokenizer.eos_token_id
        max_len = max(x["input_ids"].size(0) for x in per_sample)

        def _pad(t: torch.Tensor, fill: int) -> torch.Tensor:
            pad_amt = max_len - t.size(0)
            if pad_amt <= 0:
                return t
            return torch.cat([t, torch.full((pad_amt,), fill, dtype=t.dtype)], dim=0)

        input_ids = torch.stack([_pad(x["input_ids"], pad_id) for x in per_sample])
        attention_mask = torch.stack([_pad(x["attention_mask"], 0) for x in per_sample])
        labels = torch.stack([_pad(x["labels"], -100) for x in per_sample])
        audio_placeholder_mask = torch.stack(
            [_pad(x["audio_placeholder_mask"].long(), 0).bool() for x in per_sample]
        )

        return {
            "input_features": feats,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "audio_placeholder_mask": audio_placeholder_mask,
        }
