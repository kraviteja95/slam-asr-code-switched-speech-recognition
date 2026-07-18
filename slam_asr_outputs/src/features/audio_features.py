"""Classical DSP feature extractors used throughout the course.

Every function is a thin, documented wrapper around ``librosa`` so that the
Kaggle notebooks can call e.g. ``compute_log_mel(y, sr)`` with the same
parameters that the SLAM-ASR model uses internally.

The default parameters match ``configs/slam_asr.yaml`` (25 ms window,
10 ms hop, 80 mel bins) so that plots in the EDA notebook look identical
to what Whisper sees at training time.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_waveform(
    path: str | Path,
    sample_rate: int = 16000,
    mono: bool = True,
) -> Tuple[np.ndarray, int]:
    """Load a waveform via ``librosa``.

    Returns ``(y, sr)`` where ``y`` is float32 in ``[-1, 1]``.
    """
    import librosa  # local import so unit tests don't need librosa
    y, sr = librosa.load(str(path), sr=sample_rate, mono=mono)
    return y.astype(np.float32), int(sr)


# ---------------------------------------------------------------------------
# Time-frequency representations
# ---------------------------------------------------------------------------


def compute_stft(
    y: np.ndarray,
    n_fft: int = 400,
    hop_length: int = 160,
    win_length: int = 400,
    window: str = "hann",
) -> np.ndarray:
    """Short-time Fourier transform → complex spectrogram of shape ``(F, T)``.

    Default parameters correspond to a 25 ms window and 10 ms hop at 16 kHz.
    """
    import librosa
    return librosa.stft(y, n_fft=n_fft, hop_length=hop_length,
                        win_length=win_length, window=window)


def compute_log_mel(
    y: np.ndarray,
    sample_rate: int = 16000,
    n_fft: int = 400,
    hop_length: int = 160,
    n_mels: int = 80,
    fmin: float = 0.0,
    fmax: Optional[float] = None,
    log_offset: float = 1e-10,
) -> np.ndarray:
    """Log-mel spectrogram: ``log(mel_filterbank @ |STFT|^2 + epsilon)``.

    Returns a ``(n_mels, T)`` array.
    """
    import librosa
    mel = librosa.feature.melspectrogram(
        y=y, sr=sample_rate, n_fft=n_fft, hop_length=hop_length,
        n_mels=n_mels, fmin=fmin, fmax=fmax or sample_rate / 2,
    )
    return np.log(mel + log_offset)


def compute_mfcc(
    y: np.ndarray,
    sample_rate: int = 16000,
    n_mfcc: int = 13,
    n_fft: int = 400,
    hop_length: int = 160,
    include_deltas: bool = True,
) -> np.ndarray:
    """Mel-Frequency Cepstral Coefficients.

    If ``include_deltas`` is True, the output stacks ``[mfcc; Δmfcc; Δ²mfcc]``,
    yielding a ``(3 * n_mfcc, T)`` matrix — the classical GMM-HMM feature.
    """
    import librosa
    mfcc = librosa.feature.mfcc(
        y=y, sr=sample_rate, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length
    )
    if not include_deltas:
        return mfcc
    d1 = librosa.feature.delta(mfcc, order=1)
    d2 = librosa.feature.delta(mfcc, order=2)
    return np.vstack([mfcc, d1, d2])


def compute_chroma(
    y: np.ndarray,
    sample_rate: int = 16000,
    n_fft: int = 400,
    hop_length: int = 160,
    n_chroma: int = 12,
) -> np.ndarray:
    """12-bin chroma feature — useful for music/lecture-jingle detection."""
    import librosa
    return librosa.feature.chroma_stft(
        y=y, sr=sample_rate, n_fft=n_fft, hop_length=hop_length, n_chroma=n_chroma
    )


def compute_spectral_centroid(
    y: np.ndarray,
    sample_rate: int = 16000,
    hop_length: int = 160,
) -> np.ndarray:
    """Spectral centroid (Hz) — the *brightness* of the signal."""
    import librosa
    return librosa.feature.spectral_centroid(
        y=y, sr=sample_rate, hop_length=hop_length
    )[0]


def compute_zero_crossing_rate(
    y: np.ndarray,
    frame_length: int = 400,
    hop_length: int = 160,
) -> np.ndarray:
    """Zero-crossing rate — separates voiced (low ZCR) from unvoiced/fricative frames."""
    import librosa
    return librosa.feature.zero_crossing_rate(
        y=y, frame_length=frame_length, hop_length=hop_length
    )[0]


# ---------------------------------------------------------------------------
# Visualisation helpers (used by notebook 01 / 02)
# ---------------------------------------------------------------------------


def plot_waveform_and_spectrogram(
    y: np.ndarray,
    sample_rate: int = 16000,
    hop_length: int = 160,
    n_fft: int = 400,
    n_mels: int = 80,
    figsize: Tuple[int, int] = (12, 6),
    title: str = "",
):
    """Draw a two-panel figure: raw waveform + log-mel spectrogram.

    Requires ``matplotlib``.  Returns the ``Figure`` object.
    """
    import matplotlib.pyplot as plt
    import librosa
    import librosa.display

    log_mel = compute_log_mel(y, sample_rate=sample_rate,
                              n_fft=n_fft, hop_length=hop_length,
                              n_mels=n_mels)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=False)
    t = np.arange(len(y)) / sample_rate
    ax1.plot(t, y, linewidth=0.6)
    ax1.set_title(f"Waveform — {title}" if title else "Waveform")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Amplitude")
    ax1.set_xlim(0, t[-1] if len(t) else 1)

    img = librosa.display.specshow(
        log_mel, sr=sample_rate, hop_length=hop_length,
        x_axis="time", y_axis="mel", ax=ax2, cmap="magma",
    )
    ax2.set_title("Log-mel spectrogram (80 bins)")
    fig.colorbar(img, ax=ax2, format="%+2.0f dB")
    fig.tight_layout()
    return fig
