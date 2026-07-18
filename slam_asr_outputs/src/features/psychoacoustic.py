"""Psychoacoustic helpers — Day-1 course topics.

Implements the *perceptual* scales and masking curves discussed in the
lectures so that the notebook can visualise **why** we use log-mel
features instead of raw STFT bins.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Frequency scales
# ---------------------------------------------------------------------------


def hz_to_mel(hz: np.ndarray | float, htk: bool = False) -> np.ndarray:
    """Convert Hz → Mel.

    Uses the O'Shaughnessy (Slaney) formula by default, which is what
    ``librosa`` and Whisper use.  Set ``htk=True`` for the classical HTK
    formula ``2595 log10(1 + f/700)``.
    """
    hz = np.asarray(hz, dtype=np.float64)
    if htk:
        return 2595.0 * np.log10(1.0 + hz / 700.0)
    # Slaney: linear below 1 kHz, log above.
    f_min = 0.0
    f_sp = 200.0 / 3
    min_log_hz = 1000.0
    min_log_mel = (min_log_hz - f_min) / f_sp
    logstep = np.log(6.4) / 27.0

    mels = (hz - f_min) / f_sp
    log_region = hz >= min_log_hz
    mels = np.where(
        log_region,
        min_log_mel + np.log(hz / min_log_hz) / logstep,
        mels,
    )
    return mels


def mel_to_hz(mels: np.ndarray | float, htk: bool = False) -> np.ndarray:
    """Inverse of :func:`hz_to_mel`."""
    mels = np.asarray(mels, dtype=np.float64)
    if htk:
        return 700.0 * (10.0 ** (mels / 2595.0) - 1.0)
    f_min = 0.0
    f_sp = 200.0 / 3
    min_log_hz = 1000.0
    min_log_mel = (min_log_hz - f_min) / f_sp
    logstep = np.log(6.4) / 27.0
    freqs = f_min + f_sp * mels
    log_region = mels >= min_log_mel
    freqs = np.where(
        log_region,
        min_log_hz * np.exp(logstep * (mels - min_log_mel)),
        freqs,
    )
    return freqs


def hz_to_bark(hz: np.ndarray | float) -> np.ndarray:
    """Hz → Bark scale (Traunmüller 1990).

    Bark divides the audible spectrum into 24 critical bands aligned with
    the physiology of the cochlea.
    """
    hz = np.asarray(hz, dtype=np.float64)
    return (26.81 * hz / (1960 + hz)) - 0.53


def bark_to_hz(bark: np.ndarray | float) -> np.ndarray:
    """Inverse of :func:`hz_to_bark`."""
    bark = np.asarray(bark, dtype=np.float64)
    return 1960.0 * (bark + 0.53) / (26.28 - bark)


# ---------------------------------------------------------------------------
# Hearing & masking
# ---------------------------------------------------------------------------


def absolute_threshold_of_hearing(freq_hz: np.ndarray) -> np.ndarray:
    """Terhardt's absolute threshold of hearing curve (dB SPL vs Hz).

    Below this curve a tone is inaudible even in perfect silence — this is
    why the log-mel filterbank can safely down-weight very low and very
    high frequencies.
    """
    f_khz = np.asarray(freq_hz, dtype=np.float64) / 1000.0
    # Terhardt 1979
    with np.errstate(divide="ignore"):
        threshold = (
            3.64 * (f_khz ** -0.8)
            - 6.5 * np.exp(-0.6 * (f_khz - 3.3) ** 2)
            + 1e-3 * (f_khz ** 4)
        )
    return threshold


def simultaneous_masking_curve(masker_hz: float, masker_db: float,
                               freqs_hz: np.ndarray) -> np.ndarray:
    """A rough simultaneous-masking spreading function around a masker tone.

    Follows the piece-wise linear model in Painter & Spanias 2000:
    upward masking is +25 dB/Bark, downward is -10 dB/Bark.  Returns the
    masking threshold in dB SPL at each ``freqs_hz`` position.
    """
    b_m = float(hz_to_bark(masker_hz))
    b = hz_to_bark(freqs_hz)
    dz = b - b_m
    slope = np.where(dz >= 0, -25.0, +10.0)   # attenuation per Bark
    return masker_db + slope * np.abs(dz)
