"""Course-day baselines: GMM keyword spotter + Bi-LSTM CTC ASR.

These live alongside SLAM-ASR so that notebook 03 (classical ML) and
notebook 04 (early neural) have concrete, importable models.
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Day 2 — Gaussian Mixture Model keyword spotter
# ---------------------------------------------------------------------------


class GmmKeywordSpotter:
    """Per-class GMM classifier over MFCC frames.

    A simple pre-deep-learning baseline for spotting words like
    ``"tutorial"``, ``"linux"``, ``"python"`` in the MUCS corpus.

    Given a set of *keyword* clips and *background* clips, we fit one
    ``sklearn.mixture.GaussianMixture`` per class over concatenated MFCC
    frames.  Inference computes the log-likelihood of each frame under
    every GMM and averages over the utterance.  Argmax = predicted class.
    """

    def __init__(self, n_components: int = 16, covariance_type: str = "diag",
                 random_state: int = 42):
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.random_state = random_state
        self.classes_: List[str] = []
        self.models_: dict = {}

    def fit(self, features_by_class: dict[str, Sequence[np.ndarray]]) -> "GmmKeywordSpotter":
        """Fit one GMM per class.

        Parameters
        ----------
        features_by_class:
            Mapping ``class_name -> list of (T, D) MFCC arrays``.
        """
        from sklearn.mixture import GaussianMixture
        self.classes_ = list(features_by_class.keys())
        for cls in self.classes_:
            X = np.vstack(list(features_by_class[cls]))
            gmm = GaussianMixture(
                n_components=self.n_components,
                covariance_type=self.covariance_type,
                random_state=self.random_state,
                reg_covar=1e-4,
                max_iter=200,
            ).fit(X)
            self.models_[cls] = gmm
        return self

    def predict(self, feats: np.ndarray) -> Tuple[str, np.ndarray]:
        """Return ``(predicted_class, score_vector)`` for one (T, D) feature matrix."""
        if not self.models_:
            raise RuntimeError("Call .fit(...) first.")
        scores = np.array([self.models_[c].score(feats) for c in self.classes_])
        return self.classes_[int(np.argmax(scores))], scores

    def predict_many(self, feats_list: Sequence[np.ndarray]) -> List[str]:
        return [self.predict(f)[0] for f in feats_list]


# ---------------------------------------------------------------------------
# Day 3 — Bi-LSTM + CTC ASR
# ---------------------------------------------------------------------------


class BiLstmCtcModel(nn.Module):
    """Small Bi-LSTM acoustic model trained with CTC loss.

    Character-level output vocabulary; the tokeniser lives in
    :meth:`build_vocab`.  This is the "early neural" baseline (Day 3) —
    training runs in <30 minutes on a Kaggle T4 over a few hours of MUCS
    audio and gives a WER floor we can compare SLAM-ASR against.

    Parameters
    ----------
    input_dim:
        Feature dimension (typically 80 for log-mel).
    hidden_dim:
        Bi-LSTM hidden width per direction.
    num_layers:
        Number of stacked Bi-LSTM layers.
    vocab_size:
        Includes the CTC blank at index 0.
    dropout:
        Dropout between LSTM layers.
    """

    BLANK_ID = 0

    def __init__(
        self,
        input_dim: int = 80,
        hidden_dim: int = 256,
        num_layers: int = 3,
        vocab_size: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden_dim * 2, vocab_size)
        self.vocab_size = vocab_size

    def forward(
        self,
        features: torch.Tensor,          # (B, T, D)
        feature_lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return frame-level log-probabilities ``(B, T, V)``."""
        x = self.input_proj(features)
        if feature_lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                x, feature_lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            out, _ = self.lstm(packed)
            out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        else:
            out, _ = self.lstm(x)
        logits = self.head(out)
        return torch.log_softmax(logits, dim=-1)

    # -------------------- vocab helpers --------------------

    @staticmethod
    def build_vocab(texts: Iterable[str]) -> Tuple[dict, dict]:
        """Build a character-level vocab; blank is fixed at index 0.

        Returns ``(char2id, id2char)`` dictionaries.
        """
        chars: set[str] = set()
        for t in texts:
            chars.update(t)
        # Reserve 0 for CTC blank.
        char2id = {"<blank>": 0}
        for c in sorted(chars):
            char2id[c] = len(char2id)
        id2char = {i: c for c, i in char2id.items()}
        return char2id, id2char

    @staticmethod
    def encode(text: str, char2id: dict) -> List[int]:
        return [char2id[c] for c in text if c in char2id]

    @staticmethod
    def greedy_decode(log_probs: torch.Tensor, id2char: dict) -> List[str]:
        """Collapse repeats then remove blanks — the canonical CTC greedy decoder."""
        preds = log_probs.argmax(dim=-1).cpu().numpy()  # (B, T)
        results: List[str] = []
        blank = BiLstmCtcModel.BLANK_ID
        for row in preds:
            out: List[str] = []
            prev = -1
            for idx in row:
                if idx != prev and idx != blank:
                    out.append(id2char.get(int(idx), ""))
                prev = idx
            results.append("".join(out))
        return results

    @staticmethod
    def ctc_loss(
        log_probs: torch.Tensor,        # (B, T, V)
        targets: torch.Tensor,          # (sum(target_lengths),)
        input_lengths: torch.Tensor,
        target_lengths: torch.Tensor,
    ) -> torch.Tensor:
        return nn.functional.ctc_loss(
            log_probs.transpose(0, 1),  # (T, B, V) as CTC expects
            targets,
            input_lengths,
            target_lengths,
            blank=BiLstmCtcModel.BLANK_ID,
            reduction="mean",
            zero_infinity=True,
        )
