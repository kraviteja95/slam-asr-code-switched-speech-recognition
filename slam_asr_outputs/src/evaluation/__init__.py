"""Evaluation metrics: WER, CER, code-switch analytics."""

from .metrics import (
    compute_wer_cer,
    compute_wer,
    compute_cer,
    per_language_wer,
    oov_rate,
    evaluate_predictions_file,
)
from .code_switch_analysis import (
    code_switch_stats,
    confusion_by_script,
)

__all__ = [
    "compute_wer_cer",
    "compute_wer",
    "compute_cer",
    "per_language_wer",
    "oov_rate",
    "evaluate_predictions_file",
    "code_switch_stats",
    "confusion_by_script",
]
