"""Training utilities for SLAM-ASR (HF Trainer wrapper)."""

from .train_slam import build_slam_trainer, train_slam_asr, TrainingConfigDict

__all__ = ["build_slam_trainer", "train_slam_asr", "TrainingConfigDict"]
