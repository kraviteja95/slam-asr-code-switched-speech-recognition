"""End-to-end training loop for SLAM-ASR wrapped around HuggingFace ``Trainer``.

The Trainer handles distributed dataloading, gradient accumulation,
gradient checkpointing, mixed-precision, and checkpoint rotation for us.
We only need to:

* build the model (:class:`SlamAsrModel`),
* build the collator (:class:`SlamASRCollator`),
* build the datasets (:class:`MUCSDataset`),
* configure the optimizer so the projector gets a slightly higher LR
  than the LoRA matrices, and
* provide a ``compute_metrics`` that decodes greedy predictions and
  returns WER.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, TypedDict

import torch
from torch.utils.data import Dataset

from ..data.dataset import MUCSDataset, SlamASRCollator
from ..data.text_normalization import CodeSwitchTextNormalizer
from ..evaluation.metrics import compute_wer_cer
from ..models.slam_asr import SlamAsrConfig, SlamAsrModel


class TrainingConfigDict(TypedDict, total=False):
    output_dir: str
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    gradient_accumulation_steps: int
    num_train_epochs: int
    learning_rate: float
    projector_learning_rate: float
    warmup_ratio: float
    weight_decay: float
    lr_scheduler: str
    logging_steps: int
    eval_steps: int
    save_steps: int
    save_total_limit: int
    bf16: bool
    gradient_checkpointing: bool
    dataloader_num_workers: int
    report_to: str


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_slam_trainer(
    model: SlamAsrModel,
    train_dataset: Dataset,
    eval_dataset: Optional[Dataset],
    training_cfg: TrainingConfigDict,
    n_audio_tokens: int = 300,
    instruction: str = (
        "Transcribe the following Hindi-English code-switched speech. "
        "Output only the transcript in mixed Devanagari and Roman script."
    ),
    system_prompt: str = "You are a bilingual Hindi-English speech recognition assistant.",
):
    """Return a configured HuggingFace ``Trainer`` ready to ``.train()``.

    Parameter groups are set up so the projector trains 10× faster than
    the LoRA adapters (LoRA carries pretrained information; the projector
    starts from scratch).
    """
    from transformers import Trainer, TrainingArguments

    tokenizer = model.llm_tokenizer

    collator = SlamASRCollator(
        whisper_processor=model.whisper_processor,
        llm_tokenizer=tokenizer,
        audio_placeholder_token=tokenizer.pad_token,
        instruction=instruction,
        system_prompt=system_prompt,
        n_audio_tokens=n_audio_tokens,
    )

    # ---- optimizer parameter groups ----
    proj_params = list(model.projector.parameters())
    proj_ids = {id(p) for p in proj_params}
    other_params = [p for p in model.parameters()
                    if p.requires_grad and id(p) not in proj_ids]
    optim_groups = [
        {"params": proj_params, "lr": training_cfg.get("projector_learning_rate", 1e-3)},
        {"params": other_params, "lr": training_cfg.get("learning_rate", 1e-4)},
    ]

    args = TrainingArguments(
        output_dir=training_cfg.get("output_dir", "checkpoints/slam_asr"),
        per_device_train_batch_size=training_cfg.get("per_device_train_batch_size", 4),
        per_device_eval_batch_size=training_cfg.get("per_device_eval_batch_size", 4),
        gradient_accumulation_steps=training_cfg.get("gradient_accumulation_steps", 8),
        num_train_epochs=training_cfg.get("num_train_epochs", 3),
        learning_rate=training_cfg.get("learning_rate", 1e-4),
        warmup_ratio=training_cfg.get("warmup_ratio", 0.03),
        weight_decay=training_cfg.get("weight_decay", 0.01),
        lr_scheduler_type=training_cfg.get("lr_scheduler", "cosine"),
        logging_steps=training_cfg.get("logging_steps", 50),
        eval_strategy="steps" if eval_dataset is not None else "no",
        eval_steps=training_cfg.get("eval_steps", 500),
        save_strategy="steps",
        save_steps=training_cfg.get("save_steps", 500),
        save_total_limit=training_cfg.get("save_total_limit", 3),
        bf16=training_cfg.get("bf16", True),
        gradient_checkpointing=training_cfg.get("gradient_checkpointing", True),
        dataloader_num_workers=training_cfg.get("dataloader_num_workers", 2),
        report_to=training_cfg.get("report_to", "none"),
        remove_unused_columns=False,           # our collator needs custom columns
        label_names=["labels"],
    )

    # Build optimizer manually so per-group LRs stick.
    optimizer = torch.optim.AdamW(optim_groups, weight_decay=args.weight_decay)

    class _SlamTrainer(Trainer):
        # HF Trainer computes loss = outputs.loss when labels are present.
        # We just need to preserve our custom kwargs through the model call.
        def _prepare_inputs(self, inputs):
            inputs = super()._prepare_inputs(inputs)
            return inputs

    trainer = _SlamTrainer(
        model=model,
        args=args,
        data_collator=collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        optimizers=(optimizer, None),
    )
    return trainer


def train_slam_asr(
    train_manifest: str | Path,
    eval_manifest: Optional[str | Path],
    slam_cfg: SlamAsrConfig,
    training_cfg: TrainingConfigDict,
    sample_rate: int = 16000,
    max_duration_s: float = 30.0,
    min_duration_s: float = 0.5,
    n_audio_tokens: int = 300,
    audio_path_prefix: Optional[str] = None,
):
    """One-call training helper used by the Kaggle notebook.

    Builds the model, datasets and trainer, kicks off training, and
    returns the trained :class:`SlamAsrModel`.
    """
    normalizer = CodeSwitchTextNormalizer()

    train_ds = MUCSDataset(
        train_manifest,
        sample_rate=sample_rate,
        min_duration_s=min_duration_s,
        max_duration_s=max_duration_s,
        normalizer=normalizer,
        audio_path_prefix=audio_path_prefix,
    )
    eval_ds = None
    if eval_manifest is not None:
        eval_ds = MUCSDataset(
            eval_manifest,
            sample_rate=sample_rate,
            min_duration_s=min_duration_s,
            max_duration_s=max_duration_s,
            normalizer=normalizer,
            audio_path_prefix=audio_path_prefix,
        )

    model = SlamAsrModel(slam_cfg)
    params = model.get_trainable_parameters()
    print(
        f"[SLAM-ASR] trainable / total = "
        f"{params['trainable']/1e6:.2f} M / {params['total']/1e6:.2f} M "
        f"({100 * params['trainable'] / params['total']:.3f}%)"
    )

    trainer = build_slam_trainer(
        model,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        training_cfg=training_cfg,
        n_audio_tokens=n_audio_tokens,
    )
    trainer.train()
    model.save_trainable(training_cfg.get("output_dir", "checkpoints/slam_asr") + "/final")
    return model, trainer
