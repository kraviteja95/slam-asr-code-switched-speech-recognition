"""Batch and single-sample decoding for SLAM-ASR.

Usage from the CLI::

    python -m src.inference.decode \\
        --manifest datasets/mucs/data/processed/test/manifest.jsonl \\
        --checkpoint checkpoints/slam_asr/final \\
        --output predictions/test.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import torch
from tqdm import tqdm

from ..data.dataset import MUCSDataset, SlamASRCollator
from ..data.text_normalization import CodeSwitchTextNormalizer
from ..models.slam_asr import SlamAsrConfig, SlamAsrModel


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_slam_asr_from_checkpoint(
    ckpt_dir: str | Path,
    slam_cfg: Optional[SlamAsrConfig] = None,
    device: str = "cuda",
) -> SlamAsrModel:
    """Instantiate a :class:`SlamAsrModel` and load projector + LoRA weights.

    If ``slam_cfg`` is ``None`` we look for a ``slam_config.json`` file
    next to the checkpoint.
    """
    ckpt_dir = Path(ckpt_dir)
    if slam_cfg is None:
        cfg_path = ckpt_dir / "slam_config.json"
        if cfg_path.exists():
            with open(cfg_path) as fh:
                cfg_dict = json.load(fh)
            slam_cfg = SlamAsrConfig(**cfg_dict)
        else:
            slam_cfg = SlamAsrConfig()
    model = SlamAsrModel(slam_cfg)
    model.load_trainable(str(ckpt_dir))
    model.to(device)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Decoding
# ---------------------------------------------------------------------------


@torch.no_grad()
def decode_slam_asr(
    model: SlamAsrModel,
    audios: List[Any],
    sample_rate: int = 16000,
    max_new_tokens: int = 256,
    num_beams: int = 1,
    n_audio_tokens: int = 300,
    device: str = "cuda",
) -> List[str]:
    """Decode a *list* of raw waveforms.

    Parameters
    ----------
    audios:
        List of 1-D numpy float32 arrays in ``[-1, 1]`` at ``sample_rate``.
    """
    collator = SlamASRCollator(
        whisper_processor=model.whisper_processor,
        llm_tokenizer=model.llm_tokenizer,
        audio_placeholder_token=model.llm_tokenizer.pad_token,
        n_audio_tokens=n_audio_tokens,
    )
    # Build a fake batch with empty targets so the collator's label masking
    # ignores those positions.
    fake_batch = [{"audio": a, "text": ""} for a in audios]
    batch = collator(fake_batch)
    batch = {k: v.to(device) for k, v in batch.items()}

    generated = model.generate(
        input_features=batch["input_features"],
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        audio_placeholder_mask=batch["audio_placeholder_mask"],
        max_new_tokens=max_new_tokens,
        num_beams=num_beams,
    )
    # `generate` with `inputs_embeds` returns *only* the newly generated
    # token ids (no prompt).
    decoded = model.llm_tokenizer.batch_decode(
        generated, skip_special_tokens=True
    )
    normalizer = CodeSwitchTextNormalizer()
    return [normalizer(t.strip()) for t in decoded]


def decode_manifest(
    model: SlamAsrModel,
    manifest_path: str | Path,
    output_path: str | Path,
    batch_size: int = 4,
    audio_path_prefix: Optional[str] = None,
    max_new_tokens: int = 256,
    num_beams: int = 1,
    n_audio_tokens: int = 300,
    device: str = "cuda",
    sample_rate: int = 16000,
    limit: Optional[int] = None,
) -> str:
    """Run inference over an entire manifest, writing JSONL predictions."""
    ds = MUCSDataset(
        manifest_path,
        sample_rate=sample_rate,
        min_duration_s=0.0,
        max_duration_s=1e9,
        normalizer=CodeSwitchTextNormalizer(),
        audio_path_prefix=audio_path_prefix,
    )
    if limit is not None:
        ds.entries = ds.entries[:limit]

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        buf: List[Dict[str, Any]] = []
        for i in tqdm(range(0, len(ds), batch_size), desc="decoding"):
            batch_items = [ds[j] for j in range(i, min(i + batch_size, len(ds)))]
            audios = [b["audio"] for b in batch_items]
            preds = decode_slam_asr(
                model, audios,
                sample_rate=sample_rate,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                n_audio_tokens=n_audio_tokens,
                device=device,
            )
            for item, hyp in zip(batch_items, preds):
                fh.write(json.dumps({
                    "cut_id": item["cut_id"],
                    "speaker": item["speaker"],
                    "reference": item["text"],
                    "hypothesis": hyp,
                    "duration": item["duration"],
                }, ensure_ascii=False) + "\n")
    return str(output_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Decode a manifest with SLAM-ASR.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--audio-path-prefix", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--n-audio-tokens", type=int, default=300)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    model = load_slam_asr_from_checkpoint(args.checkpoint, device=args.device)
    decode_manifest(
        model,
        manifest_path=args.manifest,
        output_path=args.output,
        batch_size=args.batch_size,
        audio_path_prefix=args.audio_path_prefix,
        max_new_tokens=args.max_new_tokens,
        num_beams=args.num_beams,
        n_audio_tokens=args.n_audio_tokens,
        device=args.device,
        limit=args.limit,
    )
    print(f"Wrote predictions to {args.output}")


if __name__ == "__main__":  # pragma: no cover
    _cli()
