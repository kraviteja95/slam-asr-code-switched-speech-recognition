"""SLAM-ASR: Whisper encoder + projector + LoRA-adapted LLM decoder.

Reference: Ma et al., "An Embarrassingly Simple Approach for LLM with Strong
ASR Capacity" (arXiv:2402.08846).  We follow the tripartite recipe:

1. **Acoustic encoder** — a *frozen* Whisper encoder produces continuous
   speech representations at 50 Hz.
2. **Alignment projector** — a small 1-D CNN + MLP downsamples the
   audio-token stream and projects it into the LLM's embedding space.
3. **Language decoder** — a pre-trained decoder LLM (Qwen-2.5-1.5B by
   default) is loaded in 4-bit NF4 and adapted with LoRA.  We only train
   the projector and the LoRA matrices.

At forward time we splice the projected audio tokens into the LLM's
input-embedding sequence at the positions marked by
``audio_placeholder_mask`` (see :class:`SlamASRCollator`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torch.nn import functional as F

from .projector import ConvProjector

# We import HF classes lazily inside the constructor so that the rest of
# the codebase (e.g. src.evaluation, notebook 01) works without pulling
# multi-gigabyte weights into memory.


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class SlamAsrConfig:
    """Configuration for :class:`SlamAsrModel`."""

    encoder_name: str = "openai/whisper-base"
    decoder_name: str = "Qwen/Qwen2.5-1.5B-Instruct"

    # Projector
    projector_hidden_dim: int = 2048
    projector_downsample_factor: int = 5
    projector_n_conv_layers: int = 1
    projector_dropout: float = 0.0

    # LLM loading
    load_in_4bit: bool = True
    torch_dtype: str = "bfloat16"

    # LoRA
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "up_proj", "down_proj", "gate_proj",
    ])

    # Behaviour
    freeze_encoder: bool = True
    freeze_llm_backbone: bool = True   # LoRA takes care of adaptation


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class SlamAsrModel(nn.Module):
    """End-to-end SLAM-ASR module.

    The model exposes a standard ``forward(input_features, input_ids,
    attention_mask, labels, audio_placeholder_mask)`` interface so that it
    plugs into a HuggingFace ``Trainer``.  Loss is standard causal-LM
    cross-entropy computed only over the transcript positions (positions
    marked ``-100`` in ``labels`` are ignored).
    """

    def __init__(self, config: SlamAsrConfig):
        super().__init__()
        self.config = config

        # Lazy imports keep the top-level ``import src.models`` lightweight.
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            WhisperModel,
            WhisperProcessor,
        )

        # ---- 1) Frozen Whisper encoder ----
        whisper = WhisperModel.from_pretrained(config.encoder_name)
        self.whisper_encoder = whisper.encoder
        self.whisper_processor = WhisperProcessor.from_pretrained(config.encoder_name)
        encoder_hidden_size = self.whisper_encoder.config.d_model
        if config.freeze_encoder:
            for p in self.whisper_encoder.parameters():
                p.requires_grad = False
            self.whisper_encoder.eval()

        # ---- 2) 4-bit + LoRA LLM decoder ----
        dtype = getattr(torch, config.torch_dtype)
        quant_kwargs: Dict[str, Any] = {}
        if config.load_in_4bit:
            try:
                from transformers import BitsAndBytesConfig
                quant_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=dtype,
                    bnb_4bit_use_double_quant=True,
                )
            except ImportError:
                print("[SLAM-ASR] bitsandbytes unavailable; loading LLM in bfloat16")
                config.load_in_4bit = False

        self.llm_tokenizer = AutoTokenizer.from_pretrained(
            config.decoder_name, trust_remote_code=True
        )
        if self.llm_tokenizer.pad_token is None:
            self.llm_tokenizer.pad_token = self.llm_tokenizer.eos_token

        self.llm = AutoModelForCausalLM.from_pretrained(
            config.decoder_name,
            torch_dtype=dtype,
            trust_remote_code=True,
            **quant_kwargs,
        )
        self.llm.config.pad_token_id = self.llm_tokenizer.pad_token_id

        # Attach LoRA adapters.
        if config.use_lora:
            from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
            if config.load_in_4bit:
                self.llm = prepare_model_for_kbit_training(self.llm)
            lora_cfg = LoraConfig(
                r=config.lora_r,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=config.lora_target_modules,
            )
            self.llm = get_peft_model(self.llm, lora_cfg)
        elif config.freeze_llm_backbone:
            for p in self.llm.parameters():
                p.requires_grad = False

        # ---- 3) Projector ----
        llm_hidden_size = self._get_llm_hidden_size()
        self.projector = ConvProjector(
            in_dim=encoder_hidden_size,
            out_dim=llm_hidden_size,
            hidden_dim=config.projector_hidden_dim,
            downsample_factor=config.projector_downsample_factor,
            n_conv_layers=config.projector_n_conv_layers,
            dropout=config.projector_dropout,
        )

    # ---------------------------------------------------------------- utils

    def _get_llm_hidden_size(self) -> int:
        cfg = self.llm.config
        for attr in ("hidden_size", "n_embd", "d_model"):
            if hasattr(cfg, attr):
                return int(getattr(cfg, attr))
        raise AttributeError("LLM config has no recognisable hidden_size attribute.")

    def _input_embeddings(self) -> nn.Module:
        # Works transparently for PEFT-wrapped models via ``get_input_embeddings``.
        return self.llm.get_input_embeddings()

    def get_trainable_parameters(self) -> Dict[str, int]:
        """Return ``{"trainable": N, "total": M}`` for logging."""
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        return {"trainable": trainable, "total": total}

    # ---------------------------------------------------------------- core

    def encode_audio(self, input_features: torch.Tensor) -> torch.Tensor:
        """Frozen-encoder forward → projected audio tokens.

        Parameters
        ----------
        input_features:
            ``(B, 80, T_mel)`` log-Mel features from ``WhisperProcessor``.

        Returns
        -------
        Tensor
            ``(B, T_ds, D_llm)`` audio embeddings ready to be spliced into
            the LLM input.
        """
        was_training = self.whisper_encoder.training
        if self.config.freeze_encoder:
            self.whisper_encoder.eval()
        with torch.set_grad_enabled(not self.config.freeze_encoder):
            enc_out = self.whisper_encoder(input_features, return_dict=True)
        if was_training and not self.config.freeze_encoder:
            self.whisper_encoder.train()
        # enc_out.last_hidden_state: (B, T_enc, D_enc)
        return self.projector(enc_out.last_hidden_state)

    def _splice_audio_into_embeddings(
        self,
        input_ids: torch.Tensor,
        audio_embeds: torch.Tensor,
        audio_placeholder_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Replace embeddings at placeholder positions with projected audio."""
        embed_layer = self._input_embeddings()
        inputs_embeds = embed_layer(input_ids)      # (B, L, D)
        B, L, D = inputs_embeds.shape
        for b in range(B):
            mask = audio_placeholder_mask[b]         # (L,)
            n_slots = int(mask.sum().item())
            if n_slots == 0:
                continue
            audio = audio_embeds[b]                  # (T_ds, D)
            n_use = min(n_slots, audio.size(0))
            slot_idx = mask.nonzero(as_tuple=False).squeeze(-1)[:n_use]
            inputs_embeds[b, slot_idx] = audio[:n_use].to(inputs_embeds.dtype)
        return inputs_embeds

    # ---------------------------------------------------------------- fwd

    def forward(
        self,
        input_features: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        audio_placeholder_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """Standard HF-Trainer-compatible forward.

        Returns an object with a ``.loss`` attribute (or ``None`` when
        ``labels`` is not given) and ``.logits`` of shape ``(B, L, V)``.
        """
        if audio_placeholder_mask is None:
            raise ValueError("SlamAsrModel requires `audio_placeholder_mask` at forward time.")

        audio_embeds = self.encode_audio(input_features)
        inputs_embeds = self._splice_audio_into_embeddings(
            input_ids, audio_embeds, audio_placeholder_mask
        )

        return self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True,
        )

    # ---------------------------------------------------------------- generate

    @torch.no_grad()
    def generate(
        self,
        input_features: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        audio_placeholder_mask: torch.Tensor,
        max_new_tokens: int = 256,
        num_beams: int = 1,
        do_sample: bool = False,
        temperature: float = 1.0,
        repetition_penalty: float = 1.0,
    ) -> torch.Tensor:
        """Autoregressive greedy / beam-search decoding.

        Because we splice audio *embeddings* (not token ids) into the LLM
        input, we must call ``self.llm.generate`` with ``inputs_embeds``
        rather than ``input_ids``.
        """
        audio_embeds = self.encode_audio(input_features)
        inputs_embeds = self._splice_audio_into_embeddings(
            input_ids, audio_embeds, audio_placeholder_mask
        )
        # Trim trailing pads (they should never appear before the
        # generation prompt tail, but be safe).
        gen_kwargs = dict(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            do_sample=do_sample,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            pad_token_id=self.llm_tokenizer.pad_token_id,
            eos_token_id=self.llm_tokenizer.eos_token_id,
        )
        # PEFT-wrapped models transparently forward .generate.
        return self.llm.generate(**gen_kwargs)

    # ---------------------------------------------------------------- IO

    def save_trainable(self, output_dir: str) -> None:
        """Persist only the trainable pieces (projector weights + LoRA).

        This produces a ~50–100 MB checkpoint (vs. multi-GB for the full
        4-bit LLM), which is what we commit to GitHub.
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        # LoRA adapters.
        if self.config.use_lora and hasattr(self.llm, "save_pretrained"):
            self.llm.save_pretrained(os.path.join(output_dir, "lora"))
        # Projector state dict.
        torch.save(self.projector.state_dict(), os.path.join(output_dir, "projector.pt"))
        # Config for reload.
        import json
        with open(os.path.join(output_dir, "slam_config.json"), "w") as fh:
            json.dump(self.config.__dict__, fh, indent=2)

    def load_trainable(self, ckpt_dir: str, map_location: str = "cpu") -> None:
        """Reload the artefacts saved by :meth:`save_trainable`."""
        import os
        proj_path = os.path.join(ckpt_dir, "projector.pt")
        if os.path.exists(proj_path):
            self.projector.load_state_dict(torch.load(proj_path, map_location=map_location))
        lora_path = os.path.join(ckpt_dir, "lora")
        if self.config.use_lora and os.path.isdir(lora_path):
            from peft import PeftModel
            # Reload LoRA on top of the (already-attached) base model.
            self.llm.load_adapter(lora_path, adapter_name="default")
