"""Alignment projector — bridges Whisper encoder features into the LLM space.

Whisper-base emits one hidden state every 20 ms (50 Hz) with 512
dimensions.  The LLM's context is precious (each token is ~4 kB of KV
cache in fp16), so we downsample the audio-token stream with a 1-D CNN
before a two-layer MLP projects it into the LLM's embedding dimension.

Given 30 s of audio → 1500 encoder frames → 300 audio tokens with
downsample=5 → fits comfortably alongside a 128-token prompt in Qwen's
2 048-token window.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class ConvProjector(nn.Module):
    """1-D CNN downsampler followed by an MLP projector.

    Parameters
    ----------
    in_dim:
        Whisper hidden-state dimension (512 for whisper-base,
        768 for whisper-small, 1 024 for whisper-medium).
    out_dim:
        LLM embedding dimension (1 536 for Qwen-2.5-1.5B,
        4 096 for LLaMA-3-8B).
    hidden_dim:
        MLP hidden width.
    downsample_factor:
        Strided-convolution factor.  Total audio token count is
        ``ceil(T_encoder / downsample_factor)``.
    n_conv_layers:
        Number of ``stride=1`` conv layers *before* the strided one.
        Extra layers give the projector a wider receptive field to
        smooth over encoder features before the drastic downsample.
    dropout:
        Applied after each hidden activation.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int = 2048,
        downsample_factor: int = 5,
        n_conv_layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.downsample_factor = int(downsample_factor)

        # Optional stride-1 conv layers to expand the receptive field.
        conv_stack: list[nn.Module] = []
        cur = in_dim
        for _ in range(max(0, n_conv_layers - 1)):
            conv_stack += [
                nn.Conv1d(cur, cur, kernel_size=3, padding=1),
                nn.GELU(),
            ]
        # Final strided conv performs the actual downsampling.
        conv_stack += [
            nn.Conv1d(
                cur, cur,
                kernel_size=self.downsample_factor,
                stride=self.downsample_factor,
                padding=0,
            ),
            nn.GELU(),
        ]
        self.conv = nn.Sequential(*conv_stack)

        # MLP: hidden → GELU → hidden → GELU → out
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )
        # Small init so training is stable at step 0.
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def output_length(self, encoder_length: int) -> int:
        """Length of the audio-token sequence after downsampling."""
        return math.ceil(encoder_length / self.downsample_factor)

    def forward(self, encoder_hidden: torch.Tensor) -> torch.Tensor:
        """Project ``encoder_hidden`` of shape ``(B, T, D_enc)`` to
        ``(B, T_ds, D_llm)``.
        """
        # Conv expects (B, D, T)
        x = encoder_hidden.transpose(1, 2)
        x = self.conv(x)
        x = x.transpose(1, 2)          # → (B, T_ds, D_enc)
        return self.mlp(x)              # → (B, T_ds, D_llm)
