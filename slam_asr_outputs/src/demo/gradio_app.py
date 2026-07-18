"""Two-tab Gradio demo:

Tab 1 – Live transcriber
    Upload / record a Hindi-English audio clip and see the SLAM-ASR
    transcript.

Tab 2 – Meeting-room agent
    Chain SLAM-ASR + an LLM prompt to produce a bullet-point summary of
    a longer recording.  This is the "Complete Meeting Room AI Agent"
    lab component from Day 5 of the course.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np

from ..data.text_normalization import CodeSwitchTextNormalizer
from ..inference.decode import decode_slam_asr, load_slam_asr_from_checkpoint


def _resample_to_16k(audio: np.ndarray, sr: int) -> np.ndarray:
    if sr == 16000:
        return audio.astype("float32")
    try:
        import librosa
        return librosa.resample(audio.astype("float32"), orig_sr=sr, target_sr=16000)
    except ImportError:  # pragma: no cover
        idx = np.linspace(0, len(audio) - 1, int(len(audio) * 16000 / sr))
        return np.interp(idx, np.arange(len(audio)), audio).astype("float32")


def _chunk_audio(audio: np.ndarray, chunk_s: float = 25.0,
                 overlap_s: float = 1.0, sr: int = 16000) -> list[np.ndarray]:
    """Sliding window chunks for long-form transcription."""
    step = int((chunk_s - overlap_s) * sr)
    win = int(chunk_s * sr)
    if len(audio) <= win:
        return [audio]
    return [audio[i:i + win] for i in range(0, len(audio) - int(overlap_s * sr), step)]


# ---------------------------------------------------------------------------
# Meeting summariser — piggy-backs on the same LLM
# ---------------------------------------------------------------------------


def _summarise_transcript(model: Any, transcript: str,
                          max_new_tokens: int = 256) -> str:
    """Prompt the LLM decoder itself to produce a Hindi-English summary.

    We reuse the already-loaded LLM (post-LoRA merge) rather than pulling
    a second model to keep VRAM low.
    """
    tok = model.llm_tokenizer
    messages = [
        {"role": "system",
         "content": ("You are a helpful assistant that summarises technical "
                     "meeting transcripts.  Keep the same mixed Hindi-English "
                     "style as the input.")},
        {"role": "user",
         "content": (f"Please produce 3-5 bullet-point action items for the "
                     f"following meeting transcript:\n\n{transcript}")},
    ]
    if hasattr(tok, "apply_chat_template") and tok.chat_template:
        prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        prompt = f"{messages[0]['content']}\n\n{messages[1]['content']}\n\nSummary:"
    ids = tok(prompt, return_tensors="pt").to(model.llm.device)
    out = model.llm.generate(
        **ids,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tok.pad_token_id,
        eos_token_id=tok.eos_token_id,
    )
    return tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# Public entry-point
# ---------------------------------------------------------------------------


def launch_demo(
    checkpoint_dir: str | Path,
    device: str = "cuda",
    share: bool = False,
    server_port: Optional[int] = None,
) -> None:
    """Launch the Gradio app."""
    import gradio as gr

    model = load_slam_asr_from_checkpoint(checkpoint_dir, device=device)
    norm = CodeSwitchTextNormalizer()

    def transcribe(audio: Tuple[int, np.ndarray]) -> str:
        if audio is None:
            return ""
        sr, wav = audio
        if wav.dtype.kind == "i":
            wav = wav.astype("float32") / np.iinfo(wav.dtype).max
        wav = _resample_to_16k(wav, sr)
        chunks = _chunk_audio(wav, chunk_s=25.0)
        parts = decode_slam_asr(model, chunks, device=device)
        return norm(" ".join(parts))

    def summarise(audio: Tuple[int, np.ndarray]) -> tuple[str, str]:
        transcript = transcribe(audio)
        if not transcript:
            return "", ""
        return transcript, _summarise_transcript(model, transcript)

    with gr.Blocks(title="SLAM-ASR — Hindi-English Code-Switched Speech") as demo:
        gr.Markdown(
            "## SLAM-ASR — Hindi-English Code-Switched Speech Recognition\n"
            "Frozen Whisper encoder + 4-bit LoRA-adapted Qwen decoder.\n"
        )
        with gr.Tab("Transcribe"):
            audio_in = gr.Audio(sources=["microphone", "upload"],
                                type="numpy",
                                label="Record or upload Hindi-English audio")
            btn = gr.Button("Transcribe")
            text_out = gr.Textbox(label="Transcript", lines=6)
            btn.click(transcribe, inputs=audio_in, outputs=text_out)

        with gr.Tab("Meeting Room Agent"):
            audio_in2 = gr.Audio(sources=["upload"], type="numpy",
                                 label="Upload a meeting recording (≤ 5 min)")
            btn2 = gr.Button("Transcribe + Summarise")
            tr_out = gr.Textbox(label="Full transcript", lines=8)
            sm_out = gr.Textbox(label="Action items", lines=6)
            btn2.click(summarise, inputs=audio_in2, outputs=[tr_out, sm_out])

    demo.launch(share=share, server_port=server_port)


if __name__ == "__main__":  # pragma: no cover
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--port", type=int, default=None)
    args = parser.parse_args()
    launch_demo(args.checkpoint, share=args.share, server_port=args.port)
