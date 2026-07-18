#!/usr/bin/env python3
"""Generate docs/video_cheatsheet.pdf — a printable one-page reference
for filming the SLAM-ASR demo video.

Requires only `reportlab` (already installed system-wide on this machine).
Run:
    python3 scripts/build_video_cheatsheet.py

Output: docs/video_cheatsheet.pdf (portrait A4, one page, ~90 KB).
"""

from __future__ import annotations
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable,
)


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "video_cheatsheet.pdf"


# ---------------------------------------------------------------------------
# Styles — compact so everything fits on one A4 page
# ---------------------------------------------------------------------------

styles = getSampleStyleSheet()

TITLE = ParagraphStyle(
    "Title", parent=styles["Title"],
    fontName="Helvetica-Bold", fontSize=15,
    spaceAfter=1.5 * mm, alignment=1,
)

SUBTITLE = ParagraphStyle(
    "Subtitle", parent=styles["Normal"],
    fontName="Helvetica-Oblique", fontSize=8.5,
    alignment=1, textColor=colors.HexColor("#555555"),
    spaceAfter=3 * mm,
)

SECTION_HEAD = ParagraphStyle(
    "SectionHead", parent=styles["Normal"],
    fontName="Helvetica-Bold", fontSize=8.5,
    textColor=colors.white,
    backColor=colors.HexColor("#2b6cb0"),
    borderPadding=(2, 4, 2, 4),
    spaceBefore=0.5 * mm, spaceAfter=1 * mm,
    leading=11,
)

BODY = ParagraphStyle(
    "Body", parent=styles["Normal"],
    fontName="Helvetica", fontSize=7.7, leading=10,
    spaceAfter=1 * mm,
)

SAY = ParagraphStyle(
    "Say", parent=BODY,
    fontName="Helvetica-Oblique", fontSize=7.7, leading=10,
    textColor=colors.HexColor("#1a4d80"),
    leftIndent=3 * mm, spaceAfter=1 * mm,
)

SHOW = ParagraphStyle(
    "Show", parent=BODY,
    fontName="Helvetica-Bold", fontSize=7.5, leading=10,
    textColor=colors.HexColor("#8b1e00"), spaceAfter=0.5 * mm,
)

FOOTER = ParagraphStyle(
    "Footer", parent=BODY, fontSize=6.8, leading=8.5,
    textColor=colors.HexColor("#333333"),
)


# ---------------------------------------------------------------------------
# Content — each entry is (time, section_title, show_text, say_text)
# ---------------------------------------------------------------------------

SEGMENTS = [
    (
        "0:00 – 0:20",
        "Hook  (20 s)",
        "GitHub repo landing page.",
        "Hi, I\u2019m Ravi Teja Kothuru. For our IIIT-Delhi <i>Building the "
        "Future of Voice &amp; Audio</i> course, I built <b>SLAM-ASR</b> \u2014 an "
        "end-to-end speech recognition system for <b>Hindi-English code-switched</b> "
        "technical lectures. The next six minutes: what I built, a live demo, and \u2014 "
        "most importantly \u2014 the honest story of what worked and what didn\u2019t.",
    ),
    (
        "0:20 – 1:00",
        "Problem + context  (40 s)",
        "Section 4 of training notebook \u2014 manifest_statistics() output.",
        "The MUCS 2021 Subtask-2 dataset is Indian technical-education recordings \u2014 "
        "Linux tutorials, Python lectures. A typical utterance sounds like: "
        "<i>\u2018ab hum terminal khol\u0113nge aur ls command run kar\u0113nge\u2019</i> \u2014 half "
        "Hindi, half English, script-switched mid-sentence. Every second line switches script, "
        "one in four blindtest words is unseen during training. That combination breaks every "
        "off-the-shelf ASR. My project builds one that doesn\u2019t.",
    ),
    (
        "1:00 – 1:45",
        "Architecture  (45 s)",
        "README \u00a71 ASCII diagram OR Section 9 \u201ctrainable / total = 23.97 M / 933 M\u201d line.",
        "Three parts. First, a <b>frozen Whisper-base encoder</b> \u2014 74 M pretrained "
        "params turning audio into features. Never touched. Second, a <b>small trainable "
        "projector</b> \u2014 1-D conv + MLP into the LLM\u2019s embedding space. Third, a "
        "<b>Qwen-2.5 1.5B decoder</b> loaded in 4-bit precision with LoRA adapters. "
        "Only 24 M params train \u2014 2.6 % of the model \u2014 and the whole thing fits into a "
        "single free Kaggle T4 session.",
    ),
    (
        "1:45 – 4:00",
        "Working demo \u2605  (2 min 15 s)",
        "Inference notebook \u2014 slam_asr_inferencing_demo_transcribe.ipynb.",
        "This inference notebook downloads the checkpoint, reconstructs the model on one GPU, "
        "and gives a one-line <font face='Courier'>transcribe()</font> function. Everything renders inline. "
        "<br/><br/><b>Scroll to Section 7</b> \u2014 3 random test utterances. For each: <b>click play</b> "
        "on the audio (2 s), point at reference, point at hypothesis. After the 3rd, say: "
        "<i>\u201cAll three predictions look almost identical. I\u2019ll come back to that.\u201d</i> "
        "<br/><br/><b>Scroll to Section 8</b> \u2014 change <font face='Courier'>IDX = 42</font> to "
        "<font face='Courier'>IDX = 100</font>, re-run. Say: "
        "<i>\u201cNew audio, new reference \u2014 same output. This is the crux of the findings.\u201d</i>",
    ),
    (
        "4:00 – 5:00",
        "Results table  (60 s)",
        "Section 11 of training notebook \u2014 pandas comparison table.",
        "Four models on 300 test + 300 blindtest utterances. <b>Bi-LSTM+CTC</b>: WER 0.998 (near-empty "
        "output). <b>Whisper zero-shot</b>: 1.55 (hallucinates in silences). <b>Whisper fine-tuned</b>: "
        "3.93 (prompt-format drift). <b>SLAM-ASR</b>: <b>2.06 test / 1.63 blindtest</b>. "
        "None would win a leaderboard \u2014 but they\u2019re <i>informatively different</i>. Every "
        "model fails in a distinct way; that\u2019s the real finding.",
    ),
    (
        "5:00 – 5:30",
        "Mode-collapse analysis  (30 s)",
        "Sample predictions from slam_asr_outputs/predictions/test.jsonl.",
        "SLAM-ASR is under-trained. Only 236 optimizer steps on 3 h vs the paper\u2019s 400 GPU-hours. "
        "The projector hasn\u2019t learned input-dependent embeddings yet \u2014 it emits the same seed "
        "sentence for every input. This is <b>mode collapse</b>, a well-documented under-training "
        "pathology. The architecture is sound; it needs ~200\u00d7 more compute. Every SLAM-ASR "
        "prediction is still <b>coherent, bilingual, script-mixed</b> \u2014 more than the other three "
        "models produce. Most promising of the four for scale-up.",
    ),
    (
        "5:30 – 6:00",
        "Wrap-up + GitHub URL  (30 s)",
        "GitHub repo landing page. Hold on screen.",
        "Everything is at <font face='Courier'>github.com/kraviteja95/slam-asr-code-switched-speech-recognition</font>. "
        "Full report in <font face='Courier'>docs/</font>. Both Kaggle notebooks runnable end-to-end. "
        "Trained checkpoint is committed, split into small parts \u2014 <font face='Courier'>REASSEMBLE.md</font> "
        "in each folder explains how. Thanks for watching.",
    ),
]


# ---------------------------------------------------------------------------
# Build the PDF
# ---------------------------------------------------------------------------


def build() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)

    doc = SimpleDocTemplate(
        str(OUT),
        pagesize=A4,
        leftMargin=12 * mm, rightMargin=12 * mm,
        topMargin=10 * mm, bottomMargin=10 * mm,
        title="SLAM-ASR video cheat-sheet",
        author="Ravi Teja Kothuru",
    )

    story = []

    # ---- Title ----
    story.append(Paragraph(
        "SLAM-ASR \u2014 Video Presentation Cheat-Sheet", TITLE))
    story.append(Paragraph(
        "Target 6:00 (range 5\u20137). One long take. QuickTime \u21d2 File \u21d2 New Screen Recording. "
        "Both notebooks open with outputs baked in. Cursor size boosted. Font zoom 120 %.",
        SUBTITLE,
    ))

    # ---- Time overview table ----
    overview_data = [
        ["Time", "Segment", "On-screen"],
        ["0:00\u20130:20", "Hook", "Repo landing page"],
        ["0:20\u20131:00", "Problem + context", "Notebook \u00a74 (manifest stats)"],
        ["1:00\u20131:45", "Architecture", "README \u00a71 or Section 9 output"],
        ["1:45\u20134:00", "Working demo \u2605", "Inference notebook"],
        ["4:00\u20135:00", "Results", "Notebook \u00a711 table"],
        ["5:00\u20135:30", "Mode-collapse", "Sample predictions"],
        ["5:30\u20136:00", "Wrap-up", "Repo landing page"],
    ]
    overview = Table(
        overview_data,
        colWidths=[24 * mm, 45 * mm, None],
        hAlign="LEFT", rowHeights=None,
    )
    overview.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2b6cb0")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 7.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("ALIGN", (0, 0), (0, -1), "LEFT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.HexColor("#f4f7fb"), colors.white]),
        ("LINEBELOW", (0, 0), (-1, 0), 0.5, colors.HexColor("#2b6cb0")),
        ("BOX", (0, 0), (-1, -1), 0.25, colors.HexColor("#cccccc")),
        ("INNERGRID", (0, 0), (-1, -1), 0.15, colors.HexColor("#dddddd")),
    ]))
    story.append(overview)
    story.append(Spacer(1, 3 * mm))

    story.append(HRFlowable(width="100%", thickness=0.5,
                            color=colors.HexColor("#aaaaaa"),
                            spaceBefore=1, spaceAfter=2))

    # ---- Beat-by-beat segments ----
    for time_range, title, show_text, say_text in SEGMENTS:
        header = f"<b>{time_range}</b> \u00b7 {title}"
        story.append(Paragraph(header, SECTION_HEAD))
        story.append(Paragraph(f"SHOW \u2192 {show_text}", SHOW))
        story.append(Paragraph(f"SAY \u2192 \u201c{say_text}\u201d", SAY))

    # ---- Footer with quick-reference tips ----
    story.append(HRFlowable(width="100%", thickness=0.5,
                            color=colors.HexColor("#aaaaaa"),
                            spaceBefore=2, spaceAfter=2))
    story.append(Paragraph(
        "<b>Before recording:</b> Do-Not-Disturb ON \u00b7 Notifications silenced \u00b7 "
        "Notebooks open with outputs \u00b7 Mic checked \u00b7 10-s test recording done. "
        "<b>While recording:</b> One long take \u00b7 Don\u2019t rush \u00b7 2-s pauses OK \u00b7 "
        "Small verbal stumbles fine. "
        "<b>After:</b> Trim dead space \u00b7 Export MP4 H.264 \u00b7 \u2264 500 MB \u00b7 "
        "Save as <font face='Courier'>docs/demo.mp4</font> \u00b7 "
        "Verify in QuickTime + VLC before committing.",
        FOOTER,
    ))

    doc.build(story)
    size_kb = OUT.stat().st_size / 1024
    print(f"Wrote {OUT}  ({size_kb:.1f} KB)")


if __name__ == "__main__":
    build()
