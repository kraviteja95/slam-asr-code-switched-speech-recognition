# Video presentation script — SLAM-ASR (target 4–6 minutes)

> **Recording tool:** QuickTime (macOS built-in), OBS, or Loom. Any works.  
> **What you're recording:** the two executed Kaggle notebooks scrolling by while you narrate.  
> **Output file:** `docs/demo.mp4` at 1080p / 30 fps, H.264, ≤ 500 MB.

The demo does **not** need Gradio. Everything you need — audio players, transcripts, WER numbers, mode-collapse evidence — is already baked into the two `.ipynb` files at `notebooks/`.

---

## Recording setup (5 minutes of prep)

1. Open both notebooks side-by-side in VS Code or JupyterLab:
   - Left: [`notebooks/slam_asr_all_in_one_kaggle.ipynb`](../notebooks/slam_asr_all_in_one_kaggle.ipynb)
   - Right: [`notebooks/slam_asr_inferencing_demo_transcribe.ipynb`](../notebooks/slam_asr_inferencing_demo_transcribe.ipynb)
2. Boost cursor size (System Settings → Accessibility → Display → Pointer size).
3. Zoom notebook fonts to ~120 % so text is legible in the recording.
4. Turn on Do-Not-Disturb (silences notifications for the take).
5. Have your microphone ready — headset or lav mic; do **not** use the built-in laptop mic (echo).
6. Do a 10-second test recording first, check that audio and video both landed.

---

## Beat sheet (2-column: time / what to say and do)

### 0:00 – 0:30 · Hook

**Say (calmly, no rush):**

> "Hi, I'm *<your name>*. For our IIIT-D Speech course project, I built SLAM-ASR — an end-to-end speech recognition system for **Hindi-English code-switched** technical lectures. The data mixes Devanagari and Roman script in every second sentence, which breaks every off-the-shelf ASR system. Let me walk you through what I built, what worked, and what didn't."

**Do:** Scroll to the top of the training notebook. Point at the title cell.

---

### 0:30 – 1:15 · The course-to-code map

**Say:**

> "The project covers every day of the 5-day course. Signal fundamentals and psychoacoustics — Days 1 and 2 — are Sections 4 and 5 of this notebook. Classical GMM is Section 6. Bi-LSTM plus CTC is Section 7. Whisper is Section 8. SLAM-ASR itself, which is Days 4 and 5, is Section 9. Evaluation is Section 10."

**Do:** Slowly scroll through the notebook, pausing at each section header. Point at the running headers `## 4.  Dataset EDA`, `## 5.  Perceptual features`, `## 6.  Classical baseline`, etc.

---

### 1:15 – 2:00 · Dataset in 45 seconds

**Say:**

> "MUCS 2021 Subtask-2 ships 90 hours of Hindi-English code-switched Spoken Tutorial lectures. Fifty thousand utterances, five hundred speakers. Half of them contain at least one code-switch. Twenty-five percent of blindtest tokens are Out-Of-Vocabulary. That's why classical vocabulary-based ASR doesn't work here — every third English word is unseen."

**Do:** Scroll to Section 4 of the training notebook. Point at:

- The `manifest_statistics(...)` output for train / test / blindtest (rows counts, hours, code-switch rates).
- The duration histogram figure.
- The OOV rate print-out.

---

### 2:00 – 2:45 · Architecture

**Say:**

> "SLAM-ASR is a tripartite design. A frozen Whisper-base encoder gives me 74 million parameters of pretrained acoustic features — I never touch these weights. A small trainable projector, a 1D convolution plus MLP, adapts those features into the LLM's embedding space. And a Qwen-2.5 1.5B-parameter decoder handles the language modelling — but it's loaded in 4-bit precision with LoRA adapters, so only 24 million parameters actually train. That's 2.6 % of the model, and it fits in a free Kaggle T4."

**Do:** Show the ASCII architecture diagram from the notebook's Section 0 header (or open `README.md` §1). Point at the "Trainable / total = 23.97 M / 933.18 M" line in Section 9's output.

---

### 2:45 – 3:30 · Live demo — the inference notebook

**Do:** Switch to the second notebook (`slam_asr_inferencing_demo_transcribe.ipynb`).

**Say:**

> "Here's the model actually running. This is the inference-only notebook — you attach the checkpoint as a Kaggle Dataset, run all the cells, and you get transcripts inline in a browser. No Gradio, no external URL, just cells with audio players and text outputs. Watch — here are three random utterances from the test split."

**Do:** Scroll to Section 7 (the "3 random test utterances" cell). For each of the three samples:

1. Play the audio player briefly (2-3 seconds of audio each).
2. Point at the reference text.
3. Point at the hypothesis text.

Then scroll to Section 8 (the "specific utterance by index" cell). Change `IDX = 42` to a different number, re-run, show another transcript.

---

### 3:30 – 4:30 · Results — and the honest story

**Do:** Switch back to the training notebook. Scroll to Section 11 (the comparison table cell).

**Say (this is the crucial narrative):**

> "Here are the numbers. Bi-LSTM CTC gets WER of 1.0 — but that's an artefact of the model producing near-empty output, not genuine skill. Whisper zero-shot is 1.55 because it hallucinates captions in the padded silences. Naive Whisper fine-tuning made it *worse*, at 3.9, because of a prompt-format mismatch between training and inference. Our SLAM-ASR ends up at 2.06 on test and 1.63 on blindtest."

**Say (turn to the mode-collapse example):**

> "But the story doesn't end at the numbers. If you look at what our model actually predicts — every input, no matter what audio, produces the same seed sentence."

**Do:** Show one or two of the reference/hypothesis lines from Section 10's per-utterance printout. Point at three consecutive lines where the hypothesis is literally identical.

**Say:**

> "This is called **mode collapse**. It happens when the alignment projector has been trained for far too few optimizer steps — in our case, 236 steps on 3 hours of data, versus the reference SLAM-LLM paper which uses about 400 GPU-hours. The architecture is sound; the projector just hasn't seen enough gradient updates to learn input-dependent embeddings. Scaling training data and steps by 100× is the direct route to competitive WER — but that's outside a free Kaggle session."

---

### 4:30 – 5:15 · Wrap-up

**Say:**

> "Everything is on GitHub — the two Kaggle notebooks, the trained checkpoint, all predictions, and the full report. The training notebook is fully self-contained; upload it to Kaggle, attach the dataset, click Run All, done. Thanks for watching."

**Do:** Show the GitHub URL on-screen as a title bar or your browser tab. Hold for the last 5 seconds so viewers can screenshot the URL.

---

## Filming checklist

Before you hit Record:

- [ ] Screen: 1920 × 1080 or higher, 30 fps.
- [ ] Cursor size boosted.
- [ ] Microphone: headset or lav; **not** the laptop mic.
- [ ] All notebook cells pre-executed (outputs visible without re-running).
- [ ] Both notebook files open in separate windows / tabs.
- [ ] Do-Not-Disturb turned on.
- [ ] Notifications silenced (Slack, Mail, etc.).
- [ ] Test recording done and audio + video verified.

After you hit Stop:

- [ ] Export as MP4 (H.264 video, AAC audio).
- [ ] Trim any dead space at the start/end.
- [ ] File size ≤ 500 MB.
- [ ] Filename: `docs/demo.mp4`.
- [ ] Verify plays back cleanly in QuickTime + VLC before committing.

---

## If you want to keep the running time under 4 minutes

Drop the Section-3 dataset EDA walkthrough (1:15 → 2:00) and go straight from the course-to-code map (0:30 – 1:15) to the architecture (2:00 – 2:45). That saves 45 seconds without losing the core narrative.

## If you want it to be tighter than 5 minutes

Skip the specific-utterance-by-index demo (3:20 – 3:30) and just do the "3 random utterances" cell. Saves 30 seconds and the story is unchanged.
