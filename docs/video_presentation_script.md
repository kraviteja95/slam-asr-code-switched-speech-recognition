# Video presentation script — SLAM-ASR (target 6 minutes, range 5–7)

> **Recording tool:** QuickTime (macOS built-in), OBS, or Loom.  
> **What you're recording:** two Kaggle notebooks + GitHub repo, one long screen-capture take with voice-over.  
> **Output file:** `docs/demo.mp4` at 1080p / 30 fps, H.264, ≤ 500 MB.  
> **Companion:** [`docs/video_cheatsheet.pdf`](video_cheatsheet.pdf) — printable one-page version of the same script.

The video needs to satisfy the course brief: *"a video presentation in .mp4 format about the project, showcasing the working demo/prototype."* The single most important thing is to show the model actually running on real audio.

---

## Total budget: 6:00 (targeted 5:45 – 6:15)

| Time | Segment | On-screen | Duration |
|---|---|---|---|
| 0:00 – 0:20 | Hook | Title slide or repo landing page | 20 s |
| 0:20 – 1:00 | Problem + course context | Section 4 of training notebook | 40 s |
| 1:00 – 1:45 | Architecture | README §1 ASCII diagram | 45 s |
| 1:45 – 4:00 | **Working demo** ⭐ | Inference notebook running | 2:15 |
| 4:00 – 5:00 | Results table + interpretation | Section 11 of training notebook | 1:00 |
| 5:00 – 5:30 | Honest mode-collapse analysis | Sample predictions | 30 s |
| 5:30 – 6:00 | Wrap-up + GitHub URL | Repo page | 30 s |

---

## Beat-by-beat script

### 0:00 – 0:20 · Hook (20 s)

**Show:** GitHub repo landing page — `https://github.com/kraviteja95/slam-asr-code-switched-speech-recognition`

**Say:**

> "Hi, I'm Ravi Teja Kothuru. For our IIIT-Delhi *Building the Future of Voice & Audio* course, I built **SLAM-ASR** — an end-to-end speech recognition system for **Hindi-English code-switched** technical lectures. The next six minutes: what I built, a live demo, and — most importantly — the honest story of what worked and what didn't."

---

### 0:20 – 1:00 · Problem + why it's hard (40 s)

**Show:** Section 4 of the training notebook — the `manifest_statistics(...)` output showing ~52 k train utts, ~55 % code-switched, ~25 % OOV on blindtest.

**Say:**

> "The MUCS 2021 Subtask-2 dataset is Indian technical-education recordings — Linux tutorials, Python lectures, LibreOffice demos. A typical utterance sounds like: *'अब हम terminal खोलेंगे और ls command run करेंगे.'* Every second line switches script, and one in four words on the blind test set is completely unseen during training. That combination breaks every off-the-shelf ASR system. My project is to build one that doesn't break."

---

### 1:00 – 1:45 · Architecture (45 s)

**Show:** The ASCII diagram from `README.md` §1, **OR** Section 9's `[SLAM-ASR] trainable / total = 23.97 M / 933.18 M (2.569%)` output line.

**Say:**

> "The architecture has three parts. First, a **frozen Whisper-base encoder** — 74 million pretrained parameters that turn 30 seconds of audio into perceptual features. I never touch these weights. Second, a **small trainable projector** — a 1-D convolution plus MLP that adapts Whisper's features into the LLM's embedding space. Third, a **Qwen-2.5 1.5-billion-parameter decoder** loaded in 4-bit precision with LoRA adapters. The result: only 24 million parameters actually train — that's 2.6 % of the model — and the whole thing fits into a single free Kaggle T4 session."

---

### 1:45 – 4:00 · Working demo ⭐ (2 minutes 15 seconds)

**Show:** Switch to `notebooks/slam_asr_inferencing_demo_transcribe.ipynb` — the executed inference notebook.

**Say (walk through as you scroll):**

> "This is the inference-only notebook. It downloads the trained checkpoint from a private Kaggle Dataset, reconstructs the model on a single GPU for fast decoding, and gives you a one-line `transcribe()` function. Everything renders inline — audio players and transcripts in the same notebook cell."

**Scroll to Section 7 (the "3 random test utterances" cell). For each of the three samples:**

1. **Click ▶ Play** on the audio widget (2 s of audio).
2. **Point at the reference line** — "here's the ground-truth transcript, mixing Devanagari and Roman."
3. **Point at the hypothesis line** — "and here's what the model actually predicted."

**After the third sample, say:**

> "Notice something? All three predictions look almost identical. That's not a coincidence — I'll come back to that in one minute."

**Scroll to Section 8 (the transcribe-by-index cell).**

**Say:**

> "You can also transcribe a specific utterance by changing this `IDX` value. Let me pick a different one — say 100."

**Change `IDX = 42` to `IDX = 100`, run the cell.**

> "New audio, new reference — same output. This is a real, reproducible failure mode, and it's the crux of my findings."

---

### 4:00 – 5:00 · Results (60 s)

**Show:** Section 11 of the training notebook — the pandas comparison table.

**Say (read the numbers as you point at each row):**

> "Here are the four models I evaluated on 300 test and 300 blindtest utterances. **Bi-LSTM plus CTC** — the Day-3 classical baseline — gets a Word Error Rate of 0.998. That's essentially 100 %, but not for the reason you'd think. **Whisper zero-shot** is 1.55 — over 100 % because it hallucinates content in silences. **Whisper naive fine-tuning made things worse** at 3.93. And **SLAM-ASR — the star of the project** — comes in at 2.06 on test and 1.63 on blindtest."

**Say (leaning into the honesty):**

> "None of these numbers would win a Kaggle leaderboard. But they're *informatively different* — every model fails in a distinct way, and that's the real finding of this project."

---

### 5:00 – 5:30 · Mode-collapse analysis (30 s)

**Show:** Sample predictions from `slam_asr_outputs/predictions/test.jsonl` (or Section 10's per-utterance printout).

**Say (pace slowly, this is the "aha" moment):**

> "SLAM-ASR is under-trained. After only 236 optimizer steps on 3 hours of data — against the reference paper's 400 GPU-hours — the alignment projector hasn't learned to produce audio-dependent embeddings yet. It produces the same seed sentence for every input. This is called **mode collapse** — a well-documented under-training pathology. The architecture is sound; it just needs about 200 times more compute. Every SLAM-ASR prediction is still a **coherent, bilingual, script-mixed sentence** — which is more than the other three models produce. That makes SLAM-ASR the most promising of the four for scale-up."

---

### 5:30 – 6:00 · Wrap-up + GitHub URL (30 s)

**Show:** GitHub repo landing page. Hold it on screen.

**Say:**

> "Everything is on GitHub at `github.com/kraviteja95/slam-asr-code-switched-speech-recognition`. The full report with all numbers is under `docs/`. Both Kaggle notebooks are runnable end-to-end — one for training and evaluation, one for the demo you just saw. The trained checkpoint is committed, split into small parts because of GitHub's file-size limit, with a `REASSEMBLE.md` in every folder. Thanks for watching."

---

## Filming logistics

### Before you record

1. Open both notebooks in JupyterLab or VS Code with **outputs already visible** (they have cell outputs baked in).
2. Open your GitHub repo in a browser tab.
3. Boost cursor size — *System Settings → Accessibility → Display → Pointer size*.
4. Zoom notebook fonts to ~120 %.
5. Do-Not-Disturb ON. Silence Slack, Mail, everything.
6. Test recording — 10 seconds — verify mic + screen capture both work.

### While recording

- **Take one long take**, not many short takes. Editing kills momentum.
- Use QuickTime → *File → New Screen Recording*.
- Include the whole notebook window, or split-screen with your face in the bottom-right if you want a webcam overlay.
- **Don't rush.** 6 minutes is plenty — 2-second pauses between beats are fine.
- If you slip up on a sentence, keep going — you can cut in post if truly needed, but usually a small verbal stumble sounds more natural than perfection.

### After recording

- Trim any dead space at the start / end (QuickTime → *Edit → Trim*).
- Export as MP4 (H.264 video, AAC audio), ≤ 500 MB.
- Save as `docs/demo.mp4`.
- Verify plays cleanly in QuickTime **and** VLC before committing to GitHub.

---

## Adjustments if you want ≠ 6 minutes

**To go under 5 min (~4:30):**

- Cut the specific-utterance-by-index demo in Section 1:45-4:00 (~30 s).
- Compress the results narration (~15 s).

**To go over 6 min toward 7:**

- Add a 30-second dataset walkthrough in Section 4 of the training notebook — histograms, script-mix analysis — between the current 1:00 and 1:45 marks.
- Show one *specific correct prediction* if any — browse `slam_asr_outputs/predictions/test.jsonl` for a hypothesis that partially matches its reference — to demonstrate the model isn't 100 % broken.

---

## The one-sentence pitch

If someone asks "what's this project about?" during the video, your answer should be:

> **"I built a working end-to-end pipeline for Hindi-English code-switched speech recognition, trained a state-of-the-art architecture end-to-end in one free Kaggle session, and characterised exactly why it doesn't yet match the paper's numbers."**

That's the story. The 6-minute script above delivers it with evidence.
