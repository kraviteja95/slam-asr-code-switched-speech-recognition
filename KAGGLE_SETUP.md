# Running the project on Kaggle — step-by-step

> Time to first WER number: **~5 minutes of clicking + ~90 minutes of GPU time on a Kaggle T4 x2**.

The project ships as **two Kaggle notebooks**:

| Notebook | Purpose | Runtime |
|---|---|---|
| [`notebooks/slam_asr_all_in_one_kaggle.ipynb`](notebooks/slam_asr_all_in_one_kaggle.ipynb) | Full pipeline: EDA → GMM → Bi-LSTM+CTC → Whisper → SLAM-ASR training → evaluation | ~90 min |
| [`notebooks/slam_asr_inferencing_demo_transcribe.ipynb`](notebooks/slam_asr_inferencing_demo_transcribe.ipynb) | Load the trained checkpoint and transcribe MUCS clips inline (audio player + reference + hypothesis) | ~5 min |

Both are **100 % self-contained** — they bootstrap the required source code from an embedded JSON blob, no `git clone` or extra dataset upload of code needed.

---

## Part A — Run the full training pipeline (first time)

### 1. Create a new Kaggle notebook

- Go to https://www.kaggle.com/code → **+ New Notebook**.

### 2. Import the training notebook

- Top-left → **File → Import Notebook**.
- Upload `notebooks/slam_asr_all_in_one_kaggle.ipynb` from your local repo.
- Rename the notebook to something meaningful, e.g. `SLAM-ASR — Hindi-English code-switched ASR`.

### 3. Attach the dataset

- Right sidebar → **Input → + Add Input**.
- Search for `mucs-2021-hindi-english-code-switched-speech`.
- Add **`kraviteja95/mucs-2021-hindi-english-code-switched-speech`**.
- Kaggle mounts it at `/kaggle/input/datasets/kraviteja95/mucs-2021-hindi-english-code-switched-speech/mucs/`.

### 4. Turn on the GPU and Internet

- Right sidebar → **Session options** → **Accelerator: GPU T4 x2**.
- Same panel → **Internet: On** — required so HuggingFace can download Whisper-base and Qwen-2.5-1.5B.

> Kaggle gives ~30 GPU-hours per week for free — plenty for one full training run.

### 5. Run the notebook

Click **Run All** at the top. The notebook is organised into these sections:

| Section | What it does | Time |
|---|---|---|
| 0 | Bootstrap the `src/` package (writes to `/kaggle/working/src/` from an embedded blob) | 30 s |
| 1 | Global switches (`QUICK_MODE`, batch sizes, etc.) | — |
| 2 | Install pinned packages | 1 min |
| 3 | Prepare manifests (rewrite paths to Kaggle mount) | 30 s |
| 4 | Dataset EDA — signal fundamentals + trait taxonomy (Days 1, 2) | 2 min |
| 5 | Perceptual features — Mel / Bark / ATH / masking (Day 1) | 30 s |
| 6 | Classical baseline — GMM keyword spotter (Day 2) | 5 min |
| 7 | Neural baseline — Bi-LSTM + CTC (Day 3) | 10 min |
| 8 | Whisper zero-shot + fine-tune (Day 4) | 25 min |
| 9 | **SLAM-ASR training** (Days 4–5) | ~45 min |
| 10 | Evaluation on test + blindtest → `results/summary.json` | ~15 min |
| 11 | Comparison table (paste into report) | — |
| 12 | Gradio demo (optional) | starts a public URL |
| 13 | Zip everything under `/kaggle/working` for one-click download | 10 s |

`QUICK_MODE = True` in Section 1 uses a small subset that fits inside a Kaggle session. Set to `False` to use the full 90 h train set (needs multiple sessions).

### 6. Download your outputs

The last section builds `slam_asr_outputs.zip` (~180 MB) under `/kaggle/working/`. From the right sidebar → **Output** tab → click the ⋮ next to the zip → **Download**.

On your laptop, unzip it under `release/slam_asr_outputs/` in this repo. That folder is git-ignored — it stays local.

The essentials inside the zip:

- `checkpoints/slam_asr/final/` — 91 MB of **your trained weights** (LoRA adapter + projector). This is the only thing you actually need to keep.
- `predictions/{test,blindtest}.jsonl` — 300 evaluated utterances each.
- `results/summary.json` — WER / CER / per-language WER / OOV numbers for every model.

---

## Part B — Run just the inference demo (subsequent sessions)

After the training run, you don't want to re-train for every demo. Use the shorter inference notebook instead.

### 1. One-time — upload the checkpoint as a private Kaggle Dataset

On your laptop:

```bash
cd release/slam_asr_outputs/checkpoints/slam_asr
zip -r ~/Desktop/slam_asr_checkpoint.zip final/
```

Then https://www.kaggle.com/datasets → **+ New Dataset** → upload the zip → name it exactly **`slam-asr-checkpoint`** → Private → *Create*.

Kaggle unzips it automatically. The `final/` folder lands under `/kaggle/input/slam-asr-checkpoint/final/`.

### 2. Every future demo session

- New Kaggle notebook → **File → Import Notebook** → upload `notebooks/slam_asr_inferencing_demo_transcribe.ipynb`.
- Add inputs:
  - `kraviteja95/mucs-2021-hindi-english-code-switched-speech`
  - `kraviteja95/slam-asr-checkpoint` (your new one)
- Session options: **GPU T4** (single GPU is enough for inference), **Internet: On**.
- Click **Run All** → ~5 min → cells produce inline audio players + reference + hypothesis for 3 random test utterances plus a per-index picker.

---

## Frequently asked

**Q. Do I need to split the data?**  
No. The dataset ships pre-split into `train/`, `test/`, `blindtest/`. The notebooks consume them as-is. `QUICK_MODE` only *subsamples* the training split for time budget.

**Q. Where do checkpoints go on Kaggle?**  
`/kaggle/working/checkpoints/slam_asr/final/` (~91 MB — LoRA + projector). Whisper-base and Qwen-2.5-1.5B are re-downloaded from HuggingFace at load time (that's why Internet must be on).

**Q. Kaggle ran out of memory / a cell crashed.**  
Common tweaks in the top-of-notebook config cell:

- Reduce training data: `FAST_SLAM_TRAIN_HOURS = 3.0` (or lower)
- Shorter clips: `FAST_MAX_DURATION = 15.0`
- Smaller batch: `SLAM_BATCH_SIZE = 2`, `GRAD_ACCUM_STEPS = 4`

If VRAM specifically is the issue, the notebook already loads Qwen with `device_map='auto'` to split it across both T4 GPUs.

**Q. My Kaggle session timed out mid-training.**  
The notebook writes lightweight rescue checkpoints (`ckpt-50`, `ckpt-100`, …) under `/kaggle/working/checkpoints/slam_asr/` every 50 optimizer steps. Look for the latest one and copy it to `.../final` (or use the rescue cell in Section 9).

**Q. Do I need Gradio?**  
No. Gradio is included as an option in Section 12 of the training notebook, but the recommended demo is the inference notebook (Part B) which shows everything inline — audio players, references, hypotheses — with no live tunnel that expires.

**Q. What if the dataset mount path differs?**  
Section 3 of the training notebook uses this default:

```python
DATASET_ROOT = '/kaggle/input/mucs-2021-hindi-english-code-switched-speech/mucs'
```

But Kaggle sometimes mounts private datasets under `/kaggle/input/datasets/<user>/<slug>/...`. The demo notebook auto-detects multiple candidates. For the training notebook, if the assertion in Section 3 fails, edit that line to match `os.listdir('/kaggle/input/')` and re-run only Section 3.
