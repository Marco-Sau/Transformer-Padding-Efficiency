# Complete Colab Setup Guide — Step by Step

This guide explains **everything** you need to do to run the experiments from scratch, assuming no prior experience with Google Colab or machine learning tools.

---

## What You Will Need

- A **Google account** (free Gmail account works)
- A **Google Colab Pro** subscription (~€11/month) — the free tier does not provide enough GPU time
- A computer with a web browser
- This project folder

> 💡 **Why Colab Pro?** Training 36 individual models, each for 3 epochs on a large dataset, takes approximately 18 hours of total GPU time. The free Colab tier disconnects sessions after ~1.5 hours of GPU use and provides slower GPUs. Colab Pro gives you priority access to the T4 GPU (16 GB VRAM) and longer session windows.

---

## Part 1 — Google Account and Colab Pro Setup

### Step 1.1 — Create or Log In to Your Google Account

1. Go to [accounts.google.com](https://accounts.google.com)
2. Sign in to your existing account, or click **Create account** to make a new one

### Step 1.2 — Subscribe to Google Colab Pro

1. Go to [colab.research.google.com](https://colab.research.google.com)
2. Click the **Settings** cog (top right) → **Manage subscriptions**, or navigate directly to [colab.research.google.com/signup](https://colab.research.google.com/signup)
3. Choose **Colab Pro** and complete payment

---

## Part 2 — Upload the Project to Google Drive

Google Drive is Google's cloud storage service. The Colab notebook reads your code files directly from Drive, so you must upload the project there first.

### Step 2.1 — Open Google Drive

1. Go to [drive.google.com](https://drive.google.com)
2. Make sure you are signed in with the same Google account you used for Colab

### Step 2.2 — Upload the Project Folder

1. In Google Drive, click **+ New** (top-left corner) → **Folder upload**
2. In the file dialog that appears, navigate to this `Project/` folder on your computer and select it
3. Google Drive will upload the entire folder with all its contents — this may take a few minutes depending on your internet connection
4. When the upload finishes, you should see a folder named `Project` in your Drive

### Step 2.3 — Rename the Folder

The notebook expects the folder to be named exactly **`llm_efficiency_study`**:

1. Right-click (or Ctrl-click on Mac) on the newly uploaded `Project` folder
2. Click **Rename**
3. Type `llm_efficiency_study` and press Enter

Your Drive should now contain:
```
My Drive/
└── llm_efficiency_study/     ← this is the crucial folder
    ├── src/
    ├── configs/
    ├── notebooks/
    ├── results/
    ├── figures/
    ├── run_experiments.py
    ├── requirements.txt
    └── report.tex
```

---

## Part 3 — Open the Notebook in Colab

### Step 3.1 — Navigate to the Notebook File

1. In Google Drive, double-click on `llm_efficiency_study` to open it
2. Double-click on the `notebooks/` folder
3. Double-click on `colab_notebook_extended.ipynb`
4. The file will open in **Google Colaboratory** automatically (if it doesn't, right-click → Open with → Google Colaboratory)

### Step 3.2 — Enable the GPU

By default, Colab runs without a GPU (just a CPU), which would make training ~100× slower. You must manually enable the GPU:

1. In the Colab menu at the top, click **Runtime**
2. Click **Change runtime type**
3. Under **Hardware accelerator**, select **T4 GPU**
4. Click **Save**

You will see a green indicator at the top right of the page once the GPU is connected. If it says "Connecting..." wait a few seconds.

> ⚠️ **Every time you close and reopen the notebook**, you need to repeat this step. Colab does not remember your runtime choice between sessions.

---

## Part 4 — Understanding the Notebook Structure

The notebook is divided into numbered sections. Each section is a **cell** (a box containing code). You run cells one by one by clicking the ▶ play button on the left side of each cell, or by pressing **Shift+Enter** on your keyboard.

> ⚠️ **You must run cells in order, from top to bottom.** Each cell depends on the ones above it. Do not skip cells.

Here is what each section does:

### Cell 1 — Mount Google Drive and Set Paths

```python
from google.colab import drive
drive.mount('/content/drive')
PROJECT_ROOT = '/content/drive/MyDrive/llm_efficiency_study'
```

**What it does:** Connects the notebook to your Google Drive so it can read your files.

**What to expect:** A pop-up will ask you to grant permission. Click **Connect to Google Drive**, choose your account, and click **Allow**. You will see `Mounted at /content/drive` in the output.

**If it fails:** Make sure you renamed the folder to `llm_efficiency_study` exactly (see Step 2.3).

---

### Cell 2 — Install Python Dependencies

```python
!pip install transformers==4.44.0 datasets accelerate peft ...
```

**What it does:** Installs the specific versions of libraries needed to run the experiments. This is like installing apps on your phone — it only needs to run once per session.

**What to expect:** Lots of text output showing packages being downloaded and installed. This takes 3–5 minutes. A warning like `WARNING: pip is configured with locations that require TLS/SSL` is normal and can be ignored.

**If it asks to restart:** Click **Restart session** if prompted. Then re-run Cell 1 (Drive mount) and Cell 2.

---

### Cell 3 — Pre-flight Test

**What it does:** Verifies that all code files are accessible and the GPU is working correctly.

**What to expect:** Output like:
```
✅ Project path: /content/drive/MyDrive/llm_efficiency_study
✅ src/ found
✅ GPU available: Tesla T4 (15.78 GB)
✅ All imports successful
PRE-FLIGHT TEST PASSED
```

**If it fails with `No module named 'src'`:** The project path is wrong. Make sure the Drive folder is named `llm_efficiency_study` (not `Project` or anything else).

**If it fails with `GPU not available`:** Go back to **Runtime → Change runtime type** and select T4 GPU, then reconnect.

---

### Training Cells (one per model per dataset)

There are 6 training cells:
1. **IMDb: BERT-base** (~3.5 hours)
2. **IMDb: DistilBERT-base** (~1.7 hours)
3. **IMDb: RoBERTa-base** (~3.5 hours)
4. **Emotion: BERT-base** (~30 minutes)
5. **Emotion: DistilBERT-base** (~18 minutes)
6. **Emotion: RoBERTa-base** (~30 minutes)

**What each one does:** Trains the specified model on the dataset with both static and dynamic padding, across 3 different random seeds. That is 6 individual training runs per cell (2 strategies × 3 seeds).

**What to expect during training:**
- A progress bar like `Running bert-base-uncased on imdb (static): 33% | 1/3`
- Inside each run, a per-epoch table appears when an epoch finishes:
  ```
  Epoch  Training Loss  Validation Loss  Accuracy
  1      0.407          0.240            0.920
  2      0.298          0.292            0.922
  3      0.183          0.306            0.931
  ```
- `Evaluating Batches: 100% | 3125/3125` — runs inference on the test set
- Results are saved automatically to Google Drive in `results/tables/`

**What the warnings mean (safe to ignore):**
- `UNEXPECTED keys` / `MISSING keys` — Normal. BERT was pre-trained for a different task; the classification head is new.
- `logging_dir is deprecated` — A minor API change in newer library versions, does not affect results.
- `There were missing/unexpected keys in checkpoint` — Normal when restoring the best checkpoint at the end of training.

---

### Final Results Cell

**What it does:** Loads the saved CSV files and prints a clean summary table of all results.

**What to expect:** A formatted table showing accuracy, F1, and training time for all 36 runs.

---

## Part 5 — Handling Disconnections

Colab sessions can disconnect due to inactivity or network issues. Here is what to do:

### If Colab disconnects mid-training
1. Results that were already written to CSV **are not lost** — they are saved on Google Drive
2. Reconnect by clicking **Reconnect** in the toolbar
3. Re-run **Cell 1** (Drive mount) and **Cell 2** (install dependencies) — these must always run at session start
4. Re-run the **specific training cell** that was interrupted — it will automatically skip rows that already exist in the CSV

### If you get an "Out of Memory" error
This means the GPU ran out of VRAM. This should not happen with the current configuration, but if it does:
1. Click **Runtime → Restart session**
2. Re-run from Cell 1

---

## Part 6 — Downloading Your Results

After all experiments finish, download the result files to your computer:

1. In Google Drive, navigate to `llm_efficiency_study/results/tables/`
2. Right-click `padding_comparison_imdb.csv` → **Download**
3. Right-click `padding_comparison_emotion.csv` → **Download**
4. Copy both files into your local `results/tables/` folder

You can also download the entire `results/` folder by right-clicking on it → **Download** (it will be zipped automatically).

---

## Part 7 — Verifying Your Results

Your results should match these reference values (mean across 3 seeds):

| Model | Dataset | Strategy | Expected Accuracy |
|---|---|---|---|
| BERT-base | IMDb | Static | ~93.97% |
| BERT-base | IMDb | Dynamic | ~93.87% |
| DistilBERT | IMDb | Static | ~93.07% |
| RoBERTa-base | IMDb | Static | ~95.35% |
| BERT-base | Emotion | Static | ~92.70% |
| BERT-base | Emotion | Dynamic | ~92.83% |

Minor differences (up to ±0.3%) are expected due to hardware non-determinism on GPU.

---

## Common Problems and Solutions

| Problem | Cause | Solution |
|---|---|---|
| `ModuleNotFoundError: No module named 'src'` | Wrong folder name on Drive | Rename Drive folder to exactly `llm_efficiency_study` |
| `GPU not available` | Runtime type not set | Runtime → Change runtime type → T4 GPU |
| Session keeps disconnecting | Free tier timeout | Subscribe to Colab Pro |
| `CUDA out of memory` | Batch too large | Restart session; config already sets batch=8 |
| Results CSV is empty | Cell crashed before saving | Re-run the interrupted cell |
| `Drive not mounted` | Cell 1 not re-run after reconnect | Always re-run Cell 1 after any disconnection |
