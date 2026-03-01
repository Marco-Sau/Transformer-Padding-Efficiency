# Quick Start — Reproducing the Experiments

This guide is for users who are already familiar with Python, Jupyter notebooks, and Google Colab. For a beginner-friendly step-by-step guide, see `COLAB_SETUP.md`.

---

## Prerequisites

- A **Google account**
- A **Google Colab Pro** subscription (required — free tier times out before experiments finish)
- The project folder uploaded to **Google Drive** as `llm_efficiency_study`

---

## 1. Upload the Project to Google Drive

Upload the entire `Project/` folder to your Google Drive and rename it `llm_efficiency_study`. The expected path after upload is:

```
My Drive/
└── llm_efficiency_study/
    ├── src/
    ├── configs/
    ├── notebooks/
    ├── results/
    ├── figures/
    ├── run_experiments.py
    ├── requirements.txt
    └── ...
```

> ⚠️ The folder name **must** be `llm_efficiency_study`. The notebook uses this path to locate source files.

---

## 2. Open the Notebook in Colab

1. Go to [colab.research.google.com](https://colab.research.google.com)
2. Click **File → Open notebook → Google Drive**
3. Navigate to `llm_efficiency_study/notebooks/`
4. Open `colab_notebook_extended.ipynb`

---

## 3. Set Runtime to GPU

1. Click **Runtime → Change runtime type**
2. Set **Hardware accelerator** to `T4 GPU`
3. Click **Save**

---

## 4. Run Cells in Order

The notebook is structured in sequential sections. Run them **top to bottom**:

| Cell | Purpose | Expected time |
|---|---|---|
| **Step 1** — Drive mount & path setup | Mounts Google Drive; sets `PROJECT_ROOT` | < 1 min |
| **Step 2** — Install dependencies | Installs compatible library versions | 3–5 min |
| **Step 3** — Pre-flight test | Verifies imports and GPU availability | < 1 min |
| **IMDb: BERT-base** | 3 seeds × 2 strategies (6 runs) | ~3.5 h |
| **IMDb: DistilBERT** | 3 seeds × 2 strategies (6 runs) | ~1.7 h |
| **IMDb: RoBERTa-base** | 3 seeds × 2 strategies (6 runs) | ~3.5 h |
| **Emotion: BERT-base** | 3 seeds × 2 strategies (6 runs) | ~0.5 h |
| **Emotion: DistilBERT** | 3 seeds × 2 strategies (6 runs) | ~0.3 h |
| **Emotion: RoBERTa-base** | 3 seeds × 2 strategies (6 runs) | ~0.5 h |
| **Results summary** | Loads CSVs and prints final tables | < 1 min |

> 💡 **Each training cell is independent.** If Colab disconnects mid-run, results already written to CSV are preserved. Re-run from the cell that was interrupted.

---

## 5. Retrieve Results

Results are automatically saved to Google Drive at:
```
llm_efficiency_study/results/tables/
├── padding_comparison_imdb.csv     # 18 rows (3 models × 2 strategies × 3 seeds)
└── padding_comparison_emotion.csv  # 18 rows
```

Download these files from Google Drive to your local machine and copy them into the local `results/tables/` directory to have a complete local copy.

---

## Configuration Reference

The main experiment uses `configs/extended_imdb_config.yaml`:

```yaml
training:
  num_epochs: 3
  batch_size: 8                    # per-device (T4 GPU limit at max_length=512)
  gradient_accumulation_steps: 2   # effective batch = 16
  learning_rate: 2e-5
  warmup_steps: 500
  weight_decay: 0.01
  fp16: true                       # FP16 mixed precision
  load_best_model_at_end: true

models:
  - name: bert-base-uncased        # max_length: 512
  - name: distilbert-base-uncased  # max_length: 512
  - name: roberta-base             # max_length: 512

seeds: [42, 123, 456]
padding_strategies: [static, dynamic]
```

---

## Expected Output Format

Every completed training run appends one row to the CSV with these columns:

| Column | Description |
|---|---|
| `model_name` | e.g. `bert-base-uncased` |
| `dataset_name` | `imdb` or `emotion` |
| `padding_strategy` | `static` or `dynamic` |
| `seed` | `42`, `123`, or `456` |
| `accuracy` | Test set accuracy (0–1) |
| `f1_macro` | Macro-averaged F1 |
| `total_time_minutes` | Wall-clock training time |
| `gpu_peak_memory_mb` | Peak VRAM usage |
| `mean_latency_ms` | Per-sample inference latency |
